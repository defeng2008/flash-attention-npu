/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Allocation snapshots and CANN SplitCore are shared by every MLA layout.
#include "mla_metadata.h"
#include "mla_split.h"
#include "mla_metadata_consumer.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <vector>

namespace flash_attn::mla {
bool BuildNonquantMetadata(const MlaRecordArgs& a)
{
    auto c = a.context;
    MlaLayout l{};
    if (!MlaRecordLayout(c, l) || !a.output || a.output % 4 || a.words != l.words ||
        a.words > (UINT64_MAX - a.output) / 4)
        return false;
    const uint64_t addresses[] = {a.cuQuery, a.cuKv, a.usedQuery, a.usedKv};
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t address = addresses[axis], count = uint64_t(c.batch) + (axis < 2);
        if (address && count && address < a.output + uint64_t(a.words) * 4 &&
            (address >= a.output || count > (a.output - address) / 4))
            return false;
    }
    auto* out = reinterpret_cast<uint32_t*>(a.output);
    std::fill(out, out + a.words, 0);
    out[0] = a.words;
    out[1] = c.dtype;
    out[2] = 1;
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t address = addresses[axis], count = uint64_t(c.batch) + (axis < 2);
        const bool required = axis == 0 ? MlaTnd(c) : axis == 1 ? c.layout == 1 : c.usedFlags & (1U << (axis - 2));
        if ((required && count && !address) || (!required && address) || address % 4 ||
            (address && count > (UINT64_MAX - address) / 4))
            return true;
    }
    if ((c.dtype != uint32_t(MlaMetadataDtype::Fp16) && c.dtype != uint32_t(MlaMetadataDtype::Bf16)) ||
        a.inferTotalQ > 1 || (a.inferTotalQ && c.layout != 2 && c.layout != 4))
        return true;
    if (a.inferTotalQ) {
        c.totalQ = reinterpret_cast<const uint32_t*>(a.cuQuery)[c.batch];
        if (!MlaRecordLayout(c, l))
            return true;
    }
    try {
        const MlaUsedShape shape{c.batch,
                                 c.heads,
                                 {MlaTnd(c) ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd,
                                  MlaTnd(c) ? int64_t(c.totalQ) : int64_t(c.batch) * c.sq, c.sq},
                                 {c.layout == 1 ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd,
                                  c.layout == 1 ? int64_t(c.totalK) : int64_t(c.batch) * c.sk, c.sk}};
        MlaUsedInputs inputs{};
        MlaLengthArray* arrays[] = {&inputs.cuQuery, &inputs.cuKv, &inputs.usedQuery, &inputs.usedKv};
        for (uint32_t i = 0; i < 4; ++i)
            *arrays[i] = {reinterpret_cast<const int32_t*>(addresses[i]),
                          addresses[i] ? uint64_t(c.batch) + (i < 2) : 0};
        MlaUsedLengths lengths;
        if (ParseUsedLengths(shape, inputs, lengths) != MlaScheduleStatus::Success)
            return true;
        MlaPlatformInfo physical{};
        physical.aicNum = c.physicalCores;
        physical.aivNum = 2 * c.physicalCores;
        physical.localMemory = {516096, 196608, 65536, 65536, 131072};
        for (uint32_t i = 0; i < c.physicalCores; ++i)
            physical.blockDims[i] = i + 1;
        uint32_t queryCapacity = MlaTnd(c) ? lengths.maxQueryCapacity : c.sq;
        uint32_t kvCapacity = c.layout == 1 ? lengths.maxKvCapacity : c.sk;
        if (c.pageSize)
            kvCapacity = (lengths.maxKvLength + c.pageSize - 1) / c.pageSize * c.pageSize;
        if (MlaNativeCompressed(c)) {
            MlaSchedule geometry{};
            if (PrepareMlaGeometry(shape, physical, inputs, lengths, c.pageSize, geometry) !=
                MlaScheduleStatus::Success)
                return true;
            if (!MlaRecordLayout(c, l, geometry.mBaseSize, geometry.s2BaseSize))
                return true;
        }
        const MlaWindowConfig window{c.sq, c.sk, c.window.kind == 1 ? -1 : c.window.left,
                                     c.window.kind == 1 ? -1 : c.window.right, c.window.kind == 1};
        MlaNativeSplitKvSchedule native{};
        if (MlaNativeCompressed(c) && c.splits == 1) {
            std::vector<MlaCausalSequence> sequences;
            sequences.reserve(c.batch);
            for (const auto& s : lengths.sequences)
                sequences.push_back({s.queryLength, s.kvLength});
            MlaCausalSchedule whole{};
            if (BuildWindowSchedule({sequences, l.g, l.m, l.n, c.physicalCores, 2}, window, whole) !=
                MlaCausalStatus::Success)
                return true;
            native.usedCoreNum = whole.usedCoreNum;
            native.bN2End = whole.bN2End;
            native.gS1End = whole.gS1End;
            native.s2End = whole.s2End;
        } else {
            if (BuildGqaSchedule({shape, c.kvHeads, c.valueDim, l.m, l.n, c.pageSize}, lengths, window,
                                 {c.packed, c.margin}, physical, native) != MlaScheduleStatus::Success)
                return true;
        }
        std::vector<uint32_t> result(l.words, 0);
        const auto header = MlaHeader(c, l, queryCapacity, kvCapacity);
        std::memcpy(result.data(), &header, sizeof(header));
        auto* tail = result.data() + MlaTail(c);
        auto* prefix = tail + kMlaNativePartitionWords;
        uint64_t groups = 0;
        for (uint32_t b = 0; b < c.batch; ++b) {
            const auto& s = lengths.sequences[b];
            const uint32_t span[] = {uint32_t(s.queryStart), uint32_t(s.queryStart >> 32),
                                     uint32_t(s.kvStart),    uint32_t(s.kvStart >> 32),
                                     s.queryCapacity,        s.kvCapacity,
                                     s.queryLength,          s.kvLength};
            std::copy(span, span + 8, result.begin() + MlaSequenceWord(b));
            const uint64_t count = (uint64_t(s.queryLength) * l.g + l.m - 1) / l.m;
            for (uint32_t h = 0; h < l.headGroups; ++h) {
                groups += count;
                prefix += 2;
                prefix[0] = uint32_t(groups);
                prefix[1] = uint32_t(groups >> 32);
            }
        }
        MlaNativePartition partition{};
        partition.mask = c.window.kind;
        partition.left = uint32_t(c.window.left);
        partition.right = uint32_t(c.window.right);
        partition.usedCoreNum = native.usedCoreNum;
        std::copy(native.bN2End.begin(), native.bN2End.end(), partition.bEnd);
        std::copy(native.gS1End.begin(), native.gS1End.end(), partition.mEnd);
        std::copy(native.s2End.begin(), native.s2End.end(), partition.nEnd);
        const auto& fd = native.fd;
        partition.fdHeads = fd.numOfFdHead;
        partition.maxSplits = fd.maxS2SplitNum;
        partition.vectors = fd.usedVecNumOfFd;
        partition.fdRows = 8;
#define MLA_COPY_FIELD(dst, src) std::copy(fd.src.begin(), fd.src.end(), partition.dst)
        MLA_COPY_FIELD(fdBatch, bN2IdxOfFdHead);
        MLA_COPY_FIELD(fdM, gS1IdxOfFdHead);
        MLA_COPY_FIELD(fdSplits, s2SplitNumOfFdHead);
        MLA_COPY_FIELD(coreSplitStart, s2SplitStartIdxOfCore);
        MLA_COPY_FIELD(fdParts, gS1SplitNumOfFdHead);
        MLA_COPY_FIELD(fdLastRows, gS1LastPartSizeOfFdHead);
        MLA_COPY_FIELD(vectorHeadEnd, gS1IdxEndOfFdHead);
        MLA_COPY_FIELD(vectorPartEnd, gS1IdxEndOfFdHeadSplit);
#undef MLA_COPY_FIELD
        std::memcpy(tail, &partition, sizeof(partition));
        std::copy(result.begin(), result.end(), out);
    } catch (const std::bad_alloc&) {
    }
    return true;
}
} // namespace flash_attn::mla
