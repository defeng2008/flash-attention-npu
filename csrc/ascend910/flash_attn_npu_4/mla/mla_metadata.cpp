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
bool BuildNonquantMetadata(const FAMetadataArgs& a)
{
    auto c = MlaContextFromArgs(a);
    MlaLayout l{};
    if (!MlaRecordLayout(c, l) || !a.metaOutAddr || a.metaOutAddr % 4)
        return false;
    const uint64_t bytes = MlaUnifiedBytes(l);
    if (a.metaOutAddr > UINT64_MAX - bytes)
        return false;
    const uint64_t addresses[] = {a.mlaCuQuery, a.mlaCuKv, a.mlaUsedQuery, a.mlaUsedKv};
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t address = addresses[axis], count = uint64_t(c.batch) + (axis < 2);
        if (address && count && address < a.metaOutAddr + bytes &&
            (address >= a.metaOutAddr || count > (a.metaOutAddr - address) / 4))
            return false;
    }
    auto* out = reinterpret_cast<uint32_t*>(a.metaOutAddr);
    std::memset(out, 0, bytes);
    out[kMlaUnifiedHeaderSlot[0]] = l.words;
    out[kMlaUnifiedHeaderSlot[1]] = c.dtype;
    out[kMlaUnifiedHeaderSlot[2]] = 1;
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t address = addresses[axis], count = uint64_t(c.batch) + (axis < 2);
        const bool required = axis == 0 ? MlaTnd(c) : axis == 1 ? c.layout == 1 : c.usedFlags & (1U << (axis - 2));
        if ((required && count && !address) || (!required && address) || address % 4 ||
            (address && count > (UINT64_MAX - address) / 4))
            return true;
    }
    // bits 0/1 ask the device to infer the query/KV totals from the trailing
    // element of the packed prefix, so no host-side D2H read is required.
    if ((c.dtype != uint32_t(MlaMetadataDtype::Fp16) && c.dtype != uint32_t(MlaMetadataDtype::Bf16)) ||
        a.mlaInferTotalQ > 3 || ((a.mlaInferTotalQ & 1U) && !MlaTnd(c)) ||
        ((a.mlaInferTotalQ & 2U) && c.layout != 1))
        return true;
    if (a.mlaInferTotalQ) {
        if (a.mlaInferTotalQ & 1U)
            c.totalQ = reinterpret_cast<const uint32_t*>(a.mlaCuQuery)[c.batch];
        if (a.mlaInferTotalQ & 2U)
            c.totalK = reinterpret_cast<const uint32_t*>(a.mlaCuKv)[c.batch];
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
        auto* tiling = reinterpret_cast<FAInferStaticTilingData*>(a.metaOutAddr);
        auto* tail = out + kMlaUnifiedFixedWords;
        auto* prefix = tail + kMlaSequenceWords * c.batch;
        uint64_t groups = 0;
        for (uint32_t b = 0; b < c.batch; ++b) {
            const auto& s = lengths.sequences[b];
            const uint32_t span[] = {uint32_t(s.queryStart), uint32_t(s.queryStart >> 32),
                                     uint32_t(s.kvStart),    uint32_t(s.kvStart >> 32),
                                     s.queryCapacity,        s.kvCapacity,
                                     s.queryLength,          s.kvLength};
            std::copy(span, span + 8, tail + kMlaSequenceWords * b);
            const uint64_t count = (uint64_t(s.queryLength) * l.g + l.m - 1) / l.m;
            for (uint32_t h = 0; h < l.headGroups; ++h) {
                groups += count;
                prefix += 2;
                prefix[0] = uint32_t(groups);
                prefix[1] = uint32_t(groups >> 32);
            }
        }
        MlaCorePartition partition{};
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
        std::memcpy(&tiling->mlaPartition, &partition, sizeof(partition));
        // Written last: this clears the in-progress status word on success.
        WriteMlaHeader(*tiling, MlaHeader(c, l, queryCapacity, kvCapacity));
    } catch (const std::bad_alloc&) {
    }
    return true;
}
} // namespace flash_attn::mla
