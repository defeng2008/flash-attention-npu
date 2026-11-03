/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "mla_tiling.h"
#include "mla_layout.h"
#include "mla_params.h"
// The unified scheduler-metadata structures. FA and MLA share one definition:
// FAInferTilingData (output record) and FAMetadataArgs (AICPU inputs).
#include "../fa_metadata_args.h"
#if defined(__CCE__)
#define MLA_METADATA_INLINE inline[host, aicore]
#else
#define MLA_METADATA_INLINE inline
#endif
namespace flash_attn::mla {
enum class MlaMetadataDtype : uint32_t {
    Fp16 = 1,
    Bf16 = 2,
};
struct MlaMetadataWide {
    uint32_t low, high;
};
struct MlaMetadataAllocation {
    uint32_t layout;
    MlaMetadataWide totalTokens;
    uint32_t maxLength;
};
struct MlaUsedMetadataContext {
    uint32_t batch = 0, heads = 0, aicNum = 0, usedFlags = 0;
    MlaMetadataAllocation query{}, kv{};
};
struct MlaPagedMetadataArray {
    uint64_t address = 0, count = 0;
};
struct MlaPagedCacheInputs {
    MlaPagedCacheLayout layout{};
    MlaPagedMetadataArray table{}, k{}, v{};
};
// Invocation geometry. Addresses, strides, scale and scratch stay in launch arguments.
// Layout: BSND=0, TND Q/KV=1, mixed=2, BSND PA=3, TND PA=4.
struct MlaContext {
    uint32_t batch, sq, sk, heads, kvHeads, splits, dtype;
    uint32_t physicalCores, margin, packed, layout, totalQ, totalK, pageSize, usedFlags, valueDim, hasSoftcap;
    MlaWindowKernelParams window;
    uint64_t workspaceBytes = 0;
};
struct MlaLayout {
    uint32_t words = 0, m = 0, n = 0, cores = 0, headGroups = 0, g = 0, capacity = 0;
    uint64_t groups = 0, scratch = 0, outOffset = 0, sumOffset = 0, maxOffset = 0, bytes = 0;
};
// Fixed semantic header, followed by B allocation/used spans, one CANN native
// partition, and B*headGroups+1 wide M prefixes. There are no format versions.
struct MlaRecordHeader {
    uint32_t words, dtype, status, batch, sq, sk, heads, kvHeads, splits, physicalCores, margin, packed, layout, totalQ,
        totalK;
    uint32_t pageSize, usedFlags, valueDim, hasSoftcap, mask, left, right, m, n, cores, headGroups;
    uint32_t queryCapacity, kvCapacity, algorithm, reserved[3];
};
static_assert(sizeof(MlaRecordHeader) == 32 * sizeof(uint32_t));
// The MLA core partition lives in FAInferTilingData (one definition). Its field
// order is unchanged, so the word offsets below stay valid.
static_assert(sizeof(MlaCorePartition) == 346 * sizeof(uint32_t));
// Offsets name the source fields; consumers never infer a development format.
enum MlaNativeWord : uint32_t {
    kMlaNativeUsedCoreNum = offsetof(MlaCorePartition, usedCoreNum) / sizeof(uint32_t),
    kMlaNativeBEnd = offsetof(MlaCorePartition, bEnd) / sizeof(uint32_t),
    kMlaNativeMEnd = offsetof(MlaCorePartition, mEnd) / sizeof(uint32_t),
    kMlaNativeNEnd = offsetof(MlaCorePartition, nEnd) / sizeof(uint32_t),
    kMlaNativeFdHeads = offsetof(MlaCorePartition, fdHeads) / sizeof(uint32_t),
    kMlaNativeMaxSplits = offsetof(MlaCorePartition, maxSplits) / sizeof(uint32_t),
    kMlaNativeVectors = offsetof(MlaCorePartition, vectors) / sizeof(uint32_t),
    kMlaNativeFdRows = offsetof(MlaCorePartition, fdRows) / sizeof(uint32_t),
    kMlaNativeFdBatch = offsetof(MlaCorePartition, fdBatch) / sizeof(uint32_t),
    kMlaNativeFdM = offsetof(MlaCorePartition, fdM) / sizeof(uint32_t),
    kMlaNativeFdSplits = offsetof(MlaCorePartition, fdSplits) / sizeof(uint32_t),
    kMlaNativeCoreSplitStart = offsetof(MlaCorePartition, coreSplitStart) / sizeof(uint32_t),
    kMlaNativeFdParts = offsetof(MlaCorePartition, fdParts) / sizeof(uint32_t),
    kMlaNativeFdLastRows = offsetof(MlaCorePartition, fdLastRows) / sizeof(uint32_t),
    kMlaNativeVectorHeadEnd = offsetof(MlaCorePartition, vectorHeadEnd) / sizeof(uint32_t),
    kMlaNativeVectorPartEnd = offsetof(MlaCorePartition, vectorPartEnd) / sizeof(uint32_t),
};

constexpr uint32_t kMlaRecordHeaderWords = sizeof(MlaRecordHeader) / 4;
constexpr uint32_t kMlaSequenceWords = 8;
constexpr uint32_t kMlaNativePartitionWords = sizeof(MlaCorePartition) / 4;
// The unified record is FAInferTilingData followed by a variable tail region.
constexpr uint32_t kMlaUnifiedFixedWords = kFaInferTilingBytes / 4;
// Word slot of the embedded MlaCorePartition inside FAInferTilingData.
constexpr uint32_t kMlaUnifiedPartitionWord = offsetof(FAInferTilingData, mlaPartition) / sizeof(uint32_t);
MLA_METADATA_INLINE bool MlaTnd(const MlaContext& c)
{
    return c.layout == 1 || c.layout == 2 || c.layout == 4;
}
MLA_METADATA_INLINE bool MlaEmptyKv(const MlaContext& c)
{
    return !c.sk || (c.layout == 1 && !c.totalK);
}
MLA_METADATA_INLINE bool MlaNativeCompressed(const MlaContext& c)
{
    return c.valueDim == 512 && !c.margin && (c.packed || c.heads == c.kvHeads);
}
MLA_METADATA_INLINE uint32_t MlaSequenceWord(uint32_t b)
{
    return kMlaRecordHeaderWords + kMlaSequenceWords * b;
}
MLA_METADATA_INLINE uint32_t MlaTail(const MlaContext& c)
{
    return MlaSequenceWord(c.batch);
}
MLA_METADATA_INLINE bool MlaRecordLayout(const MlaContext& c, MlaLayout& l, uint32_t tileM = 0, uint32_t tileN = 0)
{
    if (!c.kvHeads || !c.heads || c.heads % c.kvHeads || c.packed > 1 || (c.packed && c.heads == c.kvHeads) ||
        !c.physicalCores || c.physicalCores > 26 || c.margin >= c.physicalCores || c.layout > 4 ||
        (c.valueDim != 128 && c.valueDim != 512) || (c.valueDim == 512 && c.kvHeads != 1) ||
        ((c.layout >= 3) != bool(c.pageSize)))
        return false;
    MlaLayout r{};
    const uint32_t available = c.physicalCores - c.margin;
    r.cores = available;
    if (!r.cores || r.cores > available)
        return false;
    r.g = c.packed ? c.heads / c.kvHeads : 1;
    r.headGroups = c.heads / r.g;
    const bool tnd = MlaTnd(c);
    if (c.hasSoftcap > 1 || c.splits > 256 || c.usedFlags > 3 || c.window.kind > 2 || c.window.left < -1 ||
        c.window.right < -1 || c.window.left > INT32_MAX || c.window.right > INT32_MAX || c.sq > kMlaMaxQueryLength ||
        c.sk > kMlaMaxKvLength || c.heads > kMlaMaxQueryHeads || uint64_t(c.sq) * c.heads > kMlaMaxDenseRows ||
        (tnd ? uint64_t(c.totalQ) * c.heads > kMlaMaxBatchRows
             : uint64_t(c.batch) * c.sq > kMlaMaxBatchRows / c.heads) ||
        c.totalQ > INT32_MAX || c.totalK > INT32_MAX || (!tnd && c.totalQ) || (c.layout != 1 && c.totalK) ||
        (c.pageSize && (c.pageSize < 16 || c.pageSize > 1024 || c.pageSize % 16 || c.sk % c.pageSize)) ||
        uint64_t(c.batch) * (8 + 2ULL * r.headGroups) > UINT32_MAX - 380)
        return false;
    // Fixed CANN CalcInnerSize/CalcMBaseSize. PA alignment is logical; the
    // physical cache stride is independent of these source numerical tiles.
    uint32_t prefillN = !tnd && c.sq <= 16 ? (r.g < 5 ? 8192 : r.g < 10 ? 4096 : 2048) : 512;
    if (c.window.kind && prefillN > 1024)
        prefillN = 1024;
    if (c.pageSize && prefillN > c.pageSize && prefillN % c.pageSize)
        prefillN = prefillN / c.pageSize * c.pageSize;
    if (prefillN > c.sk)
        prefillN = c.sk;
    if (!prefillN)
        prefillN = 32;
    const uint32_t aligned = (prefillN + 31) / 32 * 32;
    const uint32_t prefillM = aligned <= 512    ? 512
                              : aligned <= 1024 ? 256
                              : aligned <= 2048 ? 128
                              : aligned <= 4096 ? 64
                                                : 32;
    r.m = tileM ? tileM : c.valueDim == 512 ? 256 : prefillM;
    r.n = tileN ? tileN : c.valueDim == 512 ? (c.sk < 512 ? c.sk : 512) : prefillN;
    if (c.valueDim == 512 && c.pageSize && r.n > c.pageSize && r.n % c.pageSize)
        r.n = r.n / c.pageSize * c.pageSize;
    if (!r.n)
        r.n = 32;
    if ((c.valueDim == 512 && (r.m < 128 || r.m > 256 || r.n > 512)) ||
        (c.valueDim == 128 && (r.m != prefillM || r.n != prefillN)))
        return false;
    const uint64_t publicCapacity = c.splits ? c.splits : 26;
    const uint64_t strideRows =
        (tnd ? uint64_t(c.totalQ ? c.totalQ : 1) : uint64_t(c.batch ? c.batch : 1) * (c.sq ? c.sq : 1)) * c.heads;
    if (strideRows > uint64_t(INT64_MAX) / 512 / publicCapacity)
        return false;
    r.words = MlaTail(c) + kMlaNativePartitionWords + 2 + 2 * c.batch * r.headGroups;
    r.capacity = c.splits ? c.splits : available;
    r.groups = uint64_t(c.batch) * r.headGroups * ((uint64_t(c.sq) * r.g + r.m - 1) / r.m);
    if (MlaTnd(c)) {
        const uint64_t compact = (uint64_t(c.totalQ) * r.g / r.m + c.batch) * r.headGroups;
        if (compact < r.groups)
            r.groups = compact;
        if (!c.totalQ)
            r.groups = 0;
    }
    if (r.groups && !MlaEmptyKv(c)) {
        if (MlaNativeCompressed(c) && c.splits <= 1) {
            const uint64_t rows = ((uint64_t(c.sq) * r.g < r.m ? uint64_t(c.sq) * r.g : r.m) + 15) / 16 * 16;
            r.scratch = 2ULL * available * (6 * rows * ((r.n + 31) / 32 * 32) + 8 * rows * 512 + 8 * r.m);
        } else {
            r.scratch = uint64_t(available) * (12ULL * r.m * ((r.n + 31) / 32 * 32) + 16ULL * r.m * c.valueDim +
                                               (c.valueDim == 512 ? 16ULL * r.m : 0));
        }
        r.outOffset = (r.scratch + 31) / 32 * 32;
        const uint64_t rows =
            MlaNativeCompressed(c) && !c.splits ? 2ULL * available * r.m : r.groups * r.capacity * r.m;
        if (rows > (uint64_t(INT64_MAX) - r.outOffset) / ((c.valueDim + 16) * 4))
            return false;
        r.sumOffset = r.outOffset + rows * c.valueDim * 4;
        r.maxOffset = r.sumOffset + rows * 8 * 4;
        const bool direct =
            c.splits == 1 && (MlaNativeCompressed(c) || (!c.margin && (c.packed || c.heads == c.kvHeads)));
        r.bytes = direct ? r.scratch : r.maxOffset + rows * 8 * 4;
    }
    l = r;
    return true;
}
MLA_METADATA_INLINE MlaRecordHeader MlaHeader(const MlaContext& c, const MlaLayout& l, uint32_t qCapacity,
                                              uint32_t kCapacity)
{
    return {l.words,
            c.dtype,
            0,
            c.batch,
            c.sq,
            c.sk,
            c.heads,
            c.kvHeads,
            c.splits,
            c.physicalCores,
            c.margin,
            c.packed,
            c.layout,
            c.totalQ,
            c.totalK,
            c.pageSize,
            c.usedFlags,
            c.valueDim,
            c.hasSoftcap,
            c.window.kind,
            uint32_t(c.window.left),
            uint32_t(c.window.right),
            l.m,
            l.n,
            l.cores,
            l.headGroups,
            qCapacity,
            kCapacity,
            0,
            {0, 0, 0}};
}
// ---- unified record <-> FAInferTilingData mapping (see docs §10) ----
// Logical header word -> word slot inside FAInferTilingData. Words 29..31 are
// reserved zeros and have no slot.
constexpr uint32_t kMlaUnifiedHeaderSlot[kMlaRecordHeaderWords] = {
    offsetof(FAInferTilingData, mlaWords) / 4,
    offsetof(FAInferTilingData, mlaDtype) / 4,
    offsetof(FAInferTilingData, mlaStatus) / 4,
    offsetof(FAInferTilingData, batch) / 4,
    offsetof(FAInferTilingData, maxQSeqlen) / 4,
    offsetof(FAInferTilingData, maxKvSeqlen) / 4,
    offsetof(FAInferTilingData, numHeads) / 4,
    offsetof(FAInferTilingData, kvHeads) / 4,
    offsetof(FAInferTilingData, numSplits) / 4,
    offsetof(FAInferTilingData, mlaPhysicalCores) / 4,
    offsetof(FAInferTilingData, mlaMargin) / 4,
    offsetof(FAInferTilingData, mlaPacked) / 4,
    offsetof(FAInferTilingData, mlaLayout) / 4,
    offsetof(FAInferTilingData, mlaTotalQ) / 4,
    offsetof(FAInferTilingData, mlaTotalK) / 4,
    offsetof(FAInferTilingData, mlaPageSize) / 4,
    offsetof(FAInferTilingData, mlaUsedFlags) / 4,
    offsetof(FAInferTilingData, mlaValueDim) / 4,
    offsetof(FAInferTilingData, mlaHasSoftcap) / 4,
    offsetof(FAInferTilingData, maskType) / 4,
    offsetof(FAInferTilingData, windowSizeLeft) / 4,
    offsetof(FAInferTilingData, windowSizeRight) / 4,
    offsetof(FAInferTilingData, mlaM) / 4,
    offsetof(FAInferTilingData, mlaN) / 4,
    offsetof(FAInferTilingData, mlaCores) / 4,
    offsetof(FAInferTilingData, mlaHeadGroups) / 4,
    offsetof(FAInferTilingData, mlaQueryCapacity) / 4,
    offsetof(FAInferTilingData, mlaKvCapacity) / 4,
    offsetof(FAInferTilingData, mlaAlgorithm) / 4,
    UINT32_MAX,
    UINT32_MAX,
    UINT32_MAX,
};
// Physical word slot of a logical record word inside the unified buffer. This
// single mapping keeps every reader (AIC and AICPU) agnostic of the split
// between the fixed FAInferTilingData record and its variable tail.
MLA_METADATA_INLINE uint32_t MlaUnifiedSlot(uint32_t batch, uint64_t word)
{
    const uint32_t spanEnd = kMlaRecordHeaderWords + kMlaSequenceWords * batch;
    if (word < kMlaRecordHeaderWords) {
        const uint32_t slot = kMlaUnifiedHeaderSlot[word];
        return slot == UINT32_MAX ? 0 : slot;
    }
    if (word < spanEnd)
        return kMlaUnifiedFixedWords + (word - kMlaRecordHeaderWords);
    if (word < spanEnd + kMlaNativePartitionWords)
        return kMlaUnifiedPartitionWord + (word - spanEnd);
    return kMlaUnifiedFixedWords + kMlaSequenceWords * batch + (word - spanEnd - kMlaNativePartitionWords);
}
// Variable tail words after the fixed record: per-batch spans + wide prefixes.
MLA_METADATA_INLINE uint32_t MlaUnifiedTailWords(const MlaLayout& l)
{
    return l.words - kMlaRecordHeaderWords - kMlaNativePartitionWords;
}
MLA_METADATA_INLINE uint64_t MlaUnifiedBytes(const MlaLayout& l)
{
    return kFaInferTilingBytes + 4ULL * MlaUnifiedTailWords(l);
}
MLA_METADATA_INLINE void WriteMlaHeader(FAInferTilingData& t, const MlaRecordHeader& header)
{
    const auto* source = reinterpret_cast<const uint32_t*>(&header);
    auto* destination = reinterpret_cast<uint32_t*>(&t);
    for (uint32_t i = 0; i < kMlaRecordHeaderWords; ++i)
        if (kMlaUnifiedHeaderSlot[i] != UINT32_MAX)
            destination[kMlaUnifiedHeaderSlot[i]] = source[i];
}
// Rebuild the invocation geometry from the unified AICPU inputs.
MLA_METADATA_INLINE MlaContext MlaContextFromArgs(const FAMetadataArgs& a)
{
    MlaContext c{};
    c.batch = a.batch;
    c.sq = a.maxQSeqlen;
    c.sk = a.mlaMaxKv;
    c.heads = a.numHeads;
    c.kvHeads = a.numHeadsK;
    c.splits = a.numSplits;
    c.dtype = a.mlaDtype;
    c.physicalCores = a.blockDim;
    c.margin = a.mlaMargin;
    c.packed = a.mlaPacked;
    c.layout = a.mlaLayoutCode;
    c.totalQ = a.mlaTotalQ;
    c.totalK = a.mlaTotalK;
    c.pageSize = a.mlaPageSize;
    c.usedFlags = a.mlaUsedFlags;
    c.valueDim = a.embeddingSizeV;
    c.hasSoftcap = a.mlaHasSoftcap;
    c.window = {a.maskType, a.windowSizeLeft, a.windowSizeRight};
    c.workspaceBytes = a.mlaWorkspaceBytes;
    return c;
}
MLA_METADATA_INLINE MlaPagedCacheInputs MlaPagedCacheInputsFromArgs(const FAMetadataArgs& a)
{
    MlaPagedCacheInputs cache{};
    // pageSize is not carried separately; CheckPagedCacheDescriptor pins it to
    // the context page size, so rebuilding from the context is equivalent.
    cache.layout.batchSize = a.mlaCacheBatchSize;
    cache.layout.maxPages = a.mlaCacheMaxPages;
    cache.layout.logicalCapacity = a.mlaCacheLogicalCapacity;
    cache.layout.physicalPages = a.mlaCachePhysicalPages;
    cache.layout.pageSize = a.mlaPageSize;
    cache.layout.tableRowStride = a.mlaCacheTableRowStride;
    cache.layout.tableSpanElements = a.mlaCacheTableSpanElements;
    cache.layout.k = {a.mlaCacheKBatchStride, a.mlaCacheKRowStride, a.mlaCacheKHeadStride, a.mlaCacheKSpanElements};
    cache.layout.v = {a.mlaCacheVBatchStride, a.mlaCacheVRowStride, a.mlaCacheVHeadStride, a.mlaCacheVSpanElements};
    cache.table = {a.mlaCacheTableAddr, a.mlaCacheTableCount};
    cache.k = {a.mlaCacheKAddr, a.mlaCacheKCount};
    cache.v = {a.mlaCacheVAddr, a.mlaCacheVCount};
    return cache;
}
// Host-side counterpart: flatten the validated context and launch data.
MLA_METADATA_INLINE void FillMlaArgs(FAMetadataArgs& a, const MlaContext& c, uint32_t queryDim, uint64_t output,
                                     uint64_t cuQuery, uint64_t cuKv, uint64_t usedQuery, uint64_t usedKv,
                                     uint64_t checked, uint32_t inferTotalQ, bool generate,
                                     const MlaPagedCacheInputs& cache)
{
    a = FAMetadataArgs{};
    a.metaOutAddr = output;
    a.embeddingSize = queryDim;
    a.embeddingSizeV = c.valueDim;
    a.batch = c.batch;
    a.numHeads = c.heads;
    a.numHeadsK = c.kvHeads;
    a.maxQSeqlen = c.sq;
    a.maskType = c.window.kind;
    a.blockDim = c.physicalCores;
    a.isVarlen = uint32_t(MlaTnd(c));
    a.isVarlenKv = uint32_t(c.layout == 1);
    a.pagedKV = uint32_t(c.pageSize != 0);
    a.numSplits = c.splits;
    a.windowSizeLeft = c.window.left;
    a.windowSizeRight = c.window.right;
    a.mlaCuQuery = cuQuery;
    a.mlaCuKv = cuKv;
    a.mlaUsedQuery = usedQuery;
    a.mlaUsedKv = usedKv;
    a.mlaChecked = checked;
    a.mlaWorkspaceBytes = c.workspaceBytes;
    a.mlaLayoutCode = c.layout;
    a.mlaTotalQ = c.totalQ;
    a.mlaTotalK = c.totalK;
    a.mlaPageSize = c.pageSize;
    a.mlaUsedFlags = c.usedFlags;
    a.mlaMaxKv = c.sk;
    a.mlaPacked = c.packed;
    a.mlaMargin = c.margin;
    a.mlaInferTotalQ = inferTotalQ;
    a.mlaGenerate = uint32_t(generate);
    a.mlaDtype = c.dtype;
    a.mlaHasSoftcap = c.hasSoftcap;
    a.mlaCacheBatchSize = uint32_t(cache.layout.batchSize);
    a.mlaCacheMaxPages = uint32_t(cache.layout.maxPages);
    a.mlaCacheLogicalCapacity = uint32_t(cache.layout.logicalCapacity);
    a.mlaCachePhysicalPages = uint32_t(cache.layout.physicalPages);
    a.mlaCacheTableAddr = cache.table.address;
    a.mlaCacheTableCount = cache.table.count;
    a.mlaCacheTableRowStride = cache.layout.tableRowStride;
    a.mlaCacheTableSpanElements = cache.layout.tableSpanElements;
    a.mlaCacheKAddr = cache.k.address;
    a.mlaCacheKCount = cache.k.count;
    a.mlaCacheKBatchStride = cache.layout.k.batchStride;
    a.mlaCacheKRowStride = cache.layout.k.rowStride;
    a.mlaCacheKHeadStride = cache.layout.k.headStride;
    a.mlaCacheKSpanElements = cache.layout.k.spanElements;
    a.mlaCacheVAddr = cache.v.address;
    a.mlaCacheVCount = cache.v.count;
    a.mlaCacheVBatchStride = cache.layout.v.batchStride;
    a.mlaCacheVRowStride = cache.layout.v.rowStride;
    a.mlaCacheVHeadStride = cache.layout.v.headStride;
    a.mlaCacheVSpanElements = cache.layout.v.spanElements;
}
bool BuildNonquantMetadata(const FAMetadataArgs&);
} // namespace flash_attn::mla
#undef MLA_METADATA_INLINE
