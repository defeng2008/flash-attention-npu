/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 */

#ifndef FA_METADATA_ARGS_H
#define FA_METADATA_ARGS_H

#include <cstdint>
#include <type_traits>
#include "tilingdata.h"

static_assert(std::is_trivially_copyable<FAInferStaticTilingData>::value, "static tiling must be trivially copyable");
static_assert(std::is_standard_layout<FAInferStaticTilingData>::value, "static tiling must have a stable layout");
static_assert(sizeof(FAInferStaticTilingData) == 3816, "static metadata ABI changed; update Python fake size");
static_assert(std::is_trivially_copyable<FAInferRuntimeData>::value, "runtime tiling must be trivially copyable");
static_assert(std::is_standard_layout<FAInferRuntimeData>::value, "runtime tiling must have a stable layout");
static_assert(sizeof(FAInferRuntimeData) == 96, "runtime tiling ABI changed");

namespace fa_metadata {
constexpr uint32_t MASK_DIM = 2048;
constexpr uint64_t MASK_BYTES = static_cast<uint64_t>(MASK_DIM) * MASK_DIM;

// scheduler_metadata is static-tiling-only. The 2048x2048 compressed triu(1)
// mask is generated locally on the device by MakeDeviceTriuMask when required.
inline uint64_t TilingOffset(bool /*has_mask*/)
{
    return 0;
}
inline uint64_t MetadataBytes(bool /*has_mask*/)
{
    return sizeof(FAInferStaticTilingData);
}

constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = static_cast<uint64_t>(128) * 512;
constexpr uint64_t PRELAUNCH_NUM = 3;
inline uint64_t Mm1OutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM;
}
inline uint64_t SmOnlineOutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * 2 * PRELAUNCH_NUM;
}
inline uint64_t Mm2OutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM;
}
inline uint64_t UpdateOutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM;
}
inline uint64_t WorkSpaceSize(uint64_t blockDim)
{
    return Mm1OutSize(blockDim) + SmOnlineOutSize(blockDim) + Mm2OutSize(blockDim) + UpdateOutSize(blockDim);
}
} // namespace fa_metadata

// Static-tiling-only metadata arguments. Per-call values (page/block sizes,
// scale/softcap, mask/window) are carried in FAInferRuntimeData and are never
// serialized into the reusable scheduler metadata.
struct FAMetadataArgs {
    uint64_t seqlensQAddr;
    uint64_t cacheSeqlensAddr;
    uint64_t metaOutAddr;
    uint32_t batch;
    uint32_t numHeads;
    uint32_t numHeadsK;
    uint32_t embeddingSize;
    uint32_t embeddingSizeV;
    uint32_t maxQSeqlen;
    uint32_t blockDim;
    uint32_t isVarlen;
    uint32_t isVarlenKv;
    uint32_t numSplits;
    // 1 when the call is sliding-window / local attention after host window
    // rewrite. FD must stay off: SWA valid keys sit in the last KV stacks, and
    // split-KV combine of the empty prefix tiles is wrong. Matches host tiling.
    uint32_t isLocal;
    // ---- appended MLA inputs (unused/zero for ordinary FA) ----
    // Common quantities reuse the fields above: batch/numHeads/numHeadsK/
    // embeddingSize(=Dq)/embeddingSizeV(=Dv)/maxQSeqlen(=sq)/blockDim(=physical
    // cores)/numSplits(=splits)/metaOutAddr. FA4 keeps mask/window in
    // FAInferRuntimeData, so MLA carries its own mlaMask/mlaLeft/mlaRight.
    // MLA sk is carried by mlaMaxKv and stored in the output's maxKvSeqlen.
    uint64_t mlaCuQuery, mlaCuKv, mlaUsedQuery, mlaUsedKv, mlaChecked, mlaWorkspaceBytes;
    uint32_t mlaLayoutCode, mlaTotalQ, mlaTotalK, mlaPageSize, mlaUsedFlags;
    uint32_t mlaMaxKv, mlaPacked, mlaMargin, mlaInferTotalQ, mlaGenerate, mlaDtype, mlaHasSoftcap;
    uint32_t mlaMask, mlaLeft, mlaRight;
    // Flattened MlaPagedCacheInputs (layout + page table + K + V descriptors).
    uint32_t mlaCacheBatchSize, mlaCacheMaxPages, mlaCacheLogicalCapacity, mlaCachePhysicalPages;
    uint64_t mlaCacheTableAddr, mlaCacheTableCount, mlaCacheTableRowStride, mlaCacheTableSpanElements;
    uint64_t mlaCacheKAddr, mlaCacheKCount, mlaCacheKBatchStride, mlaCacheKRowStride, mlaCacheKHeadStride,
        mlaCacheKSpanElements;
    uint64_t mlaCacheVAddr, mlaCacheVCount, mlaCacheVBatchStride, mlaCacheVRowStride, mlaCacheVHeadStride,
        mlaCacheVSpanElements;
};

// CPU-computed launch payload. AICPU writes only this small, private runtime
// buffer; reusable static metadata is neither read nor modified. Writing it
// through AICPU (rather than a synchronous host->device memcpy) keeps the
// explicit-metadata forward path capture-safe under NPU graph capture.
struct FARuntimeArgs {
    uint64_t runtimeAddr;
    FAInferRuntimeData runtime;
};

#endif
