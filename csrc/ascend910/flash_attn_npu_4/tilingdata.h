/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

#include <cstddef>
#include <cstdint>

struct coreNode {
    int startBIdx;
    int startN1Idx;
    int startS1Idx;
    int startS2Idx;
    int endBIdx;
    int endN1Idx;
    int endS1Idx;
    int endS2Idx;
    int64_t firstSplitKVTaskLseOffset;
    int64_t firstSplitKVTaskOOffset;
};

struct splitNode {
    int batchIdx;
    int headStartIdx;
    int headEndIdx;
    int qStartIdx;
    int qEndIdx;
    int splitNum;
    int64_t lseTaskOffset;
    int64_t oTaskOffset;
};

// MLA core partition, appended to FAInferStaticTilingData. Field order is
// identical to the former MlaNativePartition so consumer word offsets stay
// valid.
struct MlaCorePartition {
    uint32_t mask, left, right, usedCoreNum;
    uint32_t bEnd[26], mEnd[26], nEnd[26];
    uint32_t fdHeads, maxSplits, vectors, fdRows;
    uint32_t fdBatch[26], fdM[26], fdSplits[26], coreSplitStart[26], fdParts[26], fdLastRows[26];
    uint32_t vectorHeadEnd[52], vectorPartEnd[52];
};

// Read-only scheduling data shared by all calls using the same metadata.
struct FAInferStaticTilingData {
    uint32_t numHeads;
    uint32_t embeddingSize;
    uint32_t embeddingSizeV;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t kvHeads;
    uint32_t batch;
    uint32_t firstBatchTaskNum;
    uint32_t totalTaskNum;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t UpdateSize;
    uint64_t workSpaceSize;
    uint64_t padding1;
    uint64_t padding2;
    uint32_t padding3;
    uint64_t splitLseTotalSize;
    uint64_t splitOTotalSize;
    uint32_t totalSplitNodeNum;
    uint32_t needCoreNum;
    uint32_t flashDecodeFlag;
    uint32_t numSplits;
    coreNode coreInfo[25];
    splitNode splitInfo[25];

    // ---- appended MLA schedule (all zero for ordinary FA) ----
    // Words 3..8 of the MLA header reuse batch/maxQSeqlen/maxKvSeqlen/numHeads/
    // kvHeads/numSplits; words 19..21 use the dedicated mlaMask/mlaLeft/mlaRight
    // below, since FA4 keeps mask/window in FAInferRuntimeData instead.
    uint32_t mlaWords, mlaDtype, mlaStatus, mlaLayout, mlaPacked, mlaMargin, mlaPhysicalCores;
    uint32_t mlaTotalQ, mlaTotalK, mlaPageSize, mlaUsedFlags, mlaValueDim, mlaHasSoftcap;
    uint32_t mlaM, mlaN, mlaCores, mlaHeadGroups, mlaQueryCapacity, mlaKvCapacity, mlaAlgorithm;
    uint32_t mlaMask, mlaLeft, mlaRight;
    MlaCorePartition mlaPartition;

    uint32_t get_numHeads() const
    {
        return numHeads;
    }
    uint32_t get_embeddingSize() const
    {
        return embeddingSize;
    }
    uint32_t get_embeddingSizeV() const
    {
        return embeddingSizeV;
    }
    uint32_t get_maxQSeqlen() const
    {
        return maxQSeqlen;
    }
    uint32_t get_maxKvSeqlen() const
    {
        return maxKvSeqlen;
    }
    uint32_t get_kvHeads() const
    {
        return kvHeads;
    }
    uint32_t get_batch() const
    {
        return batch;
    }
    uint32_t get_firstBatchTaskNum() const
    {
        return firstBatchTaskNum;
    }
    uint32_t get_totalTaskNum() const
    {
        return totalTaskNum;
    }
    uint64_t get_mm1OutSize() const
    {
        return mm1OutSize;
    }
    uint64_t get_smOnlineOutSize() const
    {
        return smOnlineOutSize;
    }
    uint64_t get_mm2OutSize() const
    {
        return mm2OutSize;
    }
    uint64_t get_UpdateSize() const
    {
        return UpdateSize;
    }
    uint64_t get_workSpaceSize() const
    {
        return workSpaceSize;
    }
    uint64_t get_padding1() const
    {
        return padding1;
    }
    uint64_t get_padding2() const
    {
        return padding2;
    }
    uint32_t get_padding3() const
    {
        return padding3;
    }
    uint64_t get_splitLseTotalSize() const
    {
        return splitLseTotalSize;
    }
    uint64_t get_splitOTotalSize() const
    {
        return splitOTotalSize;
    }
    uint32_t get_totalSplitNodeNum() const
    {
        return totalSplitNodeNum;
    }
    uint32_t get_needCoreNum() const
    {
        return needCoreNum;
    }
    uint32_t get_flashDecodeFlag() const
    {
        return flashDecodeFlag;
    }
    uint32_t get_numSplits() const
    {
        return numSplits;
    }

    void set_numHeads(uint32_t value)
    {
        numHeads = value;
    }
    void set_embeddingSize(uint32_t value)
    {
        embeddingSize = value;
    }
    void set_embeddingSizeV(uint32_t value)
    {
        embeddingSizeV = value;
    }
    void set_maxQSeqlen(uint32_t value)
    {
        maxQSeqlen = value;
    }
    void set_maxKvSeqlen(uint32_t value)
    {
        maxKvSeqlen = value;
    }
    void set_kvHeads(uint32_t value)
    {
        kvHeads = value;
    }
    void set_batch(uint32_t value)
    {
        batch = value;
    }
    void set_firstBatchTaskNum(uint32_t value)
    {
        firstBatchTaskNum = value;
    }
    void set_totalTaskNum(uint32_t value)
    {
        totalTaskNum = value;
    }
    void set_mm1OutSize(uint64_t value)
    {
        mm1OutSize = value;
    }
    void set_smOnlineOutSize(uint64_t value)
    {
        smOnlineOutSize = value;
    }
    void set_mm2OutSize(uint64_t value)
    {
        mm2OutSize = value;
    }
    void set_UpdateSize(uint64_t value)
    {
        UpdateSize = value;
    }
    void set_workSpaceSize(uint64_t value)
    {
        workSpaceSize = value;
    }
    void set_padding1(uint64_t value)
    {
        padding1 = value;
    }
    void set_padding2(uint64_t value)
    {
        padding2 = value;
    }
    void set_padding3(uint32_t value)
    {
        padding3 = value;
    }
    void set_splitLseTotalSize(uint64_t value)
    {
        splitLseTotalSize = value;
    }
    void set_splitOTotalSize(uint64_t value)
    {
        splitOTotalSize = value;
    }
    void set_totalSplitNodeNum(uint32_t value)
    {
        totalSplitNodeNum = value;
    }
    void set_needCoreNum(uint32_t value)
    {
        needCoreNum = value;
    }
    void set_flashDecodeFlag(uint32_t value)
    {
        flashDecodeFlag = value;
    }
    void set_numSplits(uint32_t value)
    {
        numSplits = value;
    }
};

// Pin the ABI: existing FA fields must not move, MLA payload length is fixed.
// The struct is 8-byte aligned (u64/int64 members), so the 3812-byte MLA
// payload is rounded up by 4 bytes of trailing padding; kFaInferTilingBytes
// follows sizeof and therefore includes that padding.
static_assert(offsetof(FAInferStaticTilingData, mlaWords) == 2336,
              "MLA schedule must be appended without moving existing FA fields");
static_assert(sizeof(MlaCorePartition) == 346 * sizeof(uint32_t), "MlaCorePartition layout changed");
static_assert(sizeof(FAInferStaticTilingData) == 3816, "FAInferStaticTilingData ABI changed");
constexpr uint64_t kFaInferTilingBytes = sizeof(FAInferStaticTilingData);

// Per-forward execution values; never stored in reusable scheduler metadata.
struct FAInferRuntimeData {
    uint32_t numBlocks;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t maskType;
    float scaleValue;
    float softcapValue;
    int64_t windowSizeLeft = 0;
    int64_t windowSizeRight = 0;
    // Element strides for paged KV. The API receives logical
    // [page, token, head, dim] tensors; a transposed view can retain physical
    // BNBD storage, whose token/head strides identify the inner layout.
    uint64_t keyPageStride;
    uint64_t valuePageStride;
    uint64_t keyTokenStride;
    uint64_t valueTokenStride;
    uint64_t keyHeadStride;
    uint64_t valueHeadStride;
    // TND LSE is (num_heads, q.size(0)). q.size(0) may exceed cu_seqlens_q[-1];
    // the extra tail is unused storage and this stride keeps head rows apart.
    uint64_t totalQTokens;

    uint32_t get_numBlocks() const
    {
        return numBlocks;
    }
    uint32_t get_blockSize() const
    {
        return blockSize;
    }
    uint32_t get_maxNumBlocksPerBatch() const
    {
        return maxNumBlocksPerBatch;
    }
    uint32_t get_maskType() const
    {
        return maskType;
    }
    float get_scaleValue() const
    {
        return scaleValue;
    }
    float get_softcapValue() const
    {
        return softcapValue;
    }
    int64_t get_windowSizeLeft() const
    {
        return windowSizeLeft;
    }
    int64_t get_windowSizeRight() const
    {
        return windowSizeRight;
    }
    uint64_t get_keyPageStride() const
    {
        return keyPageStride;
    }
    uint64_t get_valuePageStride() const
    {
        return valuePageStride;
    }
    uint64_t get_keyTokenStride() const
    {
        return keyTokenStride;
    }
    uint64_t get_valueTokenStride() const
    {
        return valueTokenStride;
    }
    uint64_t get_keyHeadStride() const
    {
        return keyHeadStride;
    }
    uint64_t get_valueHeadStride() const
    {
        return valueHeadStride;
    }
    uint64_t get_totalQTokens() const
    {
        return totalQTokens;
    }

    void set_numBlocks(uint32_t value)
    {
        numBlocks = value;
    }
    void set_blockSize(uint32_t value)
    {
        blockSize = value;
    }
    void set_maxNumBlocksPerBatch(uint32_t value)
    {
        maxNumBlocksPerBatch = value;
    }
    void set_maskType(uint32_t value)
    {
        maskType = value;
    }
    void set_scaleValue(float value)
    {
        scaleValue = value;
    }
    void set_softcapValue(float value)
    {
        softcapValue = value;
    }
    void set_windowSizeLeft(int64_t value)
    {
        windowSizeLeft = value;
    }
    void set_windowSizeRight(int64_t value)
    {
        windowSizeRight = value;
    }
    void set_keyPageStride(uint64_t value)
    {
        keyPageStride = value;
    }
    void set_valuePageStride(uint64_t value)
    {
        valuePageStride = value;
    }
    void set_keyTokenStride(uint64_t value)
    {
        keyTokenStride = value;
    }
    void set_valueTokenStride(uint64_t value)
    {
        valueTokenStride = value;
    }
    void set_keyHeadStride(uint64_t value)
    {
        keyHeadStride = value;
    }
    void set_valueHeadStride(uint64_t value)
    {
        valueHeadStride = value;
    }
    void set_totalQTokens(uint64_t value)
    {
        totalQTokens = value;
    }
};

// CPU tiler convenience aggregate only. Serialize the two bases separately;
// this aggregate is not a device ABI and is never passed to a kernel.
struct FAInferTilingData : FAInferStaticTilingData, FAInferRuntimeData {};

#endif
