// Copyright (c) 2026, Minghua Shen.

#ifndef MLA_DECODE_PERF_H
#define MLA_DECODE_PERF_H

#include <cstdint>

namespace MlaDecodePerf {

constexpr uint32_t kSeqlenQ = 1;
constexpr uint32_t kQHeads = 32;
constexpr uint32_t kKvHeads = 1;
constexpr uint32_t kQHeadDim = 64;
constexpr uint32_t kVHeadDim = 512;
constexpr uint32_t kQnTileCap = 16;
constexpr uint32_t kKvStackLen = 384;

static inline bool MatchesShape(int64_t seqlenQ, int64_t qHeads, int64_t kvHeads, int64_t qHeadDim, int64_t vHeadDim)
{
    return seqlenQ == kSeqlenQ && qHeads == kQHeads && kvHeads == kKvHeads && qHeadDim == kQHeadDim &&
           vHeadDim == kVHeadDim;
}

} // namespace MlaDecodePerf

#endif
