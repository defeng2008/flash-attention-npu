# Copyright (c) 2026, Minghua Shen.
"""``flash_attn_with_kvcache`` BNSD-input path: the measured slot of the layout comparison.

The harness hands in contiguous BSND storage, so the equivalent BNSD storage is materialised
**once** here and only re-viewed on later calls (see the sibling ``func`` slot for why:
re-reading the same memory as BNSD describes the transposed problem, and converting on every
call would fold a caller-side permute copy into the measurement).

The BNSD branch of the 950 interface requires ``page_table is None`` and
``max(cache_seqlens) <= k_cache.shape[1]`` in BSND terms -- both hold here.  It also requires
``S_Q > 1``: the branch is chosen from strides only, and a single-token BNSD query view is
stride-identical to contiguous BSND.  ``cache_seqlens`` is built once and kept on the module so
the host-side allocation stays out of the timed region.
"""

import math

import torch
import torch.nn as nn

from flash_attn_npu_3 import flash_attn_with_kvcache

B, H, KV_H, S_Q, S_KV, D = 1, 64, 64, 128, 4096, 128
SCALE = 1.0 / math.sqrt(D)
CAUSAL = True
CACHE_SEQLENS = S_KV


class ModelNew(nn.Module):
    """The same logical query/cache as ``model.py``, stored contiguously in BNSD."""

    def __init__(self):
        super().__init__()
        self._seqlens = None
        self._bnsd = None

    def forward(self, q, k_cache, v_cache):
        if self._seqlens is None:
            self._seqlens = torch.full((B,), CACHE_SEQLENS, dtype=torch.int32, device=q.device)
        if self._bnsd is None:
            self._bnsd = tuple(t.transpose(1, 2).contiguous() for t in (q, k_cache, v_cache))
        q, k_cache, v_cache = (t.transpose(1, 2) for t in self._bnsd)
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=self._seqlens,
            softmax_scale=SCALE,
            causal=CAUSAL,
        )
