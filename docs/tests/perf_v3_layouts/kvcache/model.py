# Copyright (c) 2026, Minghua Shen.
"""``flash_attn_with_kvcache`` BSND-input path: the reference slot of the layout comparison.

Decode-with-chunk case: a ``S_Q = 128`` query against a full ``S_KV = 4096`` cache.  No ``k``
/ ``v`` is passed (no append-KV) and no ``page_table`` is used, which keeps the BNSD path
selectable in the sibling file.

``S_Q`` is not 1 on purpose: the 950 interface selects BNSD from strides alone, and a
single-token BNSD query view is stride-identical to contiguous BSND, so the decoder shape
(S_Q = 1) can never reach the BNSD branch.  See ``layout_comparison.md``.

``cache_seqlens`` is passed explicitly rather than relying on the default.  The tensor is
built once and kept on the module so the host-side allocation stays out of the timed region.
"""

import math

import torch
import torch.nn as nn

from flash_attn_npu_3 import flash_attn_with_kvcache

B, H, KV_H, S_Q, S_KV, D = 1, 64, 64, 128, 4096, 128
DTYPE = torch.float16
SCALE = 1.0 / math.sqrt(D)
CAUSAL = True
CACHE_SEQLENS = S_KV


def _rand_bsnd(heads, seqlen, generator):
    return torch.randn(B, seqlen, heads, D, dtype=DTYPE, generator=generator)


def get_input_groups():
    """One positional ``(q, k_cache, v_cache)`` group: the harness calls ``model(*group)``."""
    generator = torch.Generator().manual_seed(2026)
    return [
        [
            _rand_bsnd(H, S_Q, generator),
            _rand_bsnd(KV_H, S_KV, generator),
            _rand_bsnd(KV_H, S_KV, generator),
        ]
    ]


class Model(nn.Module):
    """Contiguous BSND query and cache, forwarded as-is."""

    def __init__(self):
        super().__init__()
        self._seqlens = None

    def forward(self, q, k_cache, v_cache):
        if self._seqlens is None:
            self._seqlens = torch.full((B,), CACHE_SEQLENS, dtype=torch.int32, device=q.device)
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=self._seqlens,
            softmax_scale=SCALE,
            causal=CAUSAL,
        )
