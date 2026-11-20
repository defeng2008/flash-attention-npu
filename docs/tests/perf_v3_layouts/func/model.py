# Copyright (c) 2026, Minghua Shen.
"""``flash_attn_func`` BSND-input path: the reference slot of the layout comparison.

msprof compares ``model.py`` (``class Model``, reported as "Ref") against
``model_new_ascendc.py`` (``class ModelNew``, "Asc") with ``speedup = ref_us / asc_us``.
Ref is the BSND call here and Asc the BNSD one, so ``speedup > 1`` means the BNSD
input path is the faster of the two.

Both slots receive the very same contiguous storage from ``get_input_groups()``; only
the call convention differs -- this file passes the tensors as-is (BSND), while
``model_new_ascendc.py`` hands over ``transpose(1, 2)`` views, which the 950 interface
picks up as ``_LAYOUT_BNSD`` purely from the strides.  Both produce a contiguous BSND
``(B, S, H, D)`` output.
"""

import math

import torch
import torch.nn as nn

from flash_attn_npu_3 import flash_attn_func

B, H, KV_H, S, D = 1, 64, 64, 4096, 128
DTYPE = torch.float16
SCALE = 1.0 / math.sqrt(D)
CAUSAL = True


def _rand_bsnd(heads, generator):
    return torch.randn(B, S, heads, D, dtype=DTYPE, generator=generator)


def get_input_groups():
    """One positional ``(query, key, value)`` group: the harness calls ``model(*group)``."""
    generator = torch.Generator().manual_seed(2026)
    return [[_rand_bsnd(H, generator), _rand_bsnd(KV_H, generator), _rand_bsnd(KV_H, generator)]]


class Model(nn.Module):
    """Contiguous BSND ``(B, S, H, D)`` inputs, forwarded as-is."""

    def forward(self, query, key, value):
        return flash_attn_func(query, key, value, softmax_scale=SCALE, causal=CAUSAL)
