# Copyright (c) 2026, Minghua Shen.
"""Custom Ascend 950 v3 forward for the BNSD performance case.

The caller hands in contiguous BNSD ``(B, H, S, D)`` storages; the operator reads
them in place (the BSND view is non-contiguous, so no copy is materialised) and
returns a contiguous BSND ``(B, S, H, D)`` output.
"""

import math

import torch.nn as nn

from flash_attn_npu_3 import flash_attn_func

B, H, KV_H, S, D = 1, 64, 64, 4096, 128
SCALE = 1.0 / math.sqrt(D)
CAUSAL = True


class ModelNew(nn.Module):
    """Ascend 950 v3 flash attention forward (BNSD input, contiguous BSND output)."""

    def forward(self, query, key, value):
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        return flash_attn_func(q, k, v, softmax_scale=SCALE, causal=CAUSAL)
