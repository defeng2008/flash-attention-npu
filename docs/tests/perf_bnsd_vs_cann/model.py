# Copyright (c) 2026, Minghua Shen.
"""CANN baseline for the BNSD forward performance case.

``torch_npu.npu_fused_infer_attention_score`` (CANN FIA) on a contiguous BNSD
``(B, H, S, D)`` query with ``input_layout="BNSD"``.

Causal masking follows the gate suite's convention: ``sparse_mode=3`` (right-down
aligned causal) together with an upper-triangular ``[2048, 2048]`` bool
``atten_mask`` (True marks the masked positions).  That is the only accepted
combination here: FIA rejects ``sparse_mode=2/3`` with an empty mask, rejects a
2D mask under ``sparse_mode=0``, and accepts a 3D/4D ``[1, 1, S, S]`` mask under
``sparse_mode=0`` (numerically equivalent, verified by probe).  The mask is built
once and cached on the module so timing covers the operator only.
"""

import math

import torch
import torch.nn as nn
import torch_npu

B, H, KV_H, S, D = 1, 64, 64, 4096, 128
DTYPE = torch.float16
SCALE = 1.0 / math.sqrt(D)
CAUSAL = True
MASK_SIZE = 2048


def _rand_bnsd(heads, generator):
    return torch.randn(B, heads, S, D, dtype=DTYPE, generator=generator)


def get_input_groups():
    """CPU-side contiguous BNSD ``(B, H, S, D)`` inputs; the harness moves them to the NPU.

    One group of positional arguments ``(query, key, value)``: the harness calls
    ``model(*group)``, so each group must be a sequence of tensors rather than a
    kwargs mapping.  ``perf_bnsd_vs_cann_perf_cases.jsonl`` only carries the
    shape/dtype metadata used for case enumeration and the report table.
    """
    generator = torch.Generator().manual_seed(2026)
    return [[_rand_bnsd(H, generator), _rand_bnsd(KV_H, generator), _rand_bnsd(KV_H, generator)]]


def build_causal_mask(device):
    """Upper-triangular bool mask (True = masked), the shape FIA requires."""
    if not CAUSAL:
        return None
    return torch.triu(torch.ones(MASK_SIZE, MASK_SIZE, dtype=torch.bool), diagonal=1).to(device)


def cann_forward(query, key, value, atten_mask=None):
    """Single FIA call; shared with ``bench_wallclock.py`` so both use one config."""
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        atten_mask=atten_mask,
        num_heads=H,
        num_key_value_heads=KV_H,
        scale=SCALE,
        pre_tokens=2**31 - 1,
        next_tokens=2**31 - 1,
        input_layout="BNSD",
        sparse_mode=3 if CAUSAL else 0,
        softmax_lse_flag=False,
    )
    return out


class Model(nn.Module):
    """CANN FIA forward, the performance baseline."""

    def __init__(self):
        super().__init__()
        self._mask = None

    def forward(self, query, key, value):
        if self._mask is None:
            self._mask = build_causal_mask(query.device)
        return cann_forward(query, key, value, self._mask)
