# Copyright (c) 2026, Minghua Shen.
"""BNSD ``(B, H, S, D)`` QKV layout cases for the Ascend 950 v4 backend.

Mirrors the v3 BNSD coverage on the v4 entry point: the stride fast path, causal
masking, the LSE layout, the forward-only restriction and a wider grouped-query
head split.  The layout is auto-detected, so ``flash_attn_func`` keeps its
documented ``(B, S, H, D)`` signature.
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401  registers the npu device used below

from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import make_attention_inputs, make_golden_attention_mask

from flash_attn_npu_4 import flash_attn_func


def _device_is_950():
    try:
        return torch_npu.npu.device_count() > 0 and "Ascend950" in torch_npu.npu.get_device_name()
    except Exception:  # pragma: no cover - only hit on hosts without an NPU
        return False


pytestmark = pytest.mark.skipif(
    not _device_is_950(),
    reason="the BNSD QKV layout is only implemented by the Ascend 950 backend",
)


@pytest.fixture(autouse=True)
def _select_npu_device():
    torch.npu.set_device(0)


B, H, H_KV, D = 2, 8, 2, 128
SQ, SK = 64, 128
SCALE = 1.0 / math.sqrt(D)


def _bsnd_inputs(sq, sk, dtype, num_heads=H, num_heads_kv=H_KV, seed=2026):
    """CPU BSND ``(B, S, H, D)`` tensors, the reference's native layout."""
    gen = torch.Generator().manual_seed(seed)
    q, k, v, _ = make_attention_inputs(
        (B, sq, num_heads, D),
        (B, sk, num_heads_kv, D),
        (B, sk, num_heads_kv, D),
        (B, sq, num_heads, D),
        dtype,
        generator=gen,
        device="cpu",
        requires_grad=(False, False, False),
    )
    return q, k, v


def _golden(q, k, v, dtype, causal):
    mask, _, _ = make_golden_attention_mask(q.shape[1], k.shape[1], causal, -1, -1)
    return ref_flash_attention_pair(q, k, v, SCALE, mask, dtype, 0.0)


def _bnsd_storage(q, k, v):
    """Contiguous BNSD ``(B, H, S, D)`` storages on the NPU."""
    return tuple(t.permute(0, 2, 1, 3).contiguous().npu() for t in (q, k, v))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v4_bnsd_stride_view_matches_reference(dtype):
    """Stride path: contiguous BNSD fed in as its BSND-shaped transposed view."""
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    out_ref, _, out_pt, _ = _golden(q, k, v, dtype, causal=False)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=False)
    assert tuple(out.shape) == (B, SQ, H, D)  # out follows the caller's q shape
    assert_fa_close(out, out_ref, out_pt, name="v4_bnsd_stride_out")


def test_v4_bnsd_causal_matches_reference():
    """BNSD with a causal mask: the LSE offset must follow the batch plane."""
    dtype = torch.bfloat16
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    out_ref, _, out_pt, _ = _golden(q, k, v, dtype, causal=True)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=True)
    assert tuple(out.shape) == (B, SQ, H, D)
    assert_fa_close(out, out_ref, out_pt, name="v4_bnsd_causal_out")


def test_v4_bnsd_multi_group_heads_matches_reference():
    """BNSD with a wider GQA split (16 query heads over 4 KV heads)."""
    dtype = torch.bfloat16
    heads, heads_kv = 16, 4
    q, k, v = _bsnd_inputs(SQ, SK, dtype, num_heads=heads, num_heads_kv=heads_kv)
    out_ref, _, out_pt, _ = _golden(q, k, v, dtype, causal=False)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=False)
    assert tuple(out.shape) == (B, SQ, heads, D)
    assert_fa_close(out, out_ref, out_pt, name="v4_bnsd_gqa_out")


def test_v4_bnsd_lse_layout_and_values():
    """v4 reports the LSE as ``(B, H, S)`` in BNSD too."""
    dtype = torch.bfloat16
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    _, lse_ref, _, lse_pt = _golden(q, k, v, dtype, causal=False)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out, lse = flash_attn_func(
        q_view, k_view, v_view, softmax_scale=SCALE, causal=False, return_lse=True
    )
    assert tuple(out.shape) == (B, SQ, H, D)
    assert tuple(lse.shape) == (B, H, SQ)
    assert_fa_close(lse, lse_ref, lse_pt, name="v4_bnsd_lse")


def test_v4_bnsd_backward_not_implemented():
    """BNSD is forward-only: the backward pass must refuse to run."""
    dtype = torch.bfloat16
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    q_b, k_b, v_b = (
        t.permute(0, 2, 1, 3).contiguous().npu().detach().requires_grad_(True) for t in (q, k, v)
    )
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))
    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=False)
    with pytest.raises(NotImplementedError):
        out.sum().backward()


def _bnsd_case(b, h, h_kv, sq, sk, d, dtype, seed):
    """CPU BSND tensors with an explicit head size (``D`` != the module default)."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, sq, h, d, dtype=dtype, generator=gen)
    k = torch.randn(b, sk, h_kv, d, dtype=dtype, generator=gen)
    v = torch.randn(b, sk, h_kv, d, dtype=dtype, generator=gen)
    return q, k, v


def _golden_d(q, k, v, dtype, causal=False):
    scale = 1.0 / math.sqrt(q.shape[-1])
    mask, _, _ = make_golden_attention_mask(q.shape[1], k.shape[1], causal, -1, -1)
    return ref_flash_attention_pair(q, k, v, scale, mask, dtype, 0.0)


def test_v4_bnsd_dn_grouped_heads_with_lse():
    """DN fast path under BNSD: grouped heads (ndNum > 1) and the ``(B, H, S)`` LSE."""
    dtype = torch.bfloat16
    b, h, h_kv, sq, sk, d = 2, 16, 4, 64, 128, 128
    q, k, v = _bnsd_case(b, h, h_kv, sq, sk, d, dtype, 2026)
    out_ref, lse_ref, out_pt, lse_pt = _golden_d(q, k, v, dtype)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out, lse = flash_attn_func(
        q_view, k_view, v_view, softmax_scale=1.0 / math.sqrt(d), causal=False, return_lse=True
    )
    assert tuple(out.shape) == (b, sq, h, d)
    assert tuple(lse.shape) == (b, h, sq)
    assert_fa_close(out, out_ref, out_pt, name="v4_bnsd_dn_gqa_out")
    assert_fa_close(lse, lse_ref, lse_pt, name="v4_bnsd_dn_gqa_lse")


def test_v4_bnsd_dn_head_size_256():
    """DN under BNSD with the largest allowed head size and a rounded head plane."""
    dtype = torch.bfloat16
    b, h, h_kv, sq, sk, d = 2, 8, 2, 64, 128, 256
    q, k, v = _bnsd_case(b, h, h_kv, sq, sk, d, dtype, 2027)
    out_ref, _, out_pt, _ = _golden_d(q, k, v, dtype)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=1.0 / math.sqrt(d), causal=False)
    assert tuple(out.shape) == (b, sq, h, d)
    assert_fa_close(out, out_ref, out_pt, name="v4_bnsd_dn_d256_out")


def test_v4_bnsd_dn_unaligned_q_seqlen():
    """DN under BNSD with a Q length that is not a multiple of the 32-row tile."""
    dtype = torch.bfloat16
    b, h, h_kv, sq, sk, d = 2, 8, 2, 100, 192, 128
    q, k, v = _bnsd_case(b, h, h_kv, sq, sk, d, dtype, 2028)
    out_ref, _, out_pt, _ = _golden_d(q, k, v, dtype)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=1.0 / math.sqrt(d), causal=False)
    assert tuple(out.shape) == (b, sq, h, d)
    assert_fa_close(out, out_ref, out_pt, name="v4_bnsd_dn_unaligned_out")
