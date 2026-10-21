# Copyright (c) 2026, Minghua Shen.
"""BNSD ``(B, H, S, D)`` QKV layout regression cases for the Ascend 910 v3 backend.

The 910 backend accepts query/key/value stored as ``(B, H, S, D)`` and detects the
layout from the strides, so the public API keeps its documented ``(B, S, H, D)``
signature and no caller-visible switch is added.  A contiguous BNSD tensor handed
in as its BSND-shaped ``transpose(1, 2)`` view is restored with zero copies.

These cases run only on Ascend 910 hosts; the 950-only cases live in
``test_flash_attn_npu_v3_bnsd.py``.  On the 910 backend the LSE keeps its
``(B, H, S)`` layout (the 950 backend reports ``(B, S, H)``).
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401  registers the npu device used below

from tests.common.attention_ref import ref_flash_attention, ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import make_attention_inputs, make_golden_attention_mask

from flash_attn_npu_3 import flash_attn_func, flash_attn_with_kvcache


def _device_is_910():
    try:
        return torch_npu.npu.device_count() > 0 and "Ascend910" in torch_npu.npu.get_device_name()
    except Exception:  # pragma: no cover - only hit on hosts without an NPU
        return False


pytestmark = pytest.mark.skipif(
    not _device_is_910(),
    reason="the 910 BNSD QKV layout regression cases require an Ascend 910 host",
)


@pytest.fixture(autouse=True)
def _select_npu_device():
    torch.npu.set_device(0)


B, H, H_KV, D = 2, 8, 2, 128
SQ, SK = 64, 128
SCALE = 1.0 / math.sqrt(D)
TOL = {"rtol": 2e-2, "atol": 2e-2}


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


def _bnsd_views(q, k, v):
    """BSND-shaped non-contiguous views over contiguous BNSD storages."""
    return tuple(t.transpose(1, 2) for t in (q, k, v))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v3_bnsd_stride_view_matches_reference(dtype):
    """Stride path: contiguous BNSD fed in as its BSND-shaped transposed view."""
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    out_ref, _, out_pt, _ = _golden(q, k, v, dtype, causal=False)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = _bnsd_views(q_b, k_b, v_b)
    # This is the fast path: a non-contiguous view that the backend restores.
    assert not q_view.is_contiguous()

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=False)
    # The result mirrors the shape of the query the caller passed in.
    assert tuple(out.shape) == tuple(q_view.shape) == (B, SQ, H, D)
    assert_fa_close(out, out_ref, out_pt, name="v3_910_bnsd_stride_out")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v3_bnsd_matches_equivalent_bsnd(dtype):
    """The same data as BSND and as a BNSD view must produce identical results."""
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    out_bsnd = flash_attn_func(q.npu(), k.npu(), v.npu(), softmax_scale=SCALE, causal=False)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = _bnsd_views(q_b, k_b, v_b)
    out_bnsd = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=False)

    assert tuple(out_bsnd.shape) == tuple(out_bnsd.shape) == (B, SQ, H, D)
    torch.testing.assert_close(out_bnsd.float().cpu(), out_bsnd.float().cpu(), **TOL)


def test_v3_contiguous_rank4_falls_back_to_bsnd():
    """Contiguous rank-4 operands never select BNSD, even when shaped ``(B, H, S, D)``."""
    dtype = torch.bfloat16

    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    out_ref, _, out_pt, _ = _golden(q, k, v, dtype, causal=False)
    q_n, k_n, v_n = q.npu(), k.npu(), v.npu()
    assert q_n.is_contiguous()
    out = flash_attn_func(q_n, k_n, v_n, softmax_scale=SCALE, causal=False)
    assert tuple(out.shape) == (B, SQ, H, D)
    assert_fa_close(out, out_ref, out_pt, name="v3_910_contiguous_bsnd_out")

    # A *contiguous* ``(B, H, S, D)`` fails the non-contiguous stride match, so it
    # must be read verbatim as BSND: the shape is preserved, never transposed.
    b, h, h_kv, sq, skv, d = 2, 4, 6, 64, 8, 128
    gen = torch.Generator().manual_seed(2029)
    q_c = torch.randn(b, h, sq, d, dtype=dtype, generator=gen)
    k_c = torch.randn(b, h_kv, skv, d, dtype=dtype, generator=gen)
    v_c = torch.randn(b, h_kv, skv, d, dtype=dtype, generator=gen)
    mask, _, _ = make_golden_attention_mask(h, h_kv, False, -1, -1)
    c_ref, _, c_pt, _ = ref_flash_attention_pair(
        q_c, k_c, v_c, 1.0 / math.sqrt(d), mask, dtype, 0.0
    )
    q_cn, k_cn, v_cn = q_c.npu(), k_c.npu(), v_c.npu()
    assert q_cn.is_contiguous()
    out_c = flash_attn_func(q_cn, k_cn, v_cn, softmax_scale=1.0 / math.sqrt(d), causal=False)
    assert tuple(out_c.shape) == (b, h, sq, d)  # BSND reading, never (b, sq, h, d)
    assert_fa_close(out_c, c_ref, c_pt, name="v3_910_contiguous_bnsd_storage_out")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v3_bnsd_causal_matches_reference(dtype):
    """BNSD with a causal mask: the alignment must follow the batch plane."""
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    out_ref, _, out_pt, _ = _golden(q, k, v, dtype, causal=True)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = _bnsd_views(q_b, k_b, v_b)

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=True)
    assert tuple(out.shape) == (B, SQ, H, D)
    assert_fa_close(out, out_ref, out_pt, name="v3_910_bnsd_causal_out")


def test_v3_bnsd_lse_layout_and_values():
    """The 910 v3 backend keeps the LSE as ``(B, H, S)`` under BNSD as well."""
    dtype = torch.bfloat16
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    _, lse_ref, _, lse_pt = _golden(q, k, v, dtype, causal=False)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = _bnsd_views(q_b, k_b, v_b)

    out, lse = flash_attn_func(
        q_view, k_view, v_view, softmax_scale=SCALE, causal=False, return_attn_probs=True
    )
    assert tuple(out.shape) == (B, SQ, H, D)
    assert tuple(lse.shape) == (B, H, SQ)
    assert_fa_close(lse, lse_ref, lse_pt, name="v3_910_bnsd_lse")


def test_v3_bnsd_backward_not_implemented():
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


def test_v3_bnsd_gqa_grouped_heads():
    """BNSD with a grouped-query split (``H_KV >= 2``)."""
    dtype = torch.bfloat16
    heads, heads_kv = 16, 4
    q, k, v = _bsnd_inputs(SQ, SK, dtype, num_heads=heads, num_heads_kv=heads_kv)
    out_ref, _, out_pt, _ = _golden(q, k, v, dtype, causal=False)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = _bnsd_views(q_b, k_b, v_b)

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=False)
    assert tuple(out.shape) == (B, SQ, heads, D)
    assert_fa_close(out, out_ref, out_pt, name="v3_910_bnsd_gqa_out")


def test_v3_bnsd_kvcache_allocated_plane_matches_reference():
    """BNSD KV cache: the head-plane pitch comes from the allocated length."""
    dtype = torch.bfloat16
    s_alloc, used, sq = 1024, [300, 512], 32
    gen = torch.Generator().manual_seed(2026)
    q_cpu = torch.randn(B, H, sq, D, dtype=dtype, generator=gen)
    k_alloc = torch.randn(B, H_KV, s_alloc, D, dtype=dtype, generator=gen)
    v_alloc = torch.randn(B, H_KV, s_alloc, D, dtype=dtype, generator=gen)

    outs = []
    for b, n in enumerate(used):
        mask, _, _ = make_golden_attention_mask(sq, n, True, -1, -1)
        out_b, _ = ref_flash_attention(
            q_cpu[b : b + 1].transpose(1, 2).contiguous(),
            k_alloc[b : b + 1, :, :n].transpose(1, 2).contiguous(),
            v_alloc[b : b + 1, :, :n].transpose(1, 2).contiguous(),
            SCALE,
            mask,
            dtype,
            0.0,
            upcast=True,
            reorder_ops=False,
        )
        outs.append(out_b)
    out_ref = torch.cat(outs, 0)  # (B, sq, H, D)

    # A plane pitch taken from the used length would read this sentinel tail.
    k_junk, v_junk = k_alloc.clone(), v_alloc.clone()
    for b, n in enumerate(used):
        k_junk[b, :, n:, :] = 1e4
        v_junk[b, :, n:, :] = 1e4
    q, k_junk, v_junk = q_cpu.npu(), k_junk.npu(), v_junk.npu()

    out, lse, *_ = flash_attn_with_kvcache(
        q.transpose(1, 2),
        k_junk.transpose(1, 2),
        v_junk.transpose(1, 2),
        cache_seqlens=torch.tensor(used, dtype=torch.int32).npu(),
        softmax_scale=SCALE,
        causal=True,
        return_softmax_lse=True,
    )
    assert tuple(out.shape) == (B, sq, H, D)  # out follows the caller's q shape
    assert tuple(lse.shape) == (B, H, sq)
    torch.testing.assert_close(out.float().cpu(), out_ref.float().cpu(), **TOL)

    # A different pre-allocated tail must not change a single element.
    k_zero, v_zero = k_alloc.clone(), v_alloc.clone()
    for b, n in enumerate(used):
        k_zero[b, :, n:, :] = 0.0
        v_zero[b, :, n:, :] = 0.0
    k_zero, v_zero = k_zero.npu(), v_zero.npu()
    out_zero, lse_zero, *_ = flash_attn_with_kvcache(
        q.transpose(1, 2),
        k_zero.transpose(1, 2),
        v_zero.transpose(1, 2),
        cache_seqlens=torch.tensor(used, dtype=torch.int32).npu(),
        softmax_scale=SCALE,
        causal=True,
        return_softmax_lse=True,
    )
    torch.testing.assert_close(out_zero.float().cpu(), out.float().cpu(), rtol=0, atol=1e-3)
    torch.testing.assert_close(lse_zero.float().cpu(), lse.float().cpu(), rtol=0, atol=1e-3)
