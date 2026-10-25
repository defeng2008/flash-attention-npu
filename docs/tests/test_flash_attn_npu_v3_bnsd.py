# Copyright (c) 2026, Minghua Shen.
"""BNSD ``(B, H, S, D)`` QKV layout cases for the Ascend 950 v3 backend.

The 950 backend accepts query/key/value laid out as ``(B, H, S, D)`` and
auto-detects the layout, so the public API keeps its documented ``(B, S, H, D)``
signature and no caller-visible switch is added.  These cases cover:

* the stride fast path (a contiguous BNSD tensor handed in as its BSND-shaped
  transposed view, which the backend restores with zero copies),
* causal masking, the LSE layout, the forward-only restriction, and the BNSD
  KV-cache plane length used by ``flash_attn_with_kvcache``.
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401  registers the npu device used below

from tests.common.attention_ref import ref_flash_attention, ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import make_attention_inputs, make_golden_attention_mask

from flash_attn_npu_3 import flash_attn_func, flash_attn_with_kvcache


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
TOL = {"rtol": 2e-2, "atol": 2e-2}


def _bsnd_inputs(sq, sk, dtype, seed=2026):
    """CPU BSND ``(B, S, H, D)`` tensors, the reference's native layout."""
    gen = torch.Generator().manual_seed(seed)
    q, k, v, _ = make_attention_inputs(
        (B, sq, H, D),
        (B, sk, H_KV, D),
        (B, sk, H_KV, D),
        (B, sq, H, D),
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


def _assert_contiguous_bsnd(out, q, name):
    """The forward must return a *contiguous* BSND ``out`` shaped like the caller's ``q``.

    ``q`` is the tensor the caller handed in (here a non-contiguous BSND view over a
    contiguous BNSD storage): the result must match its shape and be contiguous, i.e.
    the kernel writes BSND while reading BNSD, so no view/``transpose`` is returned.
    """
    assert tuple(out.shape) == tuple(q.shape), (
        f"[{name}] expected out.shape == q.shape {tuple(q.shape)} (contiguous BSND), "
        f"got {tuple(out.shape)}"
    )
    assert out.is_contiguous() is True, (
        f"[{name}] expected a contiguous BSND output (is_contiguous() is True), "
        f"got is_contiguous() == False with strides {tuple(out.stride())}"
    )


def _snapshot_inputs(views):
    """Record identity, strides and contents of the caller's non-contiguous BNSD views."""
    return [
        (v.data_ptr(), tuple(v.stride()), v.untyped_storage().data_ptr(), v.cpu().clone())
        for v in views
    ]


def _assert_inputs_zero_copy(snapshots, views, name):
    """Guard against a "``.contiguous()`` first" regression on the Q/K/V inputs.

    The BNSD inputs must be fed to the kernel as-is; a degraded path that
    materialised a contiguous copy (or wrote through) the caller's views would show up
    as a changed ``data_ptr`` / ``stride`` / storage, or as modified contents.
    """
    for (ptr, stride, base_ptr, clone), view in zip(snapshots, views):
        assert view.data_ptr() == ptr and tuple(view.stride()) == stride, (
            f"[{name}] BNSD input must stay the caller's non-contiguous view "
            f"(zero copy, never materialised first): expected data_ptr={ptr} stride={stride}, "
            f"got data_ptr={view.data_ptr()} stride={tuple(view.stride())}"
        )
        assert view.untyped_storage().data_ptr() == base_ptr, (
            f"[{name}] BNSD input must still alias the caller's storage (zero copy)"
        )
        assert torch.equal(view.cpu(), clone), (
            f"[{name}] the forward must not modify the BNSD input in place"
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bnsd_stride_view_matches_reference(dtype):
    """Stride path: contiguous BNSD fed in as its BSND-shaped transposed view."""
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    out_ref, _, out_pt, _ = _golden(q, k, v, dtype, causal=False)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=False)
    # The result mirrors the shape of the query the caller passed in and is contiguous BSND.
    assert tuple(out.shape) == (B, SQ, H, D)
    _assert_contiguous_bsnd(out, q_view, "bnsd_stride_out")
    assert_fa_close(out, out_ref, out_pt, name="bnsd_stride_out")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bnsd_causal_matches_reference(dtype):
    """BNSD with a causal mask: the LSE offset must follow the batch plane."""
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    out_ref, _, out_pt, _ = _golden(q, k, v, dtype, causal=True)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=True)
    assert tuple(out.shape) == (B, SQ, H, D)
    _assert_contiguous_bsnd(out, q_view, "bnsd_causal_out")
    assert_fa_close(out, out_ref, out_pt, name="bnsd_causal_out")


def test_bnsd_lse_layout_and_values():
    """v3 ``flash_attn_func`` reports the LSE as ``(B, S, H)`` in BNSD too."""
    dtype = torch.bfloat16
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    _, lse_ref, _, lse_pt = _golden(q, k, v, dtype, causal=False)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out, lse = flash_attn_func(
        q_view, k_view, v_view, softmax_scale=SCALE, causal=False, return_attn_probs=True
    )
    assert tuple(out.shape) == (B, SQ, H, D)
    assert tuple(lse.shape) == (B, SQ, H)
    _assert_contiguous_bsnd(out, q_view, "bnsd_lse_out")
    assert_fa_close(
        lse,
        lse_ref.transpose(-1, -2).contiguous(),
        lse_pt.transpose(-1, -2).contiguous(),
        name="bnsd_lse",
    )


def test_bnsd_backward_not_implemented():
    """BNSD is forward-only: the backward pass must refuse to run.

    The explicit ``ctx.layout == _LAYOUT_BNSD`` guard was dropped upstream, so the
    refusal now comes from the backward input validation (the BNSD operands are read
    as BSND) and surfaces as a ``RuntimeError`` rather than ``NotImplementedError``.
    """
    dtype = torch.bfloat16
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    q_b, k_b, v_b = (
        t.permute(0, 2, 1, 3).contiguous().npu().detach().requires_grad_(True) for t in (q, k, v)
    )
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))
    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=False)
    with pytest.raises((NotImplementedError, RuntimeError)):
        out.sum().backward()


def test_bnsd_kvcache_allocated_plane_matches_reference():
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
    # S_alloc > max(cache_seqlens): the output is still contiguous BSND.
    _assert_contiguous_bsnd(out, q.transpose(1, 2), "bnsd_kvcache_out")
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
    # The zero-filled tail must give the same contiguous BSND output.
    _assert_contiguous_bsnd(out_zero, q.transpose(1, 2), "bnsd_kvcache_zero_out")
    torch.testing.assert_close(out_zero.float().cpu(), out.float().cpu(), rtol=0, atol=1e-3)
    torch.testing.assert_close(lse_zero.float().cpu(), lse.float().cpu(), rtol=0, atol=1e-3)


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


def test_bnsd_dn_grouped_heads_with_lse():
    """DN fast path under BNSD: grouped heads (ndNum > 1) and the ``(B, S, H)`` LSE."""
    dtype = torch.bfloat16
    b, h, h_kv, sq, sk, d = 2, 16, 4, 64, 128, 128
    q, k, v = _bnsd_case(b, h, h_kv, sq, sk, d, dtype, 2026)
    out_ref, lse_ref, out_pt, lse_pt = _golden_d(q, k, v, dtype)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out, lse = flash_attn_func(
        q_view, k_view, v_view, softmax_scale=1.0 / math.sqrt(d), causal=False, return_attn_probs=True
    )
    assert tuple(out.shape) == (b, sq, h, d)
    assert tuple(lse.shape) == (b, sq, h)
    _assert_contiguous_bsnd(out, q_view, "bnsd_dn_gqa_out")
    assert_fa_close(out, out_ref, out_pt, name="bnsd_dn_gqa_out")
    assert_fa_close(
        lse,
        lse_ref.transpose(-1, -2).contiguous(),
        lse_pt.transpose(-1, -2).contiguous(),
        name="bnsd_dn_gqa_lse",
    )


def test_bnsd_dn_head_size_192():
    """DN under BNSD with a head size that is not a multiple of the 128-wide tile."""
    dtype = torch.bfloat16
    b, h, h_kv, sq, sk, d = 2, 8, 2, 64, 128, 192
    q, k, v = _bnsd_case(b, h, h_kv, sq, sk, d, dtype, 2027)
    out_ref, _, out_pt, _ = _golden_d(q, k, v, dtype)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=1.0 / math.sqrt(d), causal=False)
    assert tuple(out.shape) == (b, sq, h, d)
    _assert_contiguous_bsnd(out, q_view, "bnsd_dn_d192_out")
    assert_fa_close(out, out_ref, out_pt, name="bnsd_dn_d192_out")


def test_bnsd_dn_wide_group_two_kv_heads():
    """DN under BNSD with a wide GQA split (16 query heads over 2 KV heads)."""
    dtype = torch.bfloat16
    b, h, h_kv, sq, sk, d = 2, 16, 2, 64, 128, 128
    q, k, v = _bnsd_case(b, h, h_kv, sq, sk, d, dtype, 2028)
    out_ref, _, out_pt, _ = _golden_d(q, k, v, dtype)

    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))

    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=1.0 / math.sqrt(d), causal=False)
    assert tuple(out.shape) == (b, sq, h, d)
    _assert_contiguous_bsnd(out, q_view, "bnsd_dn_wide_group_out")
    assert_fa_close(out, out_ref, out_pt, name="bnsd_dn_wide_group_out")


def test_bnsd_inputs_not_contiguous_copied():
    """BNSD inputs are read as-is: the forward must not materialise a ``.contiguous()`` copy.

    The caller's non-contiguous BNSD views are snapshotted (identity, strides, storage
    and contents) before the call and must be untouched afterwards -- the backend feeds
    the stride views straight to the kernel instead of copying Q/K/V first.
    """
    dtype = torch.bfloat16
    q, k, v = _bsnd_inputs(SQ, SK, dtype)
    q_b, k_b, v_b = _bnsd_storage(q, k, v)
    q_view, k_view, v_view = (t.transpose(1, 2) for t in (q_b, k_b, v_b))
    views = (q_view, k_view, v_view)
    assert not q_view.is_contiguous()
    assert tuple(q_view.stride()) == (H * SQ * D, D, SQ * D, 1)  # BSND-shaped BNSD strides

    before = _snapshot_inputs(views)
    out = flash_attn_func(q_view, k_view, v_view, softmax_scale=SCALE, causal=False)

    _assert_contiguous_bsnd(out, q_view, "bnsd_inputs_zero_copy_out")
    _assert_inputs_zero_copy(before, views, "bnsd_inputs_zero_copy")