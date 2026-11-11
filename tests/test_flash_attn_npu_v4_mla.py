# Copyright (c) 2026, Minghua Shen.
"""MLA (Multi-head Latent Attention) forward tests for the FA4 backend.

Two shape signatures are accepted by ``flash_attn_func`` / ``flash_attn_varlen_func``:

* prefill (unabsorbed): ``q``/``k`` head dim == ``Dq`` (192 in the user scenario),
  ``v`` head dim == ``Dv`` (128). GQA is allowed.
* decode (absorbed): ``q`` head dim 64 (rope) plus an extra ``qv`` of head dim 512
  (nope), ``k`` head dim 64, ``v`` head dim 512, KV heads == 1.

FA4 has no ``flash_attn_with_kvcache`` entry; decode runs through
``flash_attn_func`` (dense BSND/BNSD) or ``flash_attn_varlen_func`` (paged TND).
Both shapes also accept BNSD input, i.e. packed ``(B, H, S, D)`` storage handed out
as the non-contiguous ``transpose(1, 2)`` view; the output then keeps the caller's
``(B, S, H, dv)`` view shape.
"""

import torch
import torch_npu
import pytest

from flash_attn_npu_4 import (
    flash_attn_func,
    flash_attn_varlen_func,
)

DEVICE = "npu"
DTYPE = torch.bfloat16

# bf16 relative-L2 acceptance threshold (see the MLA spec).
REL_L2_TOL = 5e-3


def _rel_l2(actual, ref):
    a = actual.float().reshape(-1)
    r = ref.float().reshape(-1)
    return ((a - r).pow(2).sum() / r.pow(2).sum().clamp_min(1e-12)).sqrt().item()


def _ref_bsnd(q, k, v, softmax_scale, causal):
    """FP32 reference. q: (B,S,H,Dq); k: (B,Sk,Hk,Dq); v: (B,Sk,Hk,Dv)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    heads, heads_k = qf.shape[2], kf.shape[2]
    if heads != heads_k:
        rep = heads // heads_k
        kf = kf.repeat_interleave(rep, dim=2)
        vf = vf.repeat_interleave(rep, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", qf, kf) * softmax_scale
    if causal:
        t, s = scores.shape[-2], scores.shape[-1]
        keep = torch.arange(s).view(1, 1, 1, s) <= (
            torch.arange(t).view(1, 1, t, 1) + (s - t)
        )
        scores = scores.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshd->bthd", probs, vf)


def _ref_varlen(q, k, v, cu_q, cu_k, softmax_scale, causal):
    """FP32 reference over packed TND tensors of equal batch length."""
    outs = []
    for b in range(len(cu_q) - 1):
        qb = q[cu_q[b]:cu_q[b + 1]].unsqueeze(0)
        kb = k[cu_k[b]:cu_k[b + 1]].unsqueeze(0)
        vb = v[cu_k[b]:cu_k[b + 1]].unsqueeze(0)
        outs.append(_ref_bsnd(qb, kb, vb, softmax_scale, causal).squeeze(0))
    return torch.cat(outs, dim=0)


def _rand(shape):
    return (torch.rand(shape, device="cpu") * 2.0 - 1.0).to(DTYPE).to(DEVICE)


# --------------------------------------------------------------------------
# prefill / unabsorbed
# --------------------------------------------------------------------------


def test_mla_prefill_bsnd_func():
    """User prefill scenario: q/k=(1,198,32,192), v=(1,198,32,128), causal."""
    batch, seqlen, heads, dq, dv = 1, 198, 32, 192, 128
    q, k, v = _rand((batch, seqlen, heads, dq)), _rand((batch, seqlen, heads, dq)), _rand(
        (batch, seqlen, heads, dv)
    )
    scale = dq ** -0.5

    out = flash_attn_func(q, k, v, softmax_scale=scale, causal=True)

    assert out.shape == (batch, seqlen, heads, dv)
    ref = _ref_bsnd(q.cpu(), k.cpu(), v.cpu(), scale, True)
    assert _rel_l2(out.cpu(), ref) < REL_L2_TOL


def test_mla_prefill_bsnd_func_default_scale():
    """Default scale must be q.last ** -0.5 == 192 ** -0.5 for prefill."""
    batch, seqlen, heads, dq, dv = 2, 64, 8, 192, 128
    q, k, v = _rand((batch, seqlen, heads, dq)), _rand((batch, seqlen, heads, dq)), _rand(
        (batch, seqlen, heads, dv)
    )

    out = flash_attn_func(q, k, v, causal=False)

    ref = _ref_bsnd(q.cpu(), k.cpu(), v.cpu(), dq ** -0.5, False)
    assert _rel_l2(out.cpu(), ref) < REL_L2_TOL


def test_mla_prefill_tnd_varlen():
    seq_lens = [198, 130, 7]
    # cu_seqlens must live on the NPU: the scheduler metadata AICPU dereferences
    # it in place, so a CPU tensor would hand the kernel a host pointer.
    cu_cpu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    for i, n in enumerate(seq_lens, start=1):
        cu_cpu[i] = cu_cpu[i - 1] + n
    total = int(cu_cpu[-1])
    cu = cu_cpu.to(DEVICE)
    heads, dq, dv = 16, 192, 128
    q = _rand((total, heads, dq))
    k = _rand((total, heads, dq))
    v = _rand((total, heads, dv))
    scale = dq ** -0.5

    out = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max(seq_lens), max_seqlen_k=max(seq_lens),
        softmax_scale=scale, causal=True,
    )

    assert out.shape == (total, heads, dv)
    ref = _ref_varlen(
        q.cpu(), k.cpu(), v.cpu(), cu_cpu.tolist(), cu_cpu.tolist(), scale, True
    )
    assert _rel_l2(out.cpu(), ref) < REL_L2_TOL


def test_mla_prefill_gqa():
    """kvHeads < qHeads with Dq != Dv."""
    batch, seqlen, heads, heads_k, dq, dv = 1, 96, 16, 4, 192, 128
    q = _rand((batch, seqlen, heads, dq))
    k = _rand((batch, seqlen, heads_k, dq))
    v = _rand((batch, seqlen, heads_k, dv))
    scale = dq ** -0.5

    out = flash_attn_func(q, k, v, softmax_scale=scale, causal=True)

    ref = _ref_bsnd(q.cpu(), k.cpu(), v.cpu(), scale, True)
    assert _rel_l2(out.cpu(), ref) < REL_L2_TOL


# --------------------------------------------------------------------------
# decode / absorbed
# --------------------------------------------------------------------------


def _ref_decode(q, qv, k_rope, v_latent, softmax_scale):
    """FP32 reference for one decode step.

    q/qv: (heads, 64/512); k_rope: (kv_len, 64); v_latent: (kv_len, 512).
    scores = scale * (qv @ v^T + q @ k^T), out = softmax(scores) @ v.
    """
    qf, qvf = q.float(), qv.float()
    kf, vf = k_rope.float(), v_latent.float()
    scores = (qvf @ vf.transpose(0, 1) + qf @ kf.transpose(0, 1)) * softmax_scale
    probs = torch.softmax(scores, dim=-1)
    return probs @ vf


def test_mla_decode_bsnd_no_paged():
    """Absorbed decode: q=(1,1,32,64) + qv=(1,1,32,512), cache (1,1025,1,64/512)."""
    heads, kv_len, drope, dlatent = 32, 1025, 64, 512
    q, qv = _rand((1, 1, heads, drope)), _rand((1, 1, heads, dlatent))
    k_cache = _rand((1, kv_len, 1, drope))
    v_cache = _rand((1, kv_len, 1, dlatent))
    scale = (drope + dlatent) ** -0.5

    out = flash_attn_func(q, k_cache, v_cache, qv=qv, softmax_scale=scale)

    assert out.shape == (1, 1, heads, dlatent)
    ref = _ref_decode(
        q.cpu()[0, 0], qv.cpu()[0, 0], k_cache.cpu()[0, :, 0], v_cache.cpu()[0, :, 0], scale
    )
    assert _rel_l2(out.cpu()[0, 0], ref) < REL_L2_TOL


def test_mla_decode_bsnd_default_scale():
    """Default scale for absorbed MLA must be (64 + 512) ** -0.5, not 64 ** -0.5."""
    heads, kv_len, drope, dlatent = 16, 512, 64, 512
    q, qv = _rand((1, 1, heads, drope)), _rand((1, 1, heads, dlatent))
    k_cache = _rand((1, kv_len, 1, drope))
    v_cache = _rand((1, kv_len, 1, dlatent))

    out = flash_attn_func(q, k_cache, v_cache, qv=qv)

    ref = _ref_decode(
        q.cpu()[0, 0], qv.cpu()[0, 0], k_cache.cpu()[0, :, 0], v_cache.cpu()[0, :, 0],
        (drope + dlatent) ** -0.5,
    )
    assert _rel_l2(out.cpu()[0, 0], ref) < REL_L2_TOL


def test_mla_decode_paged_varlen():
    """Paged absorbed decode through the TND entry: 1025 tokens in 17 pages of 64."""
    heads, kv_len, drope, dlatent, page_size = 16, 1025, 64, 512, 64
    num_pages = (kv_len + page_size - 1) // page_size
    q = _rand((1, heads, drope))
    qv = _rand((1, heads, dlatent))
    k_cache = _rand((num_pages, page_size, 1, drope))
    v_cache = _rand((num_pages, page_size, 1, dlatent))
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    seqused_k = torch.tensor([kv_len], dtype=torch.int32, device=DEVICE)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    scale = (drope + dlatent) ** -0.5

    out = flash_attn_varlen_func(
        q, k_cache, v_cache, qv=qv, cu_seqlens_q=cu_q, seqused_k=seqused_k,
        max_seqlen_q=1, page_table=page_table, softmax_scale=scale,
    )

    assert out.shape == (1, heads, dlatent)
    k_flat = k_cache.cpu().reshape(num_pages * page_size, 1, drope)[:kv_len, 0]
    v_flat = v_cache.cpu().reshape(num_pages * page_size, 1, dlatent)[:kv_len, 0]
    ref = _ref_decode(q.cpu()[0], qv.cpu()[0], k_flat, v_flat, scale)
    assert _rel_l2(out.cpu()[0], ref) < REL_L2_TOL


# --------------------------------------------------------------------------
# TND (packed varlen) decode: one query token per sequence, so the packed q/qv
# and output batch bases are the cu_seqlens_q prefixes and the K/V base and
# length come from cu_seqlens_k. The same 128-token KV split and CombineScale
# reduction run as in the BSND case.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("kv_len", [129, 513, 1025])
def test_mla_decode_tnd_varlen(batch, kv_len):
    """Absorbed TND decode over a varlen batch, one latent cache per sequence."""
    heads, drope, dlatent = 32, 64, 512
    q, qv = _rand((batch, heads, drope)), _rand((batch, heads, dlatent))
    k_cache = _rand((batch * kv_len, 1, drope))
    v_cache = _rand((batch * kv_len, 1, dlatent))
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE)
    cu_k = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * kv_len
    scale = (drope + dlatent) ** -0.5

    out = flash_attn_varlen_func(
        q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=1, max_seqlen_k=kv_len, qv=qv, softmax_scale=scale,
    )

    assert out.shape == (batch, heads, dlatent)
    for b in range(batch):
        ref = _ref_decode(
            q.cpu()[b], qv.cpu()[b],
            k_cache.cpu()[b * kv_len:(b + 1) * kv_len, 0],
            v_cache.cpu()[b * kv_len:(b + 1) * kv_len, 0],
            scale,
        )
        assert _rel_l2(out.cpu()[b], ref) < REL_L2_TOL


# --------------------------------------------------------------------------
# BNSD layout: packed (B, H, S, D) storage handed out as the transpose(1, 2) view
# --------------------------------------------------------------------------


def _to_bnsd_view(x):
    """Return the (B, S, H, D) view of packed BNSD storage.

    This is exactly what a caller holds after ``contiguous_bnsd.transpose(1, 2)``: the
    values are BNSD, but the strides describe a non-contiguous BSND view, which is how
    the interface recognizes the BNSD layout.
    """
    return x.transpose(1, 2).contiguous().transpose(1, 2)


def test_mla_prefill_bnsd():
    """Prefill BNSD input must match the equivalent BSND call."""
    batch, seqlen, heads, dq, dv = 2, 64, 8, 192, 128
    q, k, v = _rand((batch, seqlen, heads, dq)), _rand(
        (batch, seqlen, heads, dq)
    ), _rand((batch, seqlen, heads, dv))
    scale = dq ** -0.5

    out = flash_attn_func(
        _to_bnsd_view(q), _to_bnsd_view(k), _to_bnsd_view(v), softmax_scale=scale, causal=True
    )

    # Output keeps the caller's view shape (B, S, H, dv), not the BNSD (B, H, S, dv).
    assert out.shape == (batch, seqlen, heads, dv)
    ref = _ref_bsnd(q.cpu(), k.cpu(), v.cpu(), scale, True)
    assert _rel_l2(out.cpu(), ref) < REL_L2_TOL


def test_mla_prefill_bnsd_gqa():
    """BNSD prefill with kvHeads < qHeads."""
    batch, seqlen, heads, heads_k, dq, dv = 2, 96, 16, 4, 192, 128
    q = _rand((batch, seqlen, heads, dq))
    k = _rand((batch, seqlen, heads_k, dq))
    v = _rand((batch, seqlen, heads_k, dv))
    scale = dq ** -0.5

    out = flash_attn_func(
        _to_bnsd_view(q), _to_bnsd_view(k), _to_bnsd_view(v), softmax_scale=scale, causal=True
    )

    assert out.shape == (batch, seqlen, heads, dv)
    ref = _ref_bsnd(q.cpu(), k.cpu(), v.cpu(), scale, True)
    assert _rel_l2(out.cpu(), ref) < REL_L2_TOL


def _ref_decode_multi(q, qv, k_rope, v_latent, softmax_scale):
    """FP32 reference for an absorbed decode step over ``Sq`` queries.

    q/qv: (Sq, heads, 64/512); k_rope: (kv_len, 64); v_latent: (kv_len, 512).
    """
    qf, qvf = q.float(), qv.float()
    kf, vf = k_rope.float(), v_latent.float()
    scores = (torch.einsum("thd,sd->hts", qvf, vf)
              + torch.einsum("thd,sd->hts", qf, kf)) * softmax_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,sd->thd", probs, vf)


def test_mla_decode_bnsd_cross_batch():
    """Absorbed BNSD decode over a batch, which pins the BNSD qv batch offset.

    Sq > 1 is required: with Sq == 1 a transposed BNSD view is contiguous storage and
    carries no stride evidence of the layout.
    """
    batch, seqlen, heads, kv_len, drope, dlatent = 2, 4, 8, 64, 64, 512
    q, qv = _rand((batch, seqlen, heads, drope)), _rand((batch, seqlen, heads, dlatent))
    k_cache = _rand((batch, kv_len, 1, drope))
    v_cache = _rand((batch, kv_len, 1, dlatent))
    scale = (drope + dlatent) ** -0.5

    out = flash_attn_func(
        _to_bnsd_view(q),
        _to_bnsd_view(k_cache),
        _to_bnsd_view(v_cache),
        qv=_to_bnsd_view(qv),
        softmax_scale=scale,
    )

    assert out.shape == (batch, seqlen, heads, dlatent)
    for b in range(batch):
        ref = _ref_decode_multi(
            q.cpu()[b], qv.cpu()[b], k_cache.cpu()[b, :, 0], v_cache.cpu()[b, :, 0], scale
        )
        assert _rel_l2(out.cpu()[b], ref) < REL_L2_TOL
