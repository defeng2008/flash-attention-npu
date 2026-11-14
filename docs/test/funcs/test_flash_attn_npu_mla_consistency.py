# Copyright (c) 2026, Minghua Shen.
"""MLA train/infer consistency and precision acceptance (FA3 + FA4).

The same latent KV state is fed through the two shapes MLA accepts:

* prefill / unabsorbed: ``q = [q_nope(128) | q_rope(64)]``, ``k = [k_nope | k_rope]``
  computed from the latent state, ``v = latent @ W_uv^T`` (128), so the kernel runs
  a plain 192-wide QK matmul plus a 128-wide PV matmul.
* decode / absorbed: ``qv = q_nope @ W_uk`` (512) attends directly to the latent
  state, ``q = q_rope`` (64), ``k = k_rope`` (64), ``v = latent`` (512).

Both are the same bilinear form, so the absorbed output projected by ``W_uv`` must
match the unabsorbed output. An independent FP64 CPU golden validates each path.
"""

import importlib

import pytest
import torch
import torch_npu

DEVICE = "npu"
DTYPE = torch.bfloat16

# bf16 relative-L2 acceptance threshold (see the MLA spec).
REL_L2_TOL = 5e-3

BACKENDS = ["flash_attn_npu_3", "flash_attn_npu_4"]

LATENT = 512
NOPE = 128
ROPE = 64
HEADS = 8
SEQLEN = 197


def _rel_l2(actual, ref):
    a = actual.float().reshape(-1)
    r = ref.float().reshape(-1)
    return ((a - r).pow(2).sum() / r.pow(2).sum().clamp_min(1e-12)).sqrt().item()


def _fp64_case(seed=0):
    """FP64 source tensors. Cast them to bf16 before handing them to a kernel."""
    gen = torch.Generator().manual_seed(seed)

    def rnd(shape, scale=1.0):
        return torch.randn(shape, generator=gen, dtype=torch.float64) * scale

    w_uk = rnd((NOPE, LATENT), LATENT ** -0.5)
    w_uv = rnd((NOPE, LATENT), LATENT ** -0.5)
    latent_kv = rnd((SEQLEN, LATENT))
    q_nope = rnd((HEADS, NOPE), NOPE ** -0.5)
    q_rope = rnd((HEADS, ROPE), ROPE ** -0.5)
    k_rope = rnd((SEQLEN, ROPE), ROPE ** -0.5)
    return w_uk, w_uv, latent_kv, q_nope, q_rope, k_rope


def _bf16_npu(x):
    return x.to(DTYPE).to(DEVICE)


def _prefill_tensors():
    """Kernel inputs for the unabsorbed path and their FP64 golden counterparts."""
    w_uk, w_uv, latent_kv, q_nope, q_rope, k_rope = _fp64_case()
    q = torch.cat([q_nope, q_rope], dim=-1)  # (H, 192)
    k = torch.cat([latent_kv @ w_uk.T, k_rope], dim=-1)  # (L, 192)
    v = latent_kv @ w_uv.T  # (L, 128)
    # Golden runs in FP64 on exactly the values the kernel sees (bf16 rounded).
    q64 = q.to(DTYPE).double()
    k64 = k.to(DTYPE).double()
    v64 = v.to(DTYPE).double()
    return (
        _bf16_npu(q).unsqueeze(0).unsqueeze(0),  # (1, 1, H, 192)
        _bf16_npu(k).unsqueeze(0).unsqueeze(2),  # (1, L, 1, 192)
        _bf16_npu(v).unsqueeze(0).unsqueeze(2),  # (1, L, 1, 128)
        q64,
        k64,
        v64,
    )


def _decode_tensors():
    """Kernel inputs for the absorbed path and their FP64 golden counterparts."""
    w_uk, w_uv, latent_kv, q_nope, q_rope, k_rope = _fp64_case()
    qv = q_nope @ w_uk  # (H, 512)
    qv64 = qv.to(DTYPE).double()
    c64 = latent_kv.to(DTYPE).double()
    return (
        _bf16_npu(q_rope).unsqueeze(0).unsqueeze(0),  # (1, 1, H, 64)
        _bf16_npu(k_rope).unsqueeze(0).unsqueeze(2),  # (1, L, 1, 64)
        _bf16_npu(latent_kv).unsqueeze(0).unsqueeze(2),  # (1, L, 1, 512)
        _bf16_npu(qv).unsqueeze(0).unsqueeze(0),  # (1, 1, H, 512)
        q_rope.to(DTYPE).double(),
        k_rope.to(DTYPE).double(),
        c64,
        qv64,
        w_uv.to(DTYPE).double(),
    )


def _golden_prefill(q64, k64, v64, scale):
    scores = torch.einsum("hd,sd->hs", q64, k64) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,sd->hd", probs, v64)


def _golden_decode(q_rope64, k_rope64, c64, qv64, scale):
    scores = (qv64 @ c64.T + q_rope64 @ k_rope64.T) * scale
    probs = torch.softmax(scores, dim=-1)
    return probs @ c64


def _prefill_out(backend, q, k, v, scale):
    mod = importlib.import_module(backend)
    return mod.flash_attn_func(q, k, v, softmax_scale=scale)


def _decode_out(backend, q_rope, k_rope, c, qv, scale):
    mod = importlib.import_module(backend)
    if backend == "flash_attn_npu_3":
        # FA3 keeps the kvcache entry point for absorbed decode.
        return mod.flash_attn_with_kvcache(
            q_rope, k_rope, c, qv=qv, cache_seqlens=SEQLEN, softmax_scale=scale
        )
    return mod.flash_attn_func(q_rope, k_rope, c, qv=qv, softmax_scale=scale)


# --------------------------------------------------------------------------
# train (unabsorbed) vs infer (absorbed)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_prefill_decode_consistency(backend):
    """Absorbed decode projected by W_uv must reproduce the unabsorbed prefill."""
    scale = (NOPE + ROPE) ** -0.5
    q, k, v, _, _, _ = _prefill_tensors()
    q_rope, k_rope, c, qv, _, _, _, _, w_uv64 = _decode_tensors()

    out_pre = _prefill_out(backend, q, k, v, scale)
    out_dec = _decode_out(backend, q_rope, k_rope, c, qv, scale)

    assert out_pre.shape == (1, 1, HEADS, NOPE)
    assert out_dec.shape == (1, 1, HEADS, LATENT)
    projected = out_dec.cpu().double()[0, 0] @ w_uv64.T
    assert _rel_l2(out_pre.cpu(), projected) < REL_L2_TOL


# --------------------------------------------------------------------------
# independent FP64 CPU golden
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_prefill_matches_fp64_golden(backend):
    scale = (NOPE + ROPE) ** -0.5
    q, k, v, q64, k64, v64 = _prefill_tensors()

    out = _prefill_out(backend, q, k, v, scale)

    golden = _golden_prefill(q64, k64, v64, scale)
    assert _rel_l2(out.cpu(), golden) < REL_L2_TOL


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_decode_matches_fp64_golden(backend):
    scale = (NOPE + ROPE) ** -0.5
    q_rope, k_rope, c, qv, q_rope64, k_rope64, c64, qv64, _ = _decode_tensors()

    out = _decode_out(backend, q_rope, k_rope, c, qv, scale)

    golden = _golden_decode(q_rope64, k_rope64, c64, qv64, scale)
    assert _rel_l2(out.cpu(), golden) < REL_L2_TOL


# --------------------------------------------------------------------------
# softmax_scale conventions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_prefill_default_scale_is_192(backend):
    """The unabsorbed default must be Dq ** -0.5 == 192 ** -0.5."""
    q, k, v, q64, k64, v64 = _prefill_tensors()

    out = importlib.import_module(backend).flash_attn_func(q, k, v)

    golden = _golden_prefill(q64, k64, v64, (NOPE + ROPE) ** -0.5)
    assert _rel_l2(out.cpu(), golden) < REL_L2_TOL


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_decode_default_scale_is_576(backend):
    """The absorbed default must be (q.last + qv.last) ** -0.5 == 576 ** -0.5.

    This is deliberately *not* 192 ** -0.5: the absorbed score already covers the
    full 576-wide latent-plus-rope inner product.
    """
    q_rope, k_rope, c, qv, q_rope64, k_rope64, c64, qv64, _ = _decode_tensors()

    out = _decode_out(backend, q_rope, k_rope, c, qv, None)

    golden = _golden_decode(q_rope64, k_rope64, c64, qv64, (ROPE + LATENT) ** -0.5)
    assert _rel_l2(out.cpu(), golden) < REL_L2_TOL


# --------------------------------------------------------------------------
# layout acceptance and rejection paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_accepts_bnsd(backend):
    """Contiguous BNSD handed out as a transpose(1, 2) view must match the BSND call.

    The two calls carry identical values, so any difference is a layout-addressing bug
    (a wrong batch/head stride would silently rotate the output).
    """
    batch, seqlen, heads, heads_k, dq, dv = 2, 32, 8, 4, 192, 128
    mod = importlib.import_module(backend)

    def rand_bsnd(shape):
        return (torch.rand(shape, device="cpu") * 2.0 - 1.0).to(DTYPE).to(DEVICE)

    def to_bnsd_view(x):
        return x.transpose(1, 2).contiguous().transpose(1, 2)

    q = rand_bsnd((batch, seqlen, heads, dq))
    k = rand_bsnd((batch, seqlen, heads_k, dq))
    v = rand_bsnd((batch, seqlen, heads_k, dv))
    scale = dq ** -0.5

    ref = mod.flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
    out = mod.flash_attn_func(
        to_bnsd_view(q), to_bnsd_view(k), to_bnsd_view(v), softmax_scale=scale, causal=True
    )

    # The BNSD output keeps the caller's (B, S, H, dv) view shape.
    assert out.shape == ref.shape
    assert _rel_l2(out.cpu(), ref.cpu()) < REL_L2_TOL


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_backward_unsupported(backend):
    """MLA is forward-only here: the FAG bwd kernel rejects unequal q/v head dims."""
    scale = (NOPE + ROPE) ** -0.5
    q, k, v, _, _, _ = _prefill_tensors()
    mod = importlib.import_module(backend)

    out = mod.flash_attn_func(q.requires_grad_(True), k, v, softmax_scale=scale)
    # The Python bwd wrapper rejects the unequal head dims before the kernel.
    with pytest.raises((RuntimeError, ValueError)):
        out.sum().backward()


def test_mla_rejects_gather_kv_indices():
    """FA4 rejects the sparse-MLA gather entry outright."""
    mod = importlib.import_module("flash_attn_npu_4")
    q, k, v, _, _, _ = _prefill_tensors()
    gather = torch.zeros((1, HEADS, 1), dtype=torch.int32, device=DEVICE)

    with pytest.raises(RuntimeError):
        mod.flash_attn_func(q, k, v, gather_kv_indices=gather)


def test_mla_rejects_invalid_layout():
    """The packed FA4 host entry only accepts layout 0 (TND), 1 (BSND), 2 (BNSD)."""
    interface = importlib.import_module("flash_attn_npu_4.flash_attn_npu_interface")
    q, k, v, _, _, _ = _prefill_tensors()

    with pytest.raises(RuntimeError):
        interface._flash_attn_forward(q, k, v, softmax_scale=(NOPE + ROPE) ** -0.5, layout=7)
