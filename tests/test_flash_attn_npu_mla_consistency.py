# Copyright (c) 2026, Minghua Shen.
# MLA training/inference consistency and precision acceptance tests.

"""Task 3 of the MLA forward spec: training/inference consistency and precision.

Covers
  3.1  prefill (unabsorbed) vs decode (absorbed) consistency on a shared latent
       ``c_KV``, plus an independent FP64/FP32 CPU golden cross-check.
  3.2  the ``softmax_scale`` default-derivation contract for both MLA forms.
  3.3  rejection paths (forward-only / no descale / no BNSD / no invalid layout).

The two MLA forms are related through the identity
``k_nope = c_KV[..., :128]``, ``v_prefill = c_KV[..., :128]`` and
``W_UK = [I_128 | 0_384]`` (so ``q_absorbed = [q_nope | 0_384]``). Under that
construction the 192-dot prefill score equals the (512 + 64)-dot absorbed decode
score, and the first 128 decode output channels equal the prefill output.

The two entries are called with an *explicit* shared ``softmax_scale`` because
their defaults differ (192**-0.5 vs 576**-0.5, see ``test_mla_default_scale``).
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401

import flash_attn_npu_3 as fa3
import flash_attn_npu_4 as fa4


DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]
# Cross-form relative L2 budget. bf16 carries 8 mantissa bits, fp16 10.
REL_TOL = {torch.bfloat16: 5e-3, torch.float16: 2e-3}
# The prefill MLA scale; both forms are driven with this value for consistency.
PREFILL_SCALE = 192 ** -0.5
COMPRESSED_DEFAULT_SCALE = 576 ** -0.5


@pytest.fixture(scope="module", autouse=True)
def _npu_guard():
    if not torch.npu.is_available():
        pytest.skip("MLA requires Ascend 910B/C")
    if not torch.npu.get_device_name(0).startswith(("Ascend910B", "Ascend910C")):
        pytest.skip("MLA requires Ascend 910B/C")
    # The small CPU goldens cost much more with a machine-wide thread pool.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    torch.npu.synchronize()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rel_l2(actual, expected):
    """``||actual - expected||_2 / max(||expected||_2, eps)`` in FP64."""
    a = actual.double()
    b = expected.double()
    denom = torch.linalg.vector_norm(b).item()
    return (torch.linalg.vector_norm(a - b) / max(denom, 1e-12)).item()


def _make_latent(dtype, batch, seq, heads, seed=0):
    """Shared latent tensors plus the two MLA input forms built from them."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return (torch.randn(*shape, generator=gen) * 0.1).to(dtype).to("npu")

    q_nope = randn(batch, seq, heads, 128)
    q_rope = randn(batch, seq, heads, 64)
    k_nope = randn(batch, seq, 1, 128)
    k_rope = randn(batch, seq, 1, 64)
    c_tail = randn(batch, seq, 1, 384)  # c_KV[..., 128:] (killed by the zero q tail)
    zeros = torch.zeros(batch, seq, heads, 384, dtype=dtype, device="npu")

    c_kv = torch.cat([k_nope, c_tail], dim=-1)  # (B, S, 1, 512)
    prefill = dict(
        q=torch.cat([q_nope, q_rope], dim=-1),  # 192 = 128 nope + 64 rope
        k=torch.cat([k_nope, k_rope], dim=-1),  # 192
        v=k_nope,  # 128
    )
    decode = dict(
        q=q_rope,  # 64 rope
        qv=torch.cat([q_nope, zeros], dim=-1),  # q_nope @ [I_128 | 0_384] = [q_nope | 0]
        k=k_rope,  # 64
        v=c_kv,  # 512
    )
    return prefill, decode


def _attention_golden(q, k, v, scale, causal, upcast):
    """Pure-torch MLA attention. ``upcast=True`` gives the FP64 reference, else
    the reference is computed in the input dtype (the "pt" comparison)."""
    dtype = torch.float64 if upcast else q.dtype
    sq, sk = q.shape[1], k.shape[1]
    groups = q.shape[2] // k.shape[2]
    query = q.detach().to("cpu", dtype).transpose(1, 2)
    key = k.detach().to("cpu", dtype).repeat_interleave(groups, dim=2).transpose(1, 2)
    value = v.detach().to("cpu", dtype).repeat_interleave(groups, dim=2).transpose(1, 2)
    if upcast:
        scores = (query @ key.transpose(-1, -2)) * scale
    else:
        scores = ((query * scale) @ key.transpose(-1, -2)).float()
    if causal:
        mask = torch.ones(sq, sk, dtype=torch.bool).triu(diagonal=sk - sq + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    if upcast:
        weights = scores.softmax(-1)
    else:
        weights = scores.softmax(-1).to(dtype)
    return (weights @ value).transpose(1, 2)


def _decode_goldens(decode, scale, causal):
    """Fold [c_KV | k_rope] and [qv | q_rope] into the 576-wide absorbed form."""
    q576 = torch.cat([decode["qv"], decode["q"]], dim=-1)
    k576 = torch.cat([decode["v"], decode["k"]], dim=-1)
    out = []
    for upcast in (True, False):
        out.append(_attention_golden(q576, k576, decode["v"], scale, causal, upcast))
    return out  # (fp64, input-dtype)


def _assert_close_to_golden(actual, goldens, name):
    """NPU output must track both the FP64 and the input-dtype CPU goldens."""
    ref, pt = goldens
    rel_ref = _rel_l2(actual.cpu(), ref)
    rel_pt = _rel_l2(actual.cpu(), pt)
    print(f"  [{name}] rel-L2 vs fp64={rel_ref:.3e} vs input-dtype={rel_pt:.3e}")
    assert rel_ref < 5e-3, f"{name}: NPU vs FP64 golden relative L2 {rel_ref:.3e} >= 5e-3"
    assert rel_pt < 5e-3, f"{name}: NPU vs input-dtype golden relative L2 {rel_pt:.3e} >= 5e-3"


# --------------------------------------------------------------------------- #
# 3.1  prefill (unabsorbed) vs decode (absorbed) consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("seq,heads", [(198, 32), (16, 8)], ids=["S198H32", "S16H8"])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
def test_prefill_decode_consistency(dtype, seq, heads, causal):
    batch = 1
    prefill, decode = _make_latent(dtype, batch, seq, heads)
    lengths = torch.full((batch,), seq, dtype=torch.int32, device="npu")

    out_p = fa3.flash_attn_func(
        prefill["q"], prefill["k"], prefill["v"],
        softmax_scale=PREFILL_SCALE, causal=causal, return_attn_probs=True,
    )[0]
    out_d = fa3.flash_attn_with_kvcache(
        decode["q"], decode["k"], decode["v"], qv=decode["qv"],
        cache_seqlens=lengths, softmax_scale=PREFILL_SCALE, causal=causal,
        return_softmax_lse=True,
    )[0]

    assert out_p.shape == (batch, seq, heads, 128)
    assert out_d.shape == (batch, seq, heads, 512)

    # Core assertion: decode's first 128 channels equal the prefill output.
    rel = _rel_l2(out_d[..., :128].cpu(), out_p.cpu())
    print(f"  [prefill-vs-decode {dtype} S={seq} H={heads} causal={causal}] rel-L2={rel:.3e}")
    assert rel < REL_TOL[dtype], f"prefill vs decode relative L2 {rel} >= {REL_TOL[dtype]}"

    # Independent CPU golden cross-check for both forms.
    gp = [
        _attention_golden(prefill["q"], prefill["k"], prefill["v"], PREFILL_SCALE, causal, u)
        for u in (True, False)
    ]
    gd = _decode_goldens(decode, PREFILL_SCALE, causal)
    # The construction itself is exact in FP64: both forms share the same math.
    assert _rel_l2(gd[0][..., :128], gp[0]) < 1e-6
    _assert_close_to_golden(out_p.cpu(), gp, "prefill")
    _assert_close_to_golden(out_d.cpu(), gd, "decode")


# --------------------------------------------------------------------------- #
# 3.2  softmax_scale default-derivation contract
# --------------------------------------------------------------------------- #
def test_mla_scale_default_derivation():
    # Documented/implemented default: 1 / sqrt(q.last + qv.last).
    # Compressed sees 64 + 512 = 576, prefill sees 192. The ratio is sqrt(3).
    assert math.isclose(COMPRESSED_DEFAULT_SCALE * math.sqrt(3.0), PREFILL_SCALE, rel_tol=1e-12)
    assert not math.isclose(COMPRESSED_DEFAULT_SCALE, PREFILL_SCALE)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mla_default_scale(dtype):
    batch, seq, heads = 1, 8, 16
    gen = torch.Generator().manual_seed(7)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(dtype).to("npu")

    # Unit-variance inputs make the softmax sensitive to the logit temperature.
    q_rope = randn(batch, seq, heads, 64)
    q_abs = randn(batch, seq, heads, 512)
    k_rope = randn(batch, seq, 1, 64)
    c_kv = randn(batch, seq, 1, 512)
    lengths = torch.full((batch,), seq, dtype=torch.int32, device="npu")

    def decode(scale):
        return fa3.flash_attn_with_kvcache(
            q_rope, k_rope, c_kv, qv=q_abs, cache_seqlens=lengths,
            softmax_scale=scale, causal=False, return_softmax_lse=True,
        )[0]

    default = decode(None)
    # The implementation of the compressed form defaults to 576**-0.5.
    assert torch.equal(default, decode(COMPRESSED_DEFAULT_SCALE))
    # ... which is sqrt(3) *smaller* than the prefill (192**-0.5) convention.
    rel = _rel_l2(decode(PREFILL_SCALE).cpu(), default.cpu())
    assert rel > 1e-2, f"compressed default scale unexpectedly equals 192**-0.5 (rel={rel})"

    # Prefill (no qv) defaults to 192**-0.5.
    q_p = randn(batch, seq, heads, 192)
    k_p = randn(batch, seq, 1, 192)
    v_p = randn(batch, seq, 1, 128)
    assert torch.equal(
        fa3.flash_attn_func(q_p, k_p, v_p, causal=False, return_attn_probs=True)[0],
        fa3.flash_attn_func(q_p, k_p, v_p, softmax_scale=PREFILL_SCALE, causal=False,
                            return_attn_probs=True)[0],
    )


# --------------------------------------------------------------------------- #
# 3.3  rejection paths
# --------------------------------------------------------------------------- #
def test_mla_describe_rejected():
    # Quantized MLA is unsupported: any descale tensor is rejected (v3).
    prefill, _ = _make_latent(torch.bfloat16, 1, 8, 4)
    descale = torch.ones(1, dtype=torch.float32, device="npu")
    with pytest.raises(RuntimeError, match="descale"):
        fa3.flash_attn_func(
            prefill["q"], prefill["k"], prefill["v"], q_descale=descale, softmax_scale=PREFILL_SCALE
        )


def test_mla_invalid_layout_rejected():
    # The native MLA host accepts only TND(0)/BSND(1); BNSD(2) and others fail.
    _, decode = _make_latent(torch.bfloat16, 1, 8, 4)
    for layout in (2, 3):
        with pytest.raises(RuntimeError, match="layout must be TND or BSND"):
            torch.ops.flash_attn_npu_3._flash_attn_forward(
                decode["q"], decode["k"], decode["v"], qv=decode["qv"],
                softmax_scale=PREFILL_SCALE, layout=layout,
            )


# --------------------------------------------------------------------------- #
# 3.1 (extension)  the same consistency through the FA4 public API
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("seq,heads", [(198, 32), (16, 8)], ids=["S198H32", "S16H8"])
def test_prefill_decode_consistency_v4(dtype, seq, heads):
    batch = 1
    prefill, decode = _make_latent(dtype, batch, seq, heads)
    lengths = torch.full((batch,), seq, dtype=torch.int32, device="npu")

    # FA4 exposes only the varlen entry point (dense BSND KV rides through it via
    # seqused_k) and returns a bare ``out`` tensor, not an ``(out, lse)`` tuple.
    out_p = fa4.flash_attn_varlen_func(
        prefill["q"], prefill["k"], prefill["v"],
        cu_seqlens_q=None, seqused_k=lengths, softmax_scale=PREFILL_SCALE, causal=False,
    )
    out_d = fa4.flash_attn_varlen_func(
        decode["q"], decode["k"], decode["v"], qv=decode["qv"],
        cu_seqlens_q=None, seqused_k=lengths, softmax_scale=PREFILL_SCALE, causal=False,
    )

    assert out_p.shape == (batch, seq, heads, 128)
    assert out_d.shape == (batch, seq, heads, 512)
    rel = _rel_l2(out_d[..., :128].cpu(), out_p.cpu())
    print(f"  [v4 prefill-vs-decode {dtype} S={seq} H={heads}] rel-L2={rel:.3e}")
    assert rel < REL_TOL[dtype], f"v4 prefill vs decode relative L2 {rel} >= {REL_TOL[dtype]}"
