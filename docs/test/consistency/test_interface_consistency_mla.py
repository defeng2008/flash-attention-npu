# Copyright (c) 2026, Minghua Shen.
"""Interface bitwise-consistency checks for MLA user scenarios (FA3 + FA4).

Requirement: for the same equivalent input, all public interfaces of one
version must return *bit-identical* tensors (``torch.equal``), for both the
prefill (unabsorbed) and the decode (absorbed) user scenario, plus a wider
``heads_q = 128`` variant of each.

Enabling configuration (why these knobs):
  * ``num_splits=1``  -> disables FlashDecode split-KV, so the per-row KV
    reduction order is identical across entry points.
  * non-paged KV      -> no page indirection reordering.
  * same math form    -> all interfaces of a version run the same form; only
    the entry layout differs (BSND dense / TND varlen / cache).
  * ``softmax_scale`` is left at the interface default, which uses the same
    ``(q.last + qv.last) ** -0.5`` formula on every entry point.

Scenario shapes mirror docs/test/perf/mla_user_scenarios.py (usershape.txt),
plus a ``heads_q = 128`` variant of each scenario.
"""

import importlib
import sys

import pytest
import torch
import torch_npu  # noqa: F401  (registers the npu device)

DTYPE = torch.bfloat16
DEVICE = "npu"

DTYPES = [torch.bfloat16, torch.float16]


@pytest.fixture(autouse=True, params=DTYPES, ids=["bf16", "fp16"])
def _precision(request, monkeypatch):
    """Run every case once per supported precision."""
    monkeypatch.setattr(sys.modules[__name__], "DTYPE", request.param)
    yield request.param


# form -> (sq, skv, heads_q, heads_kv, head_dim_q, head_dim_v, causal)
FORMS = {
    "prefill": (198, 198, 32, 32, 192, 128, True),
    "prefill_qh128": (198, 198, 128, 32, 192, 128, True),
    "decode": (1, 1025, 32, 1, 64, 512, False),
    "decode_qh128": (1, 1025, 128, 1, 64, 512, False),
}


def _mod(name):
    return importlib.import_module(name)


def _mla_inputs(form, seed=2026):
    """BSND dense tensors for one user scenario, shared by every interface."""
    sq, skv, h, hk, dq, dv, _ = FORMS[form]
    gen = torch.Generator().manual_seed(seed)

    def rnd(shape):
        return torch.randn(shape, generator=gen).to(DTYPE).to(DEVICE)

    q = rnd((1, sq, h, dq))
    k = rnd((1, skv, hk, dq))
    v = rnd((1, skv, hk, dv))
    qv = rnd((1, sq, h, dv)) if form.startswith("decode") else None
    return q, k, v, qv


def _cu_seqlens(seqlens):
    out = [0]
    for s in seqlens:
        out.append(out[-1] + s)
    return torch.tensor(out, dtype=torch.int32).to(DEVICE)


def _flat(out):
    """Drop a leading batch-1 dim so dense (1,S,H,D) matches varlen (S,H,D)."""
    return out[0] if out.dim() == 4 and out.shape[0] == 1 else out


def _assert_bitwise_equal(name, a, b):
    a, b = _flat(a), _flat(b)
    assert a.shape == b.shape, f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if torch.equal(a, b):
        return
    mismatch = (a != b).sum().item()
    max_abs = (a.float() - b.float()).abs().max().item()
    raise AssertionError(
        f"{name}: not bitwise identical; mismatched={mismatch}/{a.numel()}, "
        f"max_abs={max_abs:.3e}"
    )


def _call_func(mod, form, q, k, v, qv):
    _, _, _, _, _, _, causal = FORMS[form]
    return mod.flash_attn_func(q, k, v, qv=qv, causal=causal, num_splits=1)


def _call_varlen(mod, form, q, k, v, qv, *, fa3):
    sq, skv, _, _, _, _, causal = FORMS[form]
    cu_q, cu_k = _cu_seqlens([sq]), _cu_seqlens([skv])
    if fa3:
        # FA3: cu_seqlens are positional (q, k, v, cu_q, cu_k, maxq, maxk, ...).
        return mod.flash_attn_varlen_func(
            q[0], k[0], v[0], cu_q, cu_k, sq, skv,
            qv=None if qv is None else qv[0],
            causal=causal, num_splits=1,
        )
    # FA4: cu_seqlens are keyword arguments.
    return mod.flash_attn_varlen_func(
        q[0], k[0], v[0], qv=None if qv is None else qv[0],
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=sq, max_seqlen_k=skv,
        causal=causal, num_splits=1,
    )


def _call_kvcache(mod, form, q, k, v, qv):
    skv = FORMS[form][1]
    _, _, _, _, _, _, causal = FORMS[form]
    # k/v are the whole cache: cache_seqlens == kv_seqlen, no k_new/v_new.
    return mod.flash_attn_with_kvcache(
        q, k, v, qv=qv, cache_seqlens=skv, causal=causal, num_splits=1,
    )


@pytest.mark.parametrize("form", list(FORMS))
def test_fa3_three_interfaces_bitwise_identical(form):
    fa3 = _mod("flash_attn_npu_3")
    q, k, v, qv = _mla_inputs(form)

    outs = {
        "func": _call_func(fa3, form, q, k, v, qv),
        "varlen_func": _call_varlen(fa3, form, q, k, v, qv, fa3=True),
        "with_kvcache": _call_kvcache(fa3, form, q, k, v, qv),
    }
    base = outs["func"]
    for name, out in outs.items():
        _assert_bitwise_equal(f"fa3[{form}] func vs {name}", base, out)


@pytest.mark.parametrize("form", list(FORMS))
def test_fa4_two_interfaces_bitwise_identical(form):
    fa4 = _mod("flash_attn_npu_4")
    q, k, v, qv = _mla_inputs(form)

    base = _call_func(fa4, form, q, k, v, qv)
    other = _call_varlen(fa4, form, q, k, v, qv, fa3=False)
    _assert_bitwise_equal(f"fa4[{form}] func vs varlen_func", base, other)
