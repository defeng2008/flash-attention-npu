# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Interleaved MLA timing on the user-scenario shapes (``mla_user_scenarios.py``).

Three references are measured in the same process, one sample per round so clock
drift hits every implementation equally:

* ``fia``: ops-transformer ``npu_fused_infer_attention_score`` (rope-split MLA).
* ``faN_nomla``: the same FA3/FA4 kernel on the closest non-MLA shape
  (equal q/v head dim), i.e. what this repository ran before MLA existed.
* ``faN_mla``: the new MLA path.

Two gates are reported:

* ``mla / nomla <= 1.05``: the MLA extension must not regress the FA kernel.
* ``mla / fia`` is printed for context. This repository's FA kernel is slower
  than the ops-transformer FIA kernel on these small shapes with or without MLA
  (see ``faN_nomla / fia``), so it is not an MLA-attributable regression.

Usage::

    python tests/perf/bench_mla_vs_fia.py
"""

import statistics
import sys
import time

import torch
import torch_npu

sys.path.insert(0, "tests/perf")
sys.path.insert(0, ".")

from mla_user_scenarios import DECODE, PREFILL, scenario_summary  # noqa: E402

from flash_attn_npu_3 import (  # noqa: E402
    flash_attn_func as fa3_func,
    flash_attn_with_kvcache as fa3_kvcache,
)
from flash_attn_npu_4 import flash_attn_func as fa4_func  # noqa: E402

DEVICE = "npu"
DTYPE = torch.bfloat16
WARMUP = 10
ROUNDS = 50
MAX_REGRESSION = 1.05


def _rand(*shape):
    return (torch.rand(shape, device="cpu") - 0.5).to(DTYPE).to(DEVICE)


def _causal_mask():
    return torch.triu(torch.ones(2048, 2048), diagonal=1).bool().npu()


def _prefill_impls():
    s = PREFILL
    rope = 64
    half = s.head_dim_q - rope
    q = _rand(1, s.total_q, s.heads_q, s.head_dim_q)
    k = _rand(1, s.total_kv, s.heads_kv, s.head_dim_k)
    v = _rand(1, s.total_kv, s.heads_kv, s.head_dim_v)
    v_same = _rand(1, s.total_kv, s.heads_kv, s.head_dim_q)  # non-MLA: Dv == Dq
    mask = _causal_mask()

    def fia():
        torch_npu.npu_fused_infer_attention_score(
            q[..., :half].contiguous(), k[..., :half].contiguous(), v,
            atten_mask=mask,
            query_rope=q[..., half:].contiguous(), key_rope=k[..., half:].contiguous(),
            num_heads=s.heads_q, scale=s.softmax_scale, input_layout="BSND",
            num_key_value_heads=s.heads_kv, sparse_mode=3,
        )

    return {
        "fia": fia,
        "fa3_nomla": lambda: fa3_func(q, k, v_same, softmax_scale=s.softmax_scale, causal=True),
        "fa3_mla": lambda: fa3_func(q, k, v, softmax_scale=s.softmax_scale, causal=True),
        "fa4_nomla": lambda: fa4_func(q, k, v_same, softmax_scale=s.softmax_scale, causal=True),
        "fa4_mla": lambda: fa4_func(q, k, v, softmax_scale=s.softmax_scale, causal=True),
    }


def _decode_impls():
    s = DECODE
    nope = s.head_dim_q - s.rope_dim
    q_rope = _rand(1, s.total_q, s.heads_q, s.rope_dim)
    qv = _rand(1, s.total_q, s.heads_q, nope)
    k_rope = _rand(1, s.total_kv, s.heads_kv, s.rope_dim)
    v = _rand(1, s.total_kv, s.heads_kv, s.head_dim_v)
    # Non-MLA analogue: same k/v length and head count, one 512-wide head dim.
    q_nomla = _rand(1, s.total_q, s.heads_q, s.head_dim_v)

    def fia():
        torch_npu.npu_fused_infer_attention_score(
            qv, v, v,
            query_rope=q_rope, key_rope=k_rope,
            num_heads=s.heads_q, scale=s.softmax_scale, input_layout="BSND",
            num_key_value_heads=s.heads_kv, sparse_mode=0,
        )

    return {
        "fia": fia,
        "fa3_nomla": lambda: fa3_kvcache(
            q_nomla, v, v, cache_seqlens=s.total_kv, softmax_scale=s.softmax_scale
        ),
        "fa3_mla": lambda: fa3_kvcache(
            q_rope, k_rope, v, qv=qv, cache_seqlens=s.total_kv, softmax_scale=s.softmax_scale
        ),
        "fa4_nomla": lambda: fa4_func(q_nomla, v, v, softmax_scale=s.softmax_scale),
        "fa4_mla": lambda: fa4_func(q_rope, k_rope, v, qv=qv, softmax_scale=s.softmax_scale),
    }


def _measure(impls):
    for impl in impls.values():
        for _ in range(WARMUP):
            impl()
    torch.npu.synchronize()
    samples = {name: [] for name in impls}
    for _ in range(ROUNDS):
        for name, impl in impls.items():
            start = time.perf_counter()
            impl()
            torch.npu.synchronize()
            samples[name].append(time.perf_counter() - start)
    return {name: statistics.median(values) * 1e3 for name, values in samples.items()}


def main():
    worst_mla = 0.0
    for scenario, build_impls in ((PREFILL, _prefill_impls), (DECODE, _decode_impls)):
        print(scenario_summary(scenario))
        timings = _measure(build_impls())
        base = timings["fia"]
        for name in ("fia", "fa3_nomla", "fa3_mla", "fa4_nomla", "fa4_mla"):
            print(f"  {name:10s} {timings[name]:8.3f} ms   {timings[name] / base:6.3f}x vs FIA")
        print()
        for name in ("fa3_mla", "fa4_mla"):
            ratio = timings[name] / timings[name.replace("_mla", "_nomla")]
            status = "PASS" if ratio <= MAX_REGRESSION else "FAIL"
            print(f"  {name} / non-MLA {ratio:6.3f}x  (gate {MAX_REGRESSION:.2f}x)  {status}")
            worst_mla = max(worst_mla, ratio)
        print()

    print(f"worst MLA vs non-MLA ratio {worst_mla:.3f}x")
    return 0 if worst_mla <= MAX_REGRESSION else 1


if __name__ == "__main__":
    raise SystemExit(main())
