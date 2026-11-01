"""Performance baseline for ops-transformer FIA MLA (Task 0.4).

Runs the two MLA user scenarios from ``mla_user_scenarios`` through torch_npu's
``npu_fused_infer_attention_score`` (prefill) / ``..._v2`` (decode) and prints a
median/mean/min/max table. Timing is per-call with ``torch.npu.synchronize()``
before and after, warmup=10, iters=20.

Usage:
    python tests/perf/bench_mla_fia.py
    python tests/perf/bench_mla_fia.py --scenario decode_mla_split_d512
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Callable, List

import torch

try:  # torch_npu may be absent on CPU-only hosts.
    import torch_npu
except ImportError:  # pragma: no cover
    torch_npu = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mla_user_scenarios import (  # noqa: E402
    DECODE_SCENARIO,
    DECODE_SCENARIO_S2_996,
    PREFILL_SCENARIO,
    SCENARIOS,
    make_inputs,
)

WARMUP = 10
ITERS = 20


def _prefill_call(inputs: dict, scenario) -> torch.Tensor:
    """FIA v1 path, form ROPE_COMBINE_D128 (qkHeadDim>vHeadDim, no split rope)."""
    out = torch_npu.npu_fused_infer_attention_score(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        atten_mask=inputs["atten_mask"],
        actual_seq_lengths=inputs["cu_seqlens_q"].tolist(),
        actual_seq_lengths_kv=inputs["cu_seqlens_k"].tolist(),
        num_heads=scenario.num_heads_q,
        num_key_value_heads=scenario.num_heads_kv,
        scale=inputs["softmax_scale"],
        input_layout=scenario.input_layout,
        sparse_mode=3,
    )
    return out[0]


def _decode_call(inputs: dict, scenario) -> torch.Tensor:
    """FIA v2 path, form ROPE_SPLIT_D512 (split rope: query_rope/key_rope)."""
    out = torch_npu.npu_fused_infer_attention_score_v2(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        query_rope=inputs["q_rope"],
        key_rope=inputs["k_rope"],
        actual_seq_qlen=inputs["cu_seqlens_q"].tolist(),
        actual_seq_kvlen=inputs["cu_seqlens_k"].tolist(),
        num_query_heads=scenario.num_heads_q,
        num_key_value_heads=scenario.num_heads_kv,
        softmax_scale=inputs["softmax_scale"],
        input_layout=scenario.input_layout,
    )
    return out[0]


def _waiter(scenario) -> Callable[[dict], torch.Tensor]:
    return _prefill_call if scenario.form == "prefill" else _decode_call


def bench(fn: Callable[[], torch.Tensor], warmup: int = WARMUP, iters: int = ITERS) -> List[float]:
    """Return per-call latencies in ms (synchronized on both sides)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times_ms: List[float] = []
    for _ in range(iters):
        torch.npu.synchronize()
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times_ms.append((time.perf_counter() - start) * 1e3)
    return times_ms


def run_scenario(scenario) -> dict:
    inputs = make_inputs(scenario, seed=0, device="npu")
    fn = _waiter(scenario)
    # Output allocation outside the timed loop; reuse via a warmup call.
    samples = bench(lambda: fn(inputs, scenario))
    return {
        "name": scenario.name,
        "form": scenario.form,
        "fia_form": scenario.fia_form,
        "dtype": str(scenario.dtype).replace("torch.", ""),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "min": min(samples),
        "max": max(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="scenario name to run (repeatable); default: prefill + decode(1025)",
    )
    args = parser.parse_args()

    if torch_npu is None:
        print("torch_npu is not installed; skipping MLA FIA baseline.")
        return 0
    if not torch.npu.is_available():
        print("No Ascend NPU available; skipping MLA FIA baseline.")
        return 0

    torch_npu.npu.set_device(0)

    if args.scenario:
        selected = [SCENARIOS[name] for name in args.scenario]
    else:
        selected = [PREFILL_SCENARIO, DECODE_SCENARIO, DECODE_SCENARIO_S2_996]

    rows = []
    for scenario in selected:
        row = run_scenario(scenario)
        rows.append(row)
        print(
            f"[done] {row['name']}: median={row['median']:.3f} ms "
            f"mean={row['mean']:.3f} ms"
        )

    header = f"{'scenario':<34}{'dtype':<8}{'median(ms)':>12}{'mean(ms)':>12}{'min(ms)':>10}{'max(ms)':>10}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<34}{row['dtype']:<8}"
            f"{row['median']:>12.4f}{row['mean']:>12.4f}"
            f"{row['min']:>10.4f}{row['max']:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
