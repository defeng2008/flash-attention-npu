"""Performance acceptance for the FA3/FA4 MLA forward (Task 4, reworked).

Compares flash-attn-npu **v3** and **v4** against the ops-transformer FIA MLA
baseline (``bench_mla_fia.py``) on the real user scenarios from
``mla_user_scenarios``.

Why this rewrite
----------------
The previous version compared in-session flash-attn numbers against *hardcoded*
FIA medians measured in a different session. Cross-session clock / thermal drift
of 5-7% is comparable to the 5% acceptance threshold, so the verdicts were not
trustworthy. This version builds **both** input sets in the same process and
measures **FIA, v3 and v4 interleaved** across ``ROUNDS`` rounds, so any drift is
common-mode and cancels inside a round. The reported statistic is the
**per-round ratio** ``ours_median / fia_median`` (median over rounds) plus the
round-to-round dispersion (min / max).

Two timing口径 are reported (both interleaved per round):

  * ``per-call`` : ``bench`` -- ``torch.npu.synchronize()`` before and after every
    call (matches how a user times an end-to-end single call).
  * ``steady``   : ``bench_steady`` -- many back-to-back calls amortized with the
    launch queue primed (device / steady-state cost with Python dispatch hidden
    behind the device).

requirement #5 -- "<=5% vs the ops-transformer interface" -- is judged on both;
the ``steady`` ratio is the recommended primary because it removes the per-call
``synchronize`` round-trip and reflects the operator rather than the wrapper.
See the note printed at the end of a run. NOTE for these small shapes both口径
are actually *host-dispatch bound* (see ``probe_mla_v4_overhead.py``), so neither
isolates pure kernel time.

Timing details: inputs are built once outside the loops; v3 decode reuses a
``scheduler_metadata`` record precomputed outside the loops; tensors are never
rebuilt between rounds; a ``torch.npu.synchronize()`` precedes every round.

Usage:
    python tests/perf/bench_mla_flash_attn.py
    python tests/perf/bench_mla_flash_attn.py --backend v3
    python tests/perf/bench_mla_flash_attn.py --scenario decode_mla_split_d512
    python tests/perf/bench_mla_flash_attn.py --rounds 9 --json tests/perf/mla_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List

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

import bench_mla_fia as fia_baseline  # noqa: E402

WARMUP = 10
ITERS = 20          # per-call samples per round
ROUNDS = 7          # interleaved rounds (drift is common-mode inside a round)
STEADY_BATCHES = 3
STEADY_ITERS = 50
PASS_THRESHOLD_PCT = 5.0

BACKENDS = ("v3", "v4")
BASELINE = "fia"


def _contig(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous()


def build_tensors(scenario, device: str = "npu") -> dict:
    """Build the once-only input tensors for a scenario in the MLA call contract."""
    ins = make_inputs(scenario, seed=0, device=device)
    if scenario.form == "prefill":
        # Prefill (unabsorbed): q/k last dim 192, v last dim 128, qv omitted.
        return dict(
            q=_contig(ins["q"]),
            k=_contig(ins["k"]),
            v=_contig(ins["v"]),
            cu_seqlens_q=ins["cu_seqlens_q"],
            cu_seqlens_k=ins["cu_seqlens_k"],
        )
    # Decode (compressed/absorbed): q/k last dim 64 (rope), qv/v last dim 512.
    # make_inputs splits the combined 576 into content ``q``/``k`` and ``*_rope``;
    # the MLA contract maps the 64-dim rope part to q/k and the 512-dim content
    # part to qv.
    seqlen = scenario.total_kv
    return dict(
        q=_contig(ins["q_rope"]).unsqueeze(0),        # (1, 1, 32, 64)
        k=_contig(ins["k_rope"]).unsqueeze(0),        # (1, seqlen, 1, 64)
        v=_contig(ins["v"]).unsqueeze(0),             # (1, seqlen, 1, 512)
        qv=_contig(ins["q"]).unsqueeze(0),            # (1, 1, 32, 512)
        cache_seqlens=torch.tensor([seqlen], dtype=torch.int32, device=device),
    )


def build_fia_call(scenario, ins: dict) -> Callable[[], torch.Tensor]:
    call = fia_baseline._prefill_call if scenario.form == "prefill" else fia_baseline._decode_call
    return lambda: call(ins, scenario)


def _v3_prefill(scenario, t: dict) -> Callable[[], torch.Tensor]:
    import flash_attn_npu_3 as fa3

    def fn() -> torch.Tensor:
        return fa3.flash_attn_varlen_func(
            t["q"],
            t["k"],
            t["v"],
            t["cu_seqlens_q"],
            t["cu_seqlens_k"],
            max_seqlen_q=scenario.total_q,
            max_seqlen_k=scenario.total_kv,
            softmax_scale=scenario.softmax_scale,
            causal=True,
        )

    return fn


def _v3_decode(scenario, t: dict) -> Callable[[], torch.Tensor]:
    import flash_attn_npu_3 as fa3

    # Precompute the scheduling record once (outside the timed loops).
    scheduler_metadata = fa3.get_scheduler_metadata(
        1,
        1,
        scenario.total_kv,
        scenario.num_heads_q,
        scenario.num_heads_kv,
        t["q"].shape[-1],
        t["cache_seqlens"],
        qkv_dtype=scenario.dtype,
        headdim_v=t["v"].shape[-1],
        causal=False,
        num_splits=0,
    )

    def fn() -> torch.Tensor:
        return fa3.flash_attn_with_kvcache(
            t["q"],
            t["k"],
            t["v"],
            qv=t["qv"],
            cache_seqlens=t["cache_seqlens"],
            softmax_scale=scenario.softmax_scale,
            causal=False,
            scheduler_metadata=scheduler_metadata,
            num_splits=0,
        )

    return fn


def _v4_prefill(scenario, t: dict) -> Callable[[], torch.Tensor]:
    import flash_attn_npu_4 as fa4

    def fn() -> torch.Tensor:
        out = fa4.flash_attn_varlen_func(
            t["q"],
            t["k"],
            t["v"],
            cu_seqlens_q=t["cu_seqlens_q"],
            cu_seqlens_k=t["cu_seqlens_k"],
            max_seqlen_q=scenario.total_q,
            max_seqlen_k=scenario.total_kv,
            softmax_scale=scenario.softmax_scale,
            causal=True,
        )
        return out[0] if isinstance(out, tuple) else out

    return fn


def _v4_decode(scenario, t: dict) -> Callable[[], torch.Tensor]:
    import flash_attn_npu_4 as fa4

    def fn() -> torch.Tensor:
        # FA4 has no with_kvcache; dense BSND lengths travel through seqused_k.
        out = fa4.flash_attn_varlen_func(
            t["q"],
            t["k"],
            t["v"],
            qv=t["qv"],
            seqused_k=t["cache_seqlens"],
            softmax_scale=scenario.softmax_scale,
            causal=False,
        )
        return out[0] if isinstance(out, tuple) else out

    return fn


def make_call(backend: str, scenario, t: dict) -> Callable[[], torch.Tensor]:
    if backend == "v3":
        return _v3_prefill(scenario, t) if scenario.form == "prefill" else _v3_decode(scenario, t)
    if backend == "v4":
        return _v4_prefill(scenario, t) if scenario.form == "prefill" else _v4_decode(scenario, t)
    raise ValueError(f"unknown backend: {backend}")


def bench(fn: Callable[[], torch.Tensor], warmup: int = WARMUP, iters: int = ITERS) -> List[float]:
    """Per-call latencies in ms (synchronized on both sides), same as the FIA baseline."""
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


def bench_steady(fn: Callable[[], torch.Tensor], warmup: int = WARMUP) -> float:
    """Amortized ms/call with a primed launch queue (device-bound, wrapper hidden).

    Times ``STEADY_ITERS`` back-to-back calls per batch and returns the median
    batch cost; Python dispatch overlaps the device, so this separates the kernel
    cost from the fixed per-call sync/dispatch overhead of ``bench``.
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    batches: List[float] = []
    for _ in range(STEADY_BATCHES):
        start = time.perf_counter()
        for _ in range(STEADY_ITERS):
            fn()
        torch.npu.synchronize()
        batches.append((time.perf_counter() - start) * 1e3 / STEADY_ITERS)
    return statistics.median(batches)


def median(xs: List[float]) -> float:
    return statistics.median(xs)


def measure_interleaved(scenario, backends) -> dict:
    """R rounds; inside a round measure FIA then each backend, for both口径.

    Returns ``{metric: {name: {"rounds": [...], "values": [...]}}}`` where
    ``rounds`` are per-round medians and ``values`` the raw per-round samples used
    for the ratio statistic.
    """
    t = build_tensors(scenario, device="npu")
    ins = make_inputs(scenario, seed=0, device="npu")

    calls: Dict[str, Callable[[], torch.Tensor]] = {BASELINE: build_fia_call(scenario, ins)}
    for b in backends:
        calls[b] = make_call(b, scenario, t)

    # One full warmup of every call before the rounds so first-touch/JIT is out.
    for name in [BASELINE, *backends]:
        for _ in range(WARMUP):
            calls[name]()
    torch.npu.synchronize()

    per_round = {
        "per-call": {name: [] for name in calls},
        "steady": {name: [] for name in calls},
    }
    for _ in range(ROUNDS):
        torch.npu.synchronize()
        # per-call: median of ITERS synchronized samples, then steady, per backend.
        for name in [BASELINE, *backends]:
            per_round["per-call"][name].append(median(bench(calls[name], warmup=0)))
        for name in [BASELINE, *backends]:
            per_round["steady"][name].append(bench_steady(calls[name], warmup=2))
    return per_round


def summarize(scenario, per_round, backends) -> dict:
    out = {"name": scenario.name, "form": scenario.form, "fia_form": scenario.fia_form, "metrics": {}}
    for metric in ("per-call", "steady"):
        data = per_round[metric]
        fia_rounds = data[BASELINE]
        entry = {
            "fia_median": median(fia_rounds),
            "fia_min": min(fia_rounds),
            "fia_max": max(fia_rounds),
            "backends": {},
        }
        for b in backends:
            rounds = data[b]
            ratios = [o / f for o, f in zip(rounds, fia_rounds)]
            r_med = median(ratios)
            entry["backends"][b] = {
                "median": median(rounds),
                "min": min(rounds),
                "max": max(rounds),
                "ratio_median": r_med,
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
                "degrade_pct": (r_med - 1.0) * 100.0,
                "pass": bool(r_med <= 1.0 + PASS_THRESHOLD_PCT / 100.0),
                "rounds": rounds,
                "ratios": ratios,
            }
        out["metrics"][metric] = entry
    return out


def print_table(results: List[dict], backends) -> None:
    for metric, title in (("per-call", "per-call synchronized"), ("steady", "steady-state amortized")):
        header = f"{'scenario':<30}{'metric':<10}{'FIA(ms)':>10}"
        header += "".join(f"{b + '(ms)':>10}" for b in backends)
        header += "".join(f"{b + ' ratio':>12}" for b in backends)
        header += "".join(f"{b + ' disp':>16}" for b in backends)
        header += "".join(f"{b + ' verdict':>12}" for b in backends)
        print(f"\n[{title}] interleaved over {ROUNDS} rounds "
              f"(ratio = ours/fia per round, reported median)")
        print(header)
        print("-" * len(header))
        for row in results:
            e = row["metrics"][metric]
            line = f"{row['name']:<30}{metric:<10}{e['fia_median']:>10.4f}"
            for b in backends:
                line += f"{e['backends'][b]['median']:>10.4f}"
            for b in backends:
                line += f"{e['backends'][b]['ratio_median']:>12.4f}"
            for b in backends:
                bk = e["backends"][b]
                line += f"{'[' + format(bk['ratio_min'], '.3f') + ',' + format(bk['ratio_max'], '.3f') + ']':>16}"
            for b in backends:
                line += f"{'PASS' if e['backends'][b]['pass'] else 'FAIL':>12}"
            print(line)


def main() -> int:
    global ROUNDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("v3", "v4", "both"), default="both")
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="scenario name to run (repeatable); default: all",
    )
    parser.add_argument("--rounds", type=int, default=ROUNDS, help="interleaved rounds")
    parser.add_argument("--json", default=None, help="optional path to dump results as JSON")
    args = parser.parse_args()

    ROUNDS = args.rounds

    if torch_npu is None:
        print("torch_npu is not installed; skipping MLA flash-attn benchmark.")
        return 0
    if not torch.npu.is_available():
        print("No Ascend NPU available; skipping MLA flash-attn benchmark.")
        return 0

    torch_npu.npu.set_device(0)

    backends = BACKENDS if args.backend == "both" else (args.backend,)
    scenarios = (
        [SCENARIOS[name] for name in args.scenario]
        if args.scenario
        else [PREFILL_SCENARIO, DECODE_SCENARIO, DECODE_SCENARIO_S2_996]
    )

    results = []
    for scenario in scenarios:
        per_round = measure_interleaved(scenario, backends)
        row = summarize(scenario, per_round, backends)
        results.append(row)
        r = row["metrics"]
        print(
            f"[done] {scenario.name:<30} "
            f"per-call ratio v3={r['per-call']['backends'].get('v3', {}).get('ratio_median', float('nan')):.3f} "
            f"v4={r['per-call']['backends'].get('v4', {}).get('ratio_median', float('nan')):.3f} | "
            f"steady ratio v3={r['steady']['backends'].get('v3', {}).get('ratio_median', float('nan')):.3f} "
            f"v4={r['steady']['backends'].get('v4', {}).get('ratio_median', float('nan')):.3f}"
        )

    print_table(results, backends)

    print(
        f"\n判定阈值: ratio = ours_median/fia_median, PASS when ratio <= "
        f"{1.0 + PASS_THRESHOLD_PCT / 100.0:.2f} (degradation <= {PASS_THRESHOLD_PCT:.0f}%)."
    )
    print(
        "recommendation: requirement #5 is judged primarily on the **steady** ratio --\n"
        "  it removes the per-call synchronize round-trip and the fixed launch overhead,\n"
        "  so it is closest to the operator's own cost (per-call is kept for the user's\n"
        "  end-to-end view). For truly tiny shapes both口径 are host-dispatch bound; see\n"
        "  probe_mla_v4_overhead.py for the device-kernel breakdown."
    )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": torch.npu.get_device_name(0),
                    "dtype": "bfloat16",
                    "warmup": WARMUP,
                    "iters": ITERS,
                    "rounds": ROUNDS,
                    "steady_batches": STEADY_BATCHES,
                    "steady_iters": STEADY_ITERS,
                    "threshold_pct": PASS_THRESHOLD_PCT,
                    "backends": list(backends),
                    "results": results,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
