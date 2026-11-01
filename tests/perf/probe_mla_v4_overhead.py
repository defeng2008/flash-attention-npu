"""Task-4 rework probe: attribute the v4-vs-v3 MLA performance gap.

Runs everything in ONE process (no cross-session clock drift) and isolates:

  * device steady time  -> torch.npu.Event around N back-to-back calls (device busy);
  * AICPU metadata cost -> v3 decode WITH a precomputed scheduler_metadata (reuse,
    skips AICPU) vs v3 decode WITHOUT it (regenerates on the AICPU, same kernel);
  * host dispatch cost  -> wall time of N calls with the device queue primed vs the
    device-busy time (a null tiny-op loop gives the pybind baseline).

Only read-only instrumentation; no csrc / kernel changes.

Usage:
    python tests/perf/probe_mla_v4_overhead.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from typing import Callable, List

import torch

try:
    import torch_npu
except ImportError:  # pragma: no cover
    torch_npu = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mla_user_scenarios import (  # noqa: E402
    DECODE_SCENARIO,
    PREFILL_SCENARIO,
    make_inputs,
)
from bench_mla_flash_attn import build_tensors  # noqa: E402

WARMUP = 10
DEVICE_ITERS = 200
HOST_ITERS = 200


def device_steady(fn: Callable[[], torch.Tensor], iters: int = DEVICE_ITERS) -> float:
    """Device busy ms/call: Event span over ``iters`` primed back-to-back calls."""
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def host_wall(fn: Callable[[], torch.Tensor], iters: int = HOST_ITERS) -> float:
    """Wall ms/call of an unsynchronized loop (device queue primed)."""
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def host_dispatch_only(fn: Callable[[], torch.Tensor], iters: int = HOST_ITERS) -> float:
    """Best-effort CPU-side cost: enqueue a large batch, then measure the second
    half of the loop whose launch queue is already saturated (host-bound enqueue)."""
    for _ in range(WARMUP):
        fn()
    # Feed a deep backlog so the device is the limiter, then time only enqueue.
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters * 4):
        fn()
    enqueue = (time.perf_counter() - t0) * 1e3
    torch.npu.synchronize()
    return enqueue / (iters * 4)


def null_op_baseline(iters: int = 5000) -> float:
    """Pure pybind/dispatch floor: repeated tiny op with no sync in the loop."""
    x = torch.ones(8, dtype=torch.float32, device="npu")
    y = torch.ones(8, dtype=torch.float32, device="npu")
    fn = lambda: torch.add(x, y)  # noqa: E731
    for _ in range(100):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def main() -> int:
    if torch_npu is None or not torch.npu.is_available():
        print("torch_npu / NPU unavailable")
        return 0
    torch_npu.npu.set_device(0)

    import bench_mla_fia as fia

    rows: List[tuple] = []

    def add(name: str, fn: Callable[[], torch.Tensor]) -> None:
        dev = device_steady(fn)
        wall = host_wall(fn)
        rows.append((name, dev, wall))
        print(f"[probe] {name:<34} device={dev:.4f} ms  wall={wall:.4f} ms")

    for scenario in (PREFILL_SCENARIO, DECODE_SCENARIO):
        t = build_tensors(scenario, device="npu")
        ins = make_inputs(scenario, seed=0, device="npu")
        fia_call = fia._prefill_call if scenario.form == "prefill" else fia._decode_call
        tag = scenario.form

        add(f"fia_{tag}", lambda: fia_call(ins, scenario))

        import flash_attn_npu_3 as fa3
        import flash_attn_npu_4 as fa4

        if scenario.form == "prefill":
            add(f"v3_{tag}", lambda: fa3.flash_attn_varlen_func(
                t["q"], t["k"], t["v"], t["cu_seqlens_q"], t["cu_seqlens_k"],
                max_seqlen_q=scenario.total_q, max_seqlen_k=scenario.total_kv,
                softmax_scale=scenario.softmax_scale, causal=True))
            add(f"v4_{tag}", lambda: fa4.flash_attn_varlen_func(
                t["q"], t["k"], t["v"], cu_seqlens_q=t["cu_seqlens_q"], cu_seqlens_k=t["cu_seqlens_k"],
                max_seqlen_q=scenario.total_q, max_seqlen_k=scenario.total_kv,
                softmax_scale=scenario.softmax_scale, causal=True))
        else:
            meta = fa3.get_scheduler_metadata(
                1, 1, scenario.total_kv, scenario.num_heads_q, scenario.num_heads_kv,
                t["q"].shape[-1], t["cache_seqlens"], qkv_dtype=scenario.dtype,
                headdim_v=t["v"].shape[-1], causal=False, num_splits=0)
            add(f"v3_{tag}_meta", lambda: fa3.flash_attn_with_kvcache(
                t["q"], t["k"], t["v"], qv=t["qv"], cache_seqlens=t["cache_seqlens"],
                softmax_scale=scenario.softmax_scale, causal=False,
                scheduler_metadata=meta, num_splits=0))
            add(f"v3_{tag}_nometa", lambda: fa3.flash_attn_with_kvcache(
                t["q"], t["k"], t["v"], qv=t["qv"], cache_seqlens=t["cache_seqlens"],
                softmax_scale=scenario.softmax_scale, causal=False, num_splits=0))
            add(f"v4_{tag}", lambda: fa4.flash_attn_varlen_func(
                t["q"], t["k"], t["v"], qv=t["qv"], seqused_k=t["cache_seqlens"],
                softmax_scale=scenario.softmax_scale, causal=False))

    null = null_op_baseline()
    print(f"\n[probe] null tiny-op (torch.add) dispatch floor = {null*1000:.2f} us/call")

    print("\n" + f"{'variant':<34}{'device(ms)':>12}{'wall(ms)':>12}")
    print("-" * 58)
    for name, dev, wall in rows:
        print(f"{name:<34}{dev:>12.4f}{wall:>12.4f}")

    # --- attribution for decode ---
    d = {name: dev for name, dev, _ in rows}
    if "v3_decode_meta" in d and "v3_decode_nometa" in d:
        aicpu = d["v3_decode_nometa"] - d["v3_decode_meta"]
        print(f"\n[AICPU metadata] v3_decode nometa - meta = {aicpu*1000:.1f} us/call")
        gap34 = d["v4_decode"] - d["v3_decode_meta"]
        gap_after_aicpu = d["v4_decode"] - d["v3_decode_nometa"]
        print(f"[decode gap] v4 - v3(meta)   = {gap34*1000:.1f} us/call")
        print(f"[decode gap] v4 - v3(nometa) = {gap_after_aicpu*1000:.1f} us/call "
              f"(residual after removing the AICPU reuse advantage)")
    if "v3_prefill" in d and "v4_prefill" in d:
        print(f"[prefill gap] v4 - v3 = {(d['v4_prefill']-d['v3_prefill'])*1000:.1f} us/call")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
