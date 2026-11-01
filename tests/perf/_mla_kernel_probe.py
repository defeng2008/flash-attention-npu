"""Single-variant runner used by msprof to compare v3/v4 MLA kernel work.

Usage (called under msprof):  python _mla_kernel_probe.py <variant>
variants: fia_prefill fia_decode v3_prefill v4_prefill v3_decode_meta v3_decode_nometa v4_decode
"""

from __future__ import annotations

import os
import sys

import torch

try:
    import torch_npu
except ImportError:  # pragma: no cover
    torch_npu = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mla_user_scenarios import DECODE_SCENARIO, PREFILL_SCENARIO, make_inputs  # noqa: E402
from bench_mla_flash_attn import build_tensors  # noqa: E402

WARMUP = 20
ITERS = int(os.environ.get("MLAPROBE_ITERS", "50"))


def build_fn(variant):
    scenario = PREFILL_SCENARIO if "prefill" in variant else DECODE_SCENARIO
    t = build_tensors(scenario, device="npu")
    ins = make_inputs(scenario, seed=0, device="npu")

    if variant.startswith("fia"):
        import bench_mla_fia as fia
        call = fia._prefill_call if scenario.form == "prefill" else fia._decode_call
        return scenario, (lambda: call(ins, scenario))
    if variant.startswith("v3"):
        import flash_attn_npu_3 as fa3
        if scenario.form == "prefill":
            return scenario, (lambda: fa3.flash_attn_varlen_func(
                t["q"], t["k"], t["v"], t["cu_seqlens_q"], t["cu_seqlens_k"],
                max_seqlen_q=scenario.total_q, max_seqlen_k=scenario.total_kv,
                softmax_scale=scenario.softmax_scale, causal=True))
        meta = fa3.get_scheduler_metadata(
            1, 1, scenario.total_kv, scenario.num_heads_q, scenario.num_heads_kv,
            t["q"].shape[-1], t["cache_seqlens"], qkv_dtype=scenario.dtype,
            headdim_v=t["v"].shape[-1], causal=False, num_splits=0)
        kw = {} if "nometa" in variant else {"scheduler_metadata": meta}
        return scenario, (lambda: fa3.flash_attn_with_kvcache(
            t["q"], t["k"], t["v"], qv=t["qv"], cache_seqlens=t["cache_seqlens"],
            softmax_scale=scenario.softmax_scale, causal=False, num_splits=0, **kw))
    import flash_attn_npu_4 as fa4
    if scenario.form == "prefill":
        return scenario, (lambda: fa4.flash_attn_varlen_func(
            t["q"], t["k"], t["v"], cu_seqlens_q=t["cu_seqlens_q"], cu_seqlens_k=t["cu_seqlens_k"],
            max_seqlen_q=scenario.total_q, max_seqlen_k=scenario.total_kv,
            softmax_scale=scenario.softmax_scale, causal=True))
    return scenario, (lambda: fa4.flash_attn_varlen_func(
        t["q"], t["k"], t["v"], qv=t["qv"], seqused_k=t["cache_seqlens"],
        softmax_scale=scenario.softmax_scale, causal=False))


def run_profile(variant, out_dir):
    """Profile the variant and aggregate NPU kernel durations by name."""
    from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler
    scenario, fn = build_fn(variant)
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    with profile(
        activities=[ProfilerActivity.NPU, ProfilerActivity.CPU],
        on_trace_ready=tensorboard_trace_handler(out_dir),
    ) as prof:
        for _ in range(ITERS):
            fn()
        torch.npu.synchronize()
    agg = {}
    for ev in prof.events():
        if ev.device_type is None or "npu" not in str(ev.device_type).lower():
            continue
        agg.setdefault(ev.name, [0, 0.0])
        agg[ev.name][0] += 1
        agg[ev.name][1] += ev.device_time_total
    print(f"=== {variant} kernel totals (npu) ===")
    for name, (cnt, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:<40} n={cnt:<5} total={tot:>12.1f} us  avg={tot/cnt:>9.2f} us")
    return 0


def main() -> int:
    torch_npu.npu.set_device(0)
    if sys.argv[1] == "--profile":
        return run_profile(sys.argv[2], sys.argv[3])
    variant = sys.argv[1]
    scenario = PREFILL_SCENARIO if "prefill" in variant else DECODE_SCENARIO
    t = build_tensors(scenario, device="npu")
    ins = make_inputs(scenario, seed=0, device="npu")

    if variant.startswith("fia"):
        import bench_mla_fia as fia
        call = fia._prefill_call if scenario.form == "prefill" else fia._decode_call
        fn = lambda: call(ins, scenario)  # noqa: E731
    elif variant.startswith("v3"):
        import flash_attn_npu_3 as fa3
        if scenario.form == "prefill":
            fn = lambda: fa3.flash_attn_varlen_func(  # noqa: E731
                t["q"], t["k"], t["v"], t["cu_seqlens_q"], t["cu_seqlens_k"],
                max_seqlen_q=scenario.total_q, max_seqlen_k=scenario.total_kv,
                softmax_scale=scenario.softmax_scale, causal=True)
        else:
            meta = fa3.get_scheduler_metadata(
                1, 1, scenario.total_kv, scenario.num_heads_q, scenario.num_heads_kv,
                t["q"].shape[-1], t["cache_seqlens"], qkv_dtype=scenario.dtype,
                headdim_v=t["v"].shape[-1], causal=False, num_splits=0)
            kw = {} if "nometa" in variant else {"scheduler_metadata": meta}
            fn = lambda: fa3.flash_attn_with_kvcache(  # noqa: E731
                t["q"], t["k"], t["v"], qv=t["qv"], cache_seqlens=t["cache_seqlens"],
                softmax_scale=scenario.softmax_scale, causal=False, num_splits=0, **kw)
    else:
        import flash_attn_npu_4 as fa4
        if scenario.form == "prefill":
            fn = lambda: fa4.flash_attn_varlen_func(  # noqa: E731
                t["q"], t["k"], t["v"], cu_seqlens_q=t["cu_seqlens_q"], cu_seqlens_k=t["cu_seqlens_k"],
                max_seqlen_q=scenario.total_q, max_seqlen_k=scenario.total_kv,
                softmax_scale=scenario.softmax_scale, causal=True)
        else:
            fn = lambda: fa4.flash_attn_varlen_func(  # noqa: E731
                t["q"], t["k"], t["v"], qv=t["qv"], seqused_k=t["cache_seqlens"],
                softmax_scale=scenario.softmax_scale, causal=False)

    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    print(f"{variant}: ran {ITERS} iters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
