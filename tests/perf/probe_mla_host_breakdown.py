"""Task-4.3 probe: break the MLA host/interface cost into layers.

Peels one public MLA call into layers so the v3-vs-v4 gap can be attributed per
function instead of per question mark:

  L0 whole public call      -> e.g. fa4.flash_attn_varlen_func(...)
  L1 autograd node + body   -> FlashAttnVarlenFunc.apply(...)      (v3/v4 varlen)
  L2 custom-op + prep       -> _flash_attn_forward(...)            (v3 plain fn / v4 torch.library op)
  L3 pybind + C++ host      -> flash_attn_npu_{3,4}.fwd(...)       (native enqueue)

Layer arguments are captured from one real call (monkeypatch) so they always
match. Layers are timed **round-robin interleaved** across R rounds and the
median per layer is reported, so NPU clock/thermal drift is common-mode.

Timing is the unsynchronized "queue primed" host wall (us/call), which is the
metric that matters for these host-dispatch-bound shapes.

Read-only; no csrc / kernel changes.

Usage:
    python tests/perf/probe_mla_host_breakdown.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

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

WARMUP = 20
ROUNDS = 9
ITERS = 60


def _one(fn, iters):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def interleaved(layers, rounds=ROUNDS, iters=ITERS):
    """layers: {name: fn}; returns {name: median us/call}, round-robin timed."""
    for fn in layers.values():  # warm every layer once
        for _ in range(WARMUP):
            fn()
    torch.npu.synchronize()
    samples = {name: [] for name in layers}
    for _ in range(rounds):
        for name, fn in layers.items():
            samples[name].append(_one(fn, iters))
    return {name: statistics.median(v) for name, v in samples.items()}


def measure(public_fn, iface, label):
    pkg = getattr(iface, iface.__name__.split(".")[0])  # compiled pybind module
    orig_custom = iface._flash_attn_forward
    orig_apply = iface.FlashAttnVarlenFunc.apply
    orig_fwd = pkg.fwd
    cap = {}

    def wrap_apply(*a, **kw):
        cap["apply"] = (a, kw)
        return orig_apply(*a, **kw)

    def wrap_custom(*a, **kw):
        cap["custom"] = (a, kw)
        return orig_custom(*a, **kw)

    def wrap_fwd(*a, **kw):
        cap["fwd"] = (a, kw)
        return orig_fwd(*a, **kw)

    iface.FlashAttnVarlenFunc.apply = wrap_apply
    iface._flash_attn_forward = wrap_custom
    pkg.fwd = wrap_fwd
    try:
        public_fn()
    finally:
        iface.FlashAttnVarlenFunc.apply = orig_apply
        iface._flash_attn_forward = orig_custom
        pkg.fwd = orig_fwd

    layers = {
        "public": public_fn,
        "custom": lambda: orig_custom(*cap["custom"][0], **cap["custom"][1]),
        "pybind": lambda: orig_fwd(*cap["fwd"][0], **cap["fwd"][1]),
    }
    if "apply" in cap:
        layers["apply"] = lambda: orig_apply(*cap["apply"][0], **cap["apply"][1])
    t = interleaved(layers)

    total = t["public"]
    custom = t["custom"]
    pybind = t["pybind"]
    apply = t.get("apply", custom)
    return {
        "label": label,
        "total_us": total,
        "w_wrapper": total - apply,
        "w_autograd_body": apply - custom,
        "w_custom": custom - pybind,
        "w_pybind": pybind,
    }


def main() -> int:
    if torch_npu is None or not torch.npu.is_available():
        print("torch_npu / NPU unavailable")
        return 0
    torch_npu.npu.set_device(0)

    import flash_attn_npu_3 as fa3
    import flash_attn_npu_4 as fa4
    import flash_attn_npu_3.flash_attn_npu_interface as iface3
    import flash_attn_npu_4.flash_attn_npu_interface as iface4

    rows = []
    for scenario in (PREFILL_SCENARIO, DECODE_SCENARIO):
        t = build_tensors(scenario, device="npu")
        tag = scenario.form
        if scenario.form == "prefill":
            v3_fn = lambda: fa3.flash_attn_varlen_func(  # noqa: E731
                t["q"], t["k"], t["v"], t["cu_seqlens_q"], t["cu_seqlens_k"],
                max_seqlen_q=scenario.total_q, max_seqlen_k=scenario.total_kv,
                softmax_scale=scenario.softmax_scale, causal=True)
            v4_fn = lambda: fa4.flash_attn_varlen_func(  # noqa: E731
                t["q"], t["k"], t["v"], cu_seqlens_q=t["cu_seqlens_q"], cu_seqlens_k=t["cu_seqlens_k"],
                max_seqlen_q=scenario.total_q, max_seqlen_k=scenario.total_kv,
                softmax_scale=scenario.softmax_scale, causal=True)
        else:
            meta = fa3.get_scheduler_metadata(
                1, 1, scenario.total_kv, scenario.num_heads_q, scenario.num_heads_kv,
                t["q"].shape[-1], t["cache_seqlens"], qkv_dtype=scenario.dtype,
                headdim_v=t["v"].shape[-1], causal=False, num_splits=0)
            v3_fn = lambda: fa3.flash_attn_with_kvcache(  # noqa: E731
                t["q"], t["k"], t["v"], qv=t["qv"], cache_seqlens=t["cache_seqlens"],
                softmax_scale=scenario.softmax_scale, causal=False,
                scheduler_metadata=meta, num_splits=0)
            v4_fn = lambda: fa4.flash_attn_varlen_func(  # noqa: E731
                t["q"], t["k"], t["v"], qv=t["qv"], seqused_k=t["cache_seqlens"],
                softmax_scale=scenario.softmax_scale, causal=False)

        rows.append(measure(v3_fn, iface3, f"v3_{tag}"))
        rows.append(measure(v4_fn, iface4, f"v4_{tag}"))

    print(f"\ninterleaved host wall per layer (us/call, queue-primed loop, "
          f"{ROUNDS} rounds x {ITERS} iters, median)")
    hdr = (f"{'variant':<14}{'total':>9}{'wrapper':>9}{'autograd+body':>15}"
           f"{'custom_op':>11}{'pybind+C++':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<14}{r['total_us']:>9.1f}{r['w_wrapper']:>9.1f}"
              f"{r['w_autograd_body']:>15.1f}{r['w_custom']:>11.1f}{r['w_pybind']:>12.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
