# Copyright (c) 2026, Minghua Shen.
"""Wall-clock BNSD forward comparison: custom 950 v3 op vs the CANN FIA baseline.

One fixed case (BNSD ``(B, H, S, D) = (1, 64, 4096, 128)``, fp16, MHA, causal) is
run through both operators; a correctness sanity check runs first so timing is
only reported for a semantically equivalent configuration.

Usage:
    python docs/tests/perf_bnsd_vs_cann/bench_wallclock.py --device 4

``--device`` is the absolute card id: the script pins
``ASCEND_RT_VISIBLE_DEVICES`` to it before torch_npu is imported (logical device
0 afterwards).  Pinning is required, not cosmetic: with all eight cards visible
the causal kernel variant faults with 507033 / "Failed to send the OPP to the
device".  If ``ASCEND_RT_VISIBLE_DEVICES`` is already set in the environment it
wins, and ``--device`` is ignored (ids are then already remapped).
"""

import argparse
import os
import sys
import time

_PRE_PARSER = argparse.ArgumentParser(add_help=False)
_PRE_PARSER.add_argument("--device", type=int, default=4)
_pinned = _PRE_PARSER.parse_known_args()[0]
DEVICE_SOURCE = "env" if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") else "--device"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(_pinned.device))
LOGICAL_DEVICE = 0

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

from flash_attn_npu_3 import flash_attn_func  # noqa: E402
from model import B, CAUSAL, D, DTYPE, H, KV_H, S, SCALE, build_causal_mask, cann_forward  # noqa: E402

BATCH, HEADS, KV_HEADS, SEQLEN, HEADDIM = B, H, KV_H, S, D
TOL = {"rtol": 2e-2, "atol": 2e-2}


def _rand_bnsd(heads, device, generator):
    return torch.randn(BATCH, heads, SEQLEN, HEADDIM, dtype=DTYPE, generator=generator).to(device)


def _build_inputs(device):
    generator = torch.Generator().manual_seed(2026)
    return (
        _rand_bnsd(HEADS, device, generator),
        _rand_bnsd(KV_HEADS, device, generator),
        _rand_bnsd(KV_HEADS, device, generator),
    )


def _run_custom(query, key, value):
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    return flash_attn_func(q, k, v, softmax_scale=SCALE, causal=CAUSAL)


def _run_cann(query, key, value, atten_mask):
    return cann_forward(query, key, value, atten_mask)


def _time_us(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    mid = len(samples) // 2
    median = samples[mid] if len(samples) % 2 else 0.5 * (samples[mid - 1] + samples[mid])
    return sum(samples) / len(samples) * 1e6, median * 1e6, samples[0] * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=4, help="absolute NPU card id")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05, help="max allowed degradation")
    args = parser.parse_args()

    torch.npu.set_device(LOGICAL_DEVICE)
    print(
        f"card={os.environ['ASCEND_RT_VISIBLE_DEVICES']} (source={DEVICE_SOURCE}) "
        f"logical={LOGICAL_DEVICE} iters={args.iters} warmup={args.warmup} "
        f"shape=BNSD({BATCH},{HEADS},{SEQLEN},{HEADDIM}) {DTYPE} causal={CAUSAL}",
        flush=True,
    )

    query, key, value = _build_inputs(torch.device("npu"))
    atten_mask = build_causal_mask(query.device)
    ours = _run_custom(query, key, value)
    cann = _run_cann(query, key, value, atten_mask)
    # Ours is contiguous BSND (B, S, H, D); the baseline returns BNSD (B, H, S, D).
    diff = (cann.transpose(1, 2) - ours).abs().max().item()
    if not torch.allclose(cann.transpose(1, 2), ours, **TOL):
        sys.exit(f"correctness sanity FAILED: max abs diff {diff} (fix the causal config first)")
    print(f"correctness sanity OK: max abs diff {diff:.4f}", flush=True)

    custom_us = _time_us(lambda: _run_custom(query, key, value), args.iters, args.warmup)
    baseline_us = _time_us(
        lambda: _run_cann(query, key, value, atten_mask), args.iters, args.warmup
    )
    custom_med, baseline_med = custom_us[1], baseline_us[1]

    flops = 2.0 * (2 if CAUSAL else 1) * BATCH * HEADS * SEQLEN * SEQLEN * HEADDIM
    traffic_gb = 4 * BATCH * HEADS * SEQLEN * HEADDIM * 2 / 1024**3

    print("")
    print("| metric | custom (950 v3) | CANN FIA | degradation |")
    print("|---|---|---|---|")
    for label, custom, baseline in (
        ("mean us", custom_us[0], baseline_us[0]),
        ("median us", custom_med, baseline_med),
        ("min us", custom_us[2], baseline_us[2]),
    ):
        print(
            f"| {label} | {custom:.1f} | {baseline:.1f} | {(custom / baseline - 1) * 100:+.1f}% |"
        )

    median_degradation = custom_med / baseline_med - 1
    print("")
    print(
        f"median degradation (custom/baseline - 1) = {median_degradation * 100:+.2f}%"
        f"  (threshold {args.threshold * 100:.0f}%)"
    )
    print(
        f"custom: {flops / (custom_med * 1e-6) / 1e12:.1f} TFLOP/s,"
        f" {traffic_gb / (custom_med * 1e-6):.0f} GB/s"
        f" | CANN FIA: {flops / (baseline_med * 1e-6) / 1e12:.1f} TFLOP/s,"
        f" {traffic_gb / (baseline_med * 1e-6):.0f} GB/s"
    )

    if median_degradation > args.threshold:
        sys.exit(
            f"FAIL: degradation {median_degradation * 100:+.2f}% > {args.threshold * 100:.0f}%"
        )
    print(f"PASS: degradation {median_degradation * 100:+.2f}% within {args.threshold * 100:.0f}%")


if __name__ == "__main__":
    main()
