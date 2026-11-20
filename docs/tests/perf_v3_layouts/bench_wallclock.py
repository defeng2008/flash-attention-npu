# Copyright (c) 2026, Minghua Shen.
"""Wall-clock BSND-vs-BNSD layout comparison for the 950 v3 interfaces.

Times ``flash_attn_func`` and ``flash_attn_with_kvcache`` twice each: once with the inputs
passed as contiguous BSND, once with the very same logical numbers held in contiguous BNSD
storage and handed over as a ``transpose(1, 2)`` view (which is what makes the operator take
its BNSD path).  Both slots describe the same attention problem -- see ``model_new_ascendc.py``
for why BNSD storage is materialised once instead of re-reading the shared BSND memory -- so
the only difference is the layout the operator reads.  A sanity check runs first so timing is
only reported when both entry points produce the same numbers.

Usage:
    python docs/tests/perf_v3_layouts/bench_wallclock.py --device 5

``--device`` is the absolute card id: the script pins ``ASCEND_RT_VISIBLE_DEVICES`` to it
before torch_npu is imported (logical device 0 afterwards).  Pinning is required, not
cosmetic -- with all eight cards visible the causal kernel variant faults with 507033 /
"Failed to send the OPP to the device".  If ``ASCEND_RT_VISIBLE_DEVICES`` is already set in
the environment it wins and ``--device`` is ignored (ids are then already remapped).
"""

import argparse
import importlib.util
import os
import sys
import time

_PRE_PARSER = argparse.ArgumentParser(add_help=False)
_PRE_PARSER.add_argument("--device", type=int, default=5)
_pinned = _PRE_PARSER.parse_known_args()[0]
DEVICE_SOURCE = "env" if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") else "--device"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(_pinned.device))
LOGICAL_DEVICE = 0

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = {"rtol": 2e-2, "atol": 2e-2}
INTERFACES = ("func", "kvcache")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def _load_case(iface):
    """Load both layout slots of one interface plus the shared device-resident inputs."""
    bsnd = _load(os.path.join(HERE, iface, "model.py"), f"{iface}_bsnd")
    bnsd = _load(os.path.join(HERE, iface, "model_new_ascendc.py"), f"{iface}_bnsd")
    device = torch.device("npu")
    inputs = [[t.to(device) for t in group] for group in bsnd.get_input_groups()][0]
    return bsnd, bnsd, inputs


def _bench(iface, args):
    bsnd, bnsd, inputs = _load_case(iface)
    q, k = inputs[0], inputs[1]
    bsnd_model = bsnd.Model().to(q.device).eval()
    bnsd_model = bnsd.ModelNew().to(q.device).eval()

    out_bsnd = bsnd_model(*inputs)
    out_bnsd = bnsd_model(*inputs)
    # The 950 interface answers in the caller's query shape, and the BNSD slot passes a
    # (B, S, H, D) view over BNSD storage, so the result already arrives as contiguous BSND.
    if out_bsnd.shape != out_bnsd.shape:
        sys.exit(
            f"{iface}: output shape mismatch: {tuple(out_bsnd.shape)} vs {tuple(out_bnsd.shape)}"
        )
    diff = (out_bsnd - out_bnsd).abs().max().item()
    if not torch.allclose(out_bsnd, out_bnsd, **TOL):
        sys.exit(f"{iface}: sanity FAILED, max abs diff {diff} (BSND and BNSD paths disagree)")
    print(f"{iface}: sanity OK (max abs diff {diff:.5f})", flush=True)

    bsnd_us = _time_us(lambda: bsnd_model(*inputs), args.iters, args.warmup)
    bnsd_us = _time_us(lambda: bnsd_model(*inputs), args.iters, args.warmup)

    # Causal runs do half the score-matrix work of the full one; the cache case has S_Q = 1.
    flops = 2.0 * (2 if bsnd.CAUSAL else 1) * q.shape[0] * q.shape[2] * q.shape[1] * k.shape[1]
    flops *= q.shape[3]
    traffic_gb = (q.numel() + k.numel() + inputs[2].numel() + out_bsnd.numel()) * 2 / 1024**3

    print(f"\n### {iface}: flash_attn_{iface}  (S_Q = {q.shape[1]}, S_KV = {k.shape[1]})\n")
    print("| metric | BSND us | BNSD us | BNSD/BSND - 1 |")
    print("|---|---|---|---|")
    for label, b, n in zip(("mean", "median", "min"), bsnd_us, bnsd_us):
        print(f"| {label} | {b:.1f} | {n:.1f} | {(n / b - 1) * 100:+.2f}% |")
    bsnd_med, bnsd_med = bsnd_us[1], bnsd_us[1]
    print(
        f"\nmedian: BSND {bsnd_med:.1f} us, BNSD {bnsd_med:.1f} us, "
        f"BSND/BNSD speedup {bsnd_med / bnsd_med:.3f}x, "
        f"BNSD/BSND - 1 = {(bnsd_med / bsnd_med - 1) * 100:+.2f}%"
    )
    print(
        f"BSND: {flops / (bsnd_med * 1e-6) / 1e12:.1f} TFLOP/s, "
        f"{traffic_gb / (bsnd_med * 1e-6):.0f} GB/s"
        f" | BNSD: {flops / (bnsd_med * 1e-6) / 1e12:.1f} TFLOP/s, "
        f"{traffic_gb / (bnsd_med * 1e-6):.0f} GB/s"
    )
    return bsnd_med, bnsd_med


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=5, help="absolute NPU card id")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    torch.npu.set_device(LOGICAL_DEVICE)
    print(
        f"card={os.environ['ASCEND_RT_VISIBLE_DEVICES']} (source={DEVICE_SOURCE}) "
        f"logical={LOGICAL_DEVICE} iters={args.iters} warmup={args.warmup}",
        flush=True,
    )

    results = {iface: _bench(iface, args) for iface in INTERFACES}

    print("\n## summary (median)\n")
    print("| interface | BSND us | BNSD us | BSND/BNSD | BNSD/BSND - 1 |")
    print("|---|---|---|---|---|")
    for iface, (bsnd_med, bnsd_med) in results.items():
        print(
            f"| flash_attn_{iface} | {bsnd_med:.1f} | {bnsd_med:.1f} | "
            f"{bsnd_med / bnsd_med:.3f}x | {(bnsd_med / bsnd_med - 1) * 100:+.2f}% |"
        )


if __name__ == "__main__":
    main()
