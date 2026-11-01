# Copyright (c) 2026, Minghua Shen.
# Manual MLA smoke check: run by hand, no pytest required.
#
#   source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh
#   conda activate cann
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   cd /home/w00811046/code/mla/flash-attention-npu
#   python tests/manual_mla_check.py                 # bf16, seq=64, heads=8
#   python tests/manual_mla_check.py --causal        # add the causal mask
#   python tests/manual_mla_check.py --dtype fp16 --seq 198 --heads 32
#
# Two shapes are exercised through the same public entry point:
#   prefill (unabsorbed) : q/k 192, v 128, no qv  -> out 128
#   decode  (absorbed)   : q/k 64, qv/v 512, 1 KV head -> out 512
# The absorbed form folds into the 576-wide layout
# [qv | q] @ [v | k]^T, so its first 128 output channels must match the
# prefill result when both are fed the same latent.

import argparse
import sys

import torch
import torch_npu  # noqa: F401

import flash_attn_npu_3 as fa3


def rel_l2(actual, expected):
    a = actual.double()
    b = expected.double()
    denom = torch.linalg.vector_norm(b).item()
    return (torch.linalg.vector_norm(a - b) / max(denom, 1e-12)).item()


def golden_attention(q, k, v, scale, causal):
    """Plain FP64 CPU reference with GQA broadcast."""
    q = q.double().cpu().transpose(1, 2)
    groups = q.shape[1] // k.shape[2]
    k = k.double().cpu().repeat_interleave(groups, dim=2).transpose(1, 2)
    v = v.double().cpu().repeat_interleave(groups, dim=2).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) * scale
    if causal:
        sq, sk = scores.shape[-2:]
        mask = torch.ones(sq, sk, dtype=torch.bool).triu(diagonal=sk - sq + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    return (scores.softmax(-1) @ v).transpose(1, 2)


def randn(shape, dtype, gen):
    return torch.randn(*shape, generator=gen).mul_(0.1).to(dtype).npu()


def check_prefill(dtype, batch, seq, heads, kv_heads, scale, causal, gen):
    q = randn((batch, seq, heads, 192), dtype, gen)
    k = randn((batch, seq, kv_heads, 192), dtype, gen)
    v = randn((batch, seq, kv_heads, 128), dtype, gen)

    out = fa3.flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
    assert out.shape == (batch, seq, heads, 128), out.shape

    ref = golden_attention(q, k, v, scale, causal)
    return "prefill", out, ref


def check_decode(dtype, batch, seq, heads, scale, causal, gen):
    q = randn((batch, seq, heads, 64), dtype, gen)          # rope part
    qv = randn((batch, seq, heads, 512), dtype, gen)        # absorbed part
    k = randn((batch, seq, 1, 64), dtype, gen)              # rope part
    v = randn((batch, seq, 1, 512), dtype, gen)             # latent c_KV

    out = fa3.flash_attn_func(q, k, v, softmax_scale=scale, causal=causal, qv=qv)
    assert out.shape == (batch, seq, heads, 512), out.shape

    # Absorbed form: 576-dot scores over [qv | q] and [v | k], output 512-d.
    q576 = torch.cat([qv, q], dim=-1)
    k576 = torch.cat([v, k], dim=-1)
    ref = golden_attention(q576, k576, v, scale, causal)
    return "decode ", out, ref


def check_metadata(dtype, batch, seq, heads, causal):
    """Precompute the schedule on the AICPU.

    headdim (64) != headdim_v (512) is what routes get_scheduler_metadata into
    the MLA branch instead of the ordinary FA3 tiling path. The record is then
    bound to this exact shape/dtype/mask, so it is only reported here.
    """
    lengths = torch.full((batch,), seq, dtype=torch.int32).npu()
    meta = fa3.get_scheduler_metadata(
        batch, seq, seq, heads, 1, 64, lengths,
        qkv_dtype=dtype, headdim_v=512, causal=causal,
    )
    return "metadata", meta, None


def main():
    parser = argparse.ArgumentParser(description="Manual MLA forward check (Ascend 910)")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2, help="prefill only; decode always uses 1")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--scale", type=float, default=None, help="default: 1/sqrt(q.last) per form")
    parser.add_argument("--tol", type=float, default=5e-3, help="relative L2 budget")
    args = parser.parse_args()

    if not torch.npu.is_available():
        print("FAIL: no NPU visible (check ASCEND_RT_VISIBLE_DEVICES / set_env.sh)")
        return 1
    device = torch.npu.get_device_name(0)
    if not device.startswith(("Ascend910B", "Ascend910C")):
        print(f"FAIL: MLA needs Ascend 910B/C, got {device}")
        return 1

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    causal = args.causal
    print(f"device={device} dtype={args.dtype} batch={args.batch} seq={args.seq} "
          f"heads={args.heads} kv_heads={args.kv_heads} causal={causal}")

    gen = torch.Generator().manual_seed(0)
    prefill_scale = args.scale if args.scale is not None else 192 ** -0.5
    decode_scale = args.scale if args.scale is not None else 576 ** -0.5

    cases = [
        lambda: check_prefill(dtype, args.batch, args.seq, args.heads, args.kv_heads,
                              prefill_scale, causal, gen),
        lambda: check_decode(dtype, args.batch, args.seq, args.heads, decode_scale, causal, gen),
        lambda: check_metadata(dtype, args.batch, args.seq, args.heads, causal),
    ]

    failed = 0
    for make in cases:
        name, out, ref = make()
        if ref is None:
            print(f"  {name}: shape={tuple(out.shape)} dtype={out.dtype} PASS")
            continue
        err = rel_l2(out.cpu(), ref)
        ok = err < args.tol
        failed += 0 if ok else 1
        print(f"  {name}: out={tuple(out.shape)} rel-L2={err:.3e} "
              f"(tol {args.tol:.0e}) {'PASS' if ok else 'FAIL'}")

    print("ALL PASS" if failed == 0 else f"{failed} CASE(S) FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
