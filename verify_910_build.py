# Simple smoke test for the Ascend 910 build.
#
# Run from the repository root so the in-place built .so modules are importable:
#     source <cann>/set_env.sh
#     python verify_910_build.py
#
# For each FlashAttention generation it:
#   1. imports the compiled extension (.so) to prove the build actually loaded;
#   2. runs flash_attn_func on the NPU and compares the result with a plain
#      PyTorch CPU reference.

import importlib
import sys

import torch
import torch_npu


# (import name, module that must load, label)
BACKENDS = [
    ("flash_attn_npu", "flash_attn_npu.flash_attn_npu", "v2 (910)"),
    ("flash_attn_npu_3", "flash_attn_npu_3.flash_attn_npu_3", "v3 (910)"),
    ("flash_attn_npu_4", "flash_attn_npu_4.flash_attn_npu_4", "v4 (910)"),
]

BATCH, SEQ, HEADS, DIM = 2, 128, 4, 64
DTYPE = torch.float16
ATOL, RTOL = 1e-2, 1e-2


def ref_attention(q, k, v, causal, scale):
    """Plain PyTorch attention on CPU: softmax(q k^T * scale) @ v.

    q/k/v use the BSND layout (batch, seqlen, nheads, headdim)."""
    g = q.shape[2] // k.shape[2]
    if g > 1:
        k = k.repeat_interleave(g, dim=2)
        v = v.repeat_interleave(g, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        mask = torch.tril(torch.ones(sq, sk, dtype=torch.bool), diagonal=sk - sq)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


def run_case(mod, causal):
    scale = 1.0 / (DIM**0.5)
    torch.manual_seed(0)
    q = torch.randn(BATCH, SEQ, HEADS, DIM, dtype=DTYPE)
    k = torch.randn(BATCH, SEQ, HEADS, DIM, dtype=DTYPE)
    v = torch.randn(BATCH, SEQ, HEADS, DIM, dtype=DTYPE)

    golden = ref_attention(q.float(), k.float(), v.float(), causal, scale).to(DTYPE)

    qn, kn, vn = q.npu(), k.npu(), v.npu()
    out = mod.flash_attn_func(qn, kn, vn, causal=causal)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = out.cpu()

    torch.testing.assert_close(out, golden, atol=ATOL, rtol=RTOL)
    return out.shape


def main():
    device = torch_npu.npu.get_device_name()
    print(f"Device: {device}")
    if "Ascend910" not in device:
        print("ERROR: this script targets Ascend 910.")
        return 1

    failed = 0
    for pkg, ext_name, label in BACKENDS:
        try:
            mod = importlib.import_module(pkg)
            ext = importlib.import_module(ext_name)
            print(f"\n[{label}] extension: {ext.__file__}")
            for causal in (False, True):
                shape = run_case(mod, causal)
                print(f"[{label}] flash_attn_func causal={causal} -> {tuple(shape)} OK")
        except Exception as error:  # noqa: BLE001 - surface any failure
            failed += 1
            print(f"[{label}] FAILED: {type(error).__name__}: {error}")

    print("\n=== RESULT ===")
    if failed:
        print(f"{failed} backend(s) FAILED")
        return 1
    print("All 910 backends verified (v2, v3, v4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
