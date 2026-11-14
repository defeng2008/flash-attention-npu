# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Pure device-kernel comparison: FA3/FA4 MLA vs CANN FIA MLA on the user shapes.

Scope (kernel time only)
------------------------
Only the *attention-compute* device kernel is timed. Timings come from
``torch_npu.profiler`` ``op_summary`` task durations, and the schedule/scrub
kernels (``ComputeFAMetadata*``, ``Fill``, ...) are filtered out after being
printed, so Python wrapping, host launch cost and metadata generation are never
charged to any row. FA3 and FA4 use the exact same rule.

Interfaces (5 total; varlen uses TND, the rest use BSND)
-------------------------------------------------------
* FA3 (3): ``flash_attn_func`` (BSND), ``flash_attn_varlen_func`` (TND),
  ``flash_attn_with_kvcache`` (BSND)
* FA4 (2): ``flash_attn_func`` (BSND), ``flash_attn_varlen_func`` (TND)

Baseline
--------
Each scenario gets two FIA baselines, one per layout, so every row is compared
against a same-layout reference:
* ``fia_mla_bsnd`` -- BSND rows
* ``fia_mla_tnd``  -- TND rows
The baseline (``torch_npu.npu_fused_infer_attention_score``) must run on
CANN 9.2; the script aborts otherwise.

Scenarios (``mla_user_scenarios.py``):
* prefill: single 198-token sequence, 32 Q/KV heads, Dq=Dk=192, Dv=128, causal.
* decode: 1 query token, 32 Q heads, KV length 1025, absorbed form (q 64 rope +
  qv 512, latent KV 512).

Gates (kernel time only)
------------------------
* ``candidate / same_layout_fia <= 1.05`` for every MLA row.

Shared-device noise is about +-3%, so a single measurement near the gate is not
conclusive; re-run before acting on a marginal failure. The script exits
non-zero when any gate fails.

Usage::

    python docs/test/perf/bench_mla_vs_fia.py
"""

import csv
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu

PERF_DIR = Path(__file__).resolve().parent
PERF_OUTPUTS = PERF_DIR / "perf_outputs"
sys.path.insert(0, str(PERF_DIR.parents[2]))
sys.path.insert(0, str(PERF_DIR))

from mla_user_scenarios import DECODE, PREFILL, scenario_summary  # noqa: E402

from flash_attn_npu_3 import (  # noqa: E402
    flash_attn_func as fa3_func,
    flash_attn_varlen_func as fa3_varlen,
    flash_attn_with_kvcache as fa3_kvcache,
)
from flash_attn_npu_4 import (  # noqa: E402
    flash_attn_func as fa4_func,
    flash_attn_varlen_func as fa4_varlen,
)

DEVICE = "npu"
DTYPE = torch.bfloat16
WARMUP = 5
STEPS = 10
KERNEL_GATE = 1.05
# Compute-kernel allow-list: only these device kernels count towards the
# comparison; everything else (metadata/fill/...) is printed as excluded.
FA_COMPUTE_PATTERNS = ("FAInfer", "MlaDecodeGemv")
FIA_COMPUTE_PATTERNS = ("FusedInferAttentionScore",)
REQUIRED_CANN_PREFIX = "9.2"
SUMMARY_CSV = PERF_OUTPUTS / "bench_kernel_summary.csv"

# (label, FIA baseline impl, candidate impls). BSND rows compare against
# ``fia_mla_bsnd``, TND rows against ``fia_mla_tnd`` so each row has a
# same-layout reference.
GATED = (
    ("MLA-BSND", "fia_mla_bsnd", ("fa3_func_bsnd", "fa3_kvcache_bsnd", "fa4_func_bsnd")),
    ("MLA-TND", "fia_mla_tnd", ("fa3_varlen_tnd", "fa4_varlen_tnd")),
)


def _rand(*shape):
    return (torch.rand(shape, device="cpu") - 0.5).to(DTYPE).to(DEVICE)


def _causal_mask():
    return torch.triu(torch.ones(2048, 2048), diagonal=1).bool().npu()


def _cu_seqlens(lengths):
    """int32 cu_seqlens on the NPU (the metadata AICPU dereferences it in place)."""
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    return torch.tensor(cumulative, dtype=torch.int32, device=DEVICE)


def _prefill_impls():
    s = PREFILL
    rope = 64
    half = s.head_dim_q - rope
    q = _rand(1, s.total_q, s.heads_q, s.head_dim_q)
    k = _rand(1, s.total_kv, s.heads_kv, s.head_dim_k)
    v = _rand(1, s.total_kv, s.heads_kv, s.head_dim_v)
    mask = _causal_mask()
    # Materialise the rope-split operands once: slicing a 192-wide tensor leaves
    # a strided view, and doing the copy per iteration would charge the copy to
    # the FIA row instead of the kernel.
    q_nope, q_rope = q[..., :half].contiguous(), q[..., half:].contiguous()
    k_nope, k_rope = k[..., :half].contiguous(), k[..., half:].contiguous()
    # TND: batch 1, so the BSND tensors flatten to (total, heads, dim) views.
    q_t, k_t, v_t = q.reshape(s.total_q, s.heads_q, -1), \
        k.reshape(s.total_kv, s.heads_kv, -1), v.reshape(s.total_kv, s.heads_kv, -1)
    q_nope_t, q_rope_t = q_nope.reshape(s.total_q, s.heads_q, half), \
        q_rope.reshape(s.total_q, s.heads_q, rope)
    k_nope_t, k_rope_t = k_nope.reshape(s.total_kv, s.heads_kv, half), \
        k_rope.reshape(s.total_kv, s.heads_kv, rope)
    cu = _cu_seqlens([s.total_q])

    return {
        # One FIA baseline per layout so each row has a same-layout reference.
        "fia_mla_bsnd": lambda: torch_npu.npu_fused_infer_attention_score(
            q_nope, k_nope, v,
            atten_mask=mask,
            query_rope=q_rope, key_rope=k_rope,
            num_heads=s.heads_q, scale=s.softmax_scale, input_layout="BSND",
            num_key_value_heads=s.heads_kv, sparse_mode=3,
        ),
        "fia_mla_tnd": lambda: torch_npu.npu_fused_infer_attention_score(
            q_nope_t, k_nope_t, v_t,
            atten_mask=mask,
            query_rope=q_rope_t, key_rope=k_rope_t,
            actual_seq_lengths=[s.total_q], actual_seq_lengths_kv=[s.total_kv],
            num_heads=s.heads_q, scale=s.softmax_scale, input_layout="TND",
            num_key_value_heads=s.heads_kv, sparse_mode=3,
        ),
        # FA3 -> BSND via flash_attn_func / flash_attn_with_kvcache, TND via varlen.
        "fa3_func_bsnd": lambda: fa3_func(
            q, k, v, softmax_scale=s.softmax_scale, causal=True,
        ),
        "fa3_kvcache_bsnd": lambda: fa3_kvcache(
            q, k, v, cache_seqlens=s.total_kv, softmax_scale=s.softmax_scale, causal=True,
        ),
        "fa3_varlen_tnd": lambda: fa3_varlen(
            q_t, k_t, v_t, cu, cu,
            max_seqlen_q=s.total_q, max_seqlen_k=s.total_kv,
            softmax_scale=s.softmax_scale, causal=True,
        ),
        # FA4 -> BSND via flash_attn_func, TND via varlen.
        "fa4_func_bsnd": lambda: fa4_func(
            q, k, v, softmax_scale=s.softmax_scale, causal=True,
        ),
        "fa4_varlen_tnd": lambda: fa4_varlen(
            q_t, k_t, v_t, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=s.total_q, max_seqlen_k=s.total_kv,
            softmax_scale=s.softmax_scale, causal=True,
        ),
    }


def _decode_impls():
    s = DECODE
    nope = s.head_dim_q - s.rope_dim
    q_rope = _rand(1, s.total_q, s.heads_q, s.rope_dim)
    qv = _rand(1, s.total_q, s.heads_q, nope)
    k_rope = _rand(1, s.total_kv, s.heads_kv, s.rope_dim)
    v = _rand(1, s.total_kv, s.heads_kv, s.head_dim_v)
    # TND: batch 1, so the BSND tensors flatten to (total, heads, dim) views.
    q_rope_t, qv_t = q_rope.reshape(s.total_q, s.heads_q, s.rope_dim), \
        qv.reshape(s.total_q, s.heads_q, nope)
    k_rope_t, v_t = k_rope.reshape(s.total_kv, s.heads_kv, s.rope_dim), \
        v.reshape(s.total_kv, s.heads_kv, s.head_dim_v)
    cu_q = _cu_seqlens([s.total_q])
    cu_k = _cu_seqlens([s.total_kv])

    return {
        "fia_mla_bsnd": lambda: torch_npu.npu_fused_infer_attention_score(
            qv, v, v,
            query_rope=q_rope, key_rope=k_rope,
            num_heads=s.heads_q, scale=s.softmax_scale, input_layout="BSND",
            num_key_value_heads=s.heads_kv, sparse_mode=0,
        ),
        "fia_mla_tnd": lambda: torch_npu.npu_fused_infer_attention_score(
            qv_t, v_t, v_t,
            query_rope=q_rope_t, key_rope=k_rope_t,
            actual_seq_lengths=[s.total_q], actual_seq_lengths_kv=[s.total_kv],
            num_heads=s.heads_q, scale=s.softmax_scale, input_layout="TND",
            num_key_value_heads=s.heads_kv, sparse_mode=0,
        ),
        "fa3_func_bsnd": lambda: fa3_func(
            q_rope, k_rope, v, qv=qv, softmax_scale=s.softmax_scale,
        ),
        "fa3_kvcache_bsnd": lambda: fa3_kvcache(
            q_rope, k_rope, v, qv=qv,
            cache_seqlens=s.total_kv, softmax_scale=s.softmax_scale,
        ),
        "fa3_varlen_tnd": lambda: fa3_varlen(
            q_rope_t, k_rope_t, v_t, cu_q, cu_k,
            max_seqlen_q=s.total_q, max_seqlen_k=s.total_kv,
            softmax_scale=s.softmax_scale, qv=qv_t,
        ),
        "fa4_func_bsnd": lambda: fa4_func(
            q_rope, k_rope, v, qv=qv, softmax_scale=s.softmax_scale,
        ),
        "fa4_varlen_tnd": lambda: fa4_varlen(
            q_rope_t, k_rope_t, v_t, qv=qv_t,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=s.total_q, max_seqlen_k=s.total_kv,
            softmax_scale=s.softmax_scale,
        ),
    }


def _require_npu():
    try:
        if not torch_npu.npu.is_available() or torch_npu.npu.device_count() < 1:
            raise SystemExit("NPU 不可用：未检测到 Ascend 设备，纯 kernel 对比需要真实设备")
        print(f"设备：{torch_npu.npu.get_device_name()}")
    except SystemExit:
        raise
    except Exception as exc:  # 驱动缺失时 torch_npu 常常直接抛 ImportError/RuntimeError
        raise SystemExit(f"NPU 不可用：{exc!r}")


def _require_cann_92():
    """The FIA baseline is only valid on CANN 9.2; abort otherwise."""
    version = str(torch_npu.utils.get_cann_version())
    home = os.environ.get("ASCEND_HOME_PATH", "<unset>")
    if not version.startswith(REQUIRED_CANN_PREFIX):
        raise SystemExit(
            f"FIA 基线要求 CANN {REQUIRED_CANN_PREFIX}，当前 {version}"
            f"（ASCEND_HOME_PATH={home}）"
        )
    print(f"CANN 版本：{version}  (ASCEND_HOME_PATH={home})")


def _profile_impl(scenario_name, impl_name, fn):
    """Profile ``STEPS`` calls of ``fn`` and return the export directory."""
    import torch_npu.profiler as profiler

    out_dir = PERF_OUTPUTS / (
        f"kernel_bench_{scenario_name}_{impl_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()

    experimental_config = profiler._ExperimentalConfig(
        profiler_level=profiler.ProfilerLevel.Level0,
        data_simplification=False,
        export_type=profiler.ExportType.Text,
    )
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        on_trace_ready=profiler.tensorboard_trace_handler(
            str(out_dir), analyse_flag=True
        ),
        experimental_config=experimental_config,
    ):
        for _ in range(STEPS):
            fn()
        torch.npu.synchronize()
    return out_dir


def _is_compute_kernel(name, patterns):
    return any(pattern in name for pattern in patterns)


def _kernel_us_per_call(out_dir, steps=STEPS):
    """Map ``Op Name`` -> us/call, using the median task duration per call.

    A call may launch several tasks of the same name (e.g. two ``Fill``s), so the
    median is scaled by tasks-per-call instead of being summed twice.
    """
    durations = {}
    for path in sorted(out_dir.rglob("op_summary*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                durations.setdefault(row["Op Name"], []).append(
                    float(row["Task Duration(us)"])
                )
    return {
        name: statistics.median(values) * len(values) / steps
        for name, values in durations.items()
    }


def _compute_us(per_call, patterns):
    """Split ``per_call`` into (compute us, matched kernels, excluded kernels)."""
    matched = {n: us for n, us in per_call.items() if _is_compute_kernel(n, patterns)}
    excluded = {n: us for n, us in per_call.items() if n not in matched}
    return sum(matched.values()), matched, excluded


def _print_impl(impl, compute_us, matched, excluded):
    print(f"[{impl}] MLA 计算内核 {compute_us:.2f} us/call")
    if not matched:
        print("    !! 未匹配到 MLA 计算内核，请核对白名单 FA_COMPUTE / FIA_COMPUTE")
    for name, us in sorted(matched.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {us:8.2f} us/call  {name}")
    for name, us in sorted(excluded.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {us:8.2f} us/call  [excluded] {name}")


def _write_summary_csv(rows):
    PERF_OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "impl", "category", "name", "value", "detail"])
        writer.writerows(rows)


def main():
    _require_npu()
    _require_cann_92()
    print(f"产出：{PERF_OUTPUTS}")
    print(f"纯 kernel 口径：{STEPS} 次调用，只统计 MLA 计算内核 "
          f"(FA: {', '.join(FA_COMPUTE_PATTERNS)} | FIA: {', '.join(FIA_COMPUTE_PATTERNS)})")
    print(f"门限：每个 MLA 行 candidate / same-layout FIA <= {KERNEL_GATE:.2f}x")
    torch.npu.synchronize()

    csv_rows = []
    failures = []

    for scenario, builder in ((PREFILL, _prefill_impls), (DECODE, _decode_impls)):
        print("\n" + "=" * 78)
        print(scenario_summary(scenario))
        print("=" * 78)
        results = {}
        for impl_name, fn in builder().items():
            patterns = FIA_COMPUTE_PATTERNS if impl_name.startswith("fia_") else FA_COMPUTE_PATTERNS
            out_dir = _profile_impl(scenario.name, impl_name, fn)
            per_call = _kernel_us_per_call(out_dir)
            compute_us, matched, excluded = _compute_us(per_call, patterns)
            _print_impl(impl_name, compute_us, matched, excluded)
            print(f"    trace: {out_dir}")
            if not matched:
                failures.append(
                    f"{scenario.name}: {impl_name} 未匹配到 MLA 计算内核，"
                    f"op 名={sorted(per_call)}"
                )
                continue
            results[impl_name] = compute_us
            for name, us in sorted(matched.items(), key=lambda kv: kv[1], reverse=True):
                csv_rows.append([scenario.name, impl_name, "kernel", name, f"{us:.4f}", ""])
            for name, us in sorted(excluded.items(), key=lambda kv: kv[1], reverse=True):
                csv_rows.append([scenario.name, impl_name, "excluded", name, f"{us:.4f}", ""])
            csv_rows.append(
                [scenario.name, impl_name, "total", "MLA_COMPUTE", f"{compute_us:.4f}",
                 "sum of matched compute kernels"]
            )

        print()
        for label, baseline, candidates in GATED:
            base = results.get(baseline)
            if not base:
                failures.append(f"{scenario.name}: 缺少 {baseline}，无法判定 {label} 门限")
                continue
            for name in candidates:
                if name not in results:
                    failures.append(f"{scenario.name}: 缺少 {name}")
                    continue
                ratio = results[name] / base
                ok = ratio <= KERNEL_GATE
                status = "PASS" if ok else "FAIL"
                print(
                    f"  [{label}] {name:16s} {results[name]:8.2f} us/call  vs {baseline}"
                    f" {base:8.2f} us/call  {ratio:6.3f}x  (gate {KERNEL_GATE:.2f}x)  {status}"
                )
                csv_rows.append(
                    [scenario.name, name, "gate", baseline, f"{ratio:.4f}",
                     f"{status} {label} gate {KERNEL_GATE:.2f}x"]
                )
                if not ok:
                    failures.append(
                        f"{scenario.name}: {name}/{baseline} = {ratio:.3f}x > {KERNEL_GATE:.2f}x"
                    )

    _write_summary_csv(csv_rows)
    print(f"\n汇总表已写入 {SUMMARY_CSV}")

    if failures:
        print("\n结论：FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\n结论：PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
