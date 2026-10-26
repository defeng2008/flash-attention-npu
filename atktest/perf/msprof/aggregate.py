"""把同一接口的多轮 ``msprof --quick`` 结果按用例取中位数，合成 performance.json。

共享机器上单轮结果常被其他进程的瞬时尖峰污染：msprof quick 的解析在
``task_time.csv`` 行数不能被迭代数整除时会退化为按位置分片，干扰尖峰会被
平均进来（实测同一 case 两次可差 4 倍）。多轮取同一条用例的中位数可以剔掉
这类偶发值；只保留**每一轮都有效**的用例，避免单边数据混入。

  python3 aggregate.py func_v4 kvcache_v3
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _write_report(key, merged):
    lines = [
        "# 性能评估结果",
        "",
        f"- **Operator**: {key}",
        f"- **Passes**: {merged['passes']}（逐用例取中位数）",
        f"- **Timing method**: {merged['timing_method']}",
        "",
        "## 性能对比",
        "",
        "| Case | Shape | DType | BSND(us) | BNSD(us) | 加速比 |",
        "| ---- | ----- | ----- | -------- | -------- | ------ |",
    ]
    for case in merged["per_case"]:
        lines.append(
            f"| {case['case']} | {case['shape']} | {case['dtype']} | "
            f"{case['ref_us']:.2f} | {case['asc_us']:.2f} | {case['speedup']:.3f} |"
        )
    lines += [
        "",
        "## 全量汇总（平均值口径）",
        "",
        "| 指标 | 值 |",
        "| ---- | -- |",
        f"| 有效用例 | {merged['n_cases_valid']}/{merged['n_cases_total']} |",
        f"| 平均加速比（>1 表示 BNSD 更快） | {merged['mean_speedup']:.4f} |",
        f"| BSND 平均耗时 (us) | {merged['mean_ref_us']:.2f} |",
        f"| BNSD 平均耗时 (us) | {merged['mean_asc_us']:.2f} |",
        f"| 平均劣化 | {(merged['mean_asc_us'] / merged['mean_ref_us'] - 1) * 100:.2f}% |",
        "",
    ]
    (ROOT / key / "perf_report.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate(key):
    key_dir = ROOT / key
    passes = sorted(key_dir.glob("performance.pass*.json"))
    if len(passes) < 2:
        print(f"[{key}] 少于两轮结果，跳过合并（{len(passes)} 轮）")
        return False

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in passes]
    per_pass = [{case["case"]: case for case in run.get("per_case", [])} for run in runs]
    common = set(per_pass[0])
    for cases in per_pass[1:]:
        common &= set(cases)
    if not common:
        print(f"[{key}] 各轮没有共同的有效用例，跳过合并")
        return False

    per_case = []
    for case_id in sorted(common):
        ref = statistics.median(cases[case_id]["ref_us"] for cases in per_pass)
        asc = statistics.median(cases[case_id]["asc_us"] for cases in per_pass)
        per_case.append(
            {**per_pass[-1][case_id], "ref_us": ref, "asc_us": asc, "speedup": ref / asc}
        )

    speedups = [case["speedup"] for case in per_case]
    merged = {
        **runs[-1],
        "n_cases_valid": len(per_case),
        "mean_speedup": statistics.mean(speedups),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "mean_ref_us": statistics.mean(case["ref_us"] for case in per_case),
        "mean_asc_us": statistics.mean(case["asc_us"] for case in per_case),
        "per_case": per_case,
        "passes": len(runs),
    }
    (key_dir / "performance.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(key, merged)
    print(f"[{key}] 合并 {len(runs)} 轮 -> {len(per_case)} 用例，均值加速比 {merged['mean_speedup']:.4f}")
    return True


def main(argv):
    keys = argv[1:]
    if not keys:
        raise SystemExit("用法: python3 aggregate.py <key> [key ...]")
    return 0 if all(aggregate(key) for key in keys) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
