#!/usr/bin/env python3
"""把 ``cases/source`` 的全量用例转成目标设备可直接执行的 accuracy 用例。

用法:
    python3 cases/filter_cases.py --device 910 [--interface func_v3] [--dry-run]

910 与 950 使用同一份用例，不做裁剪；本脚本只负责按设备挑选接口、
重排 ``id``，**不改写精度标准**：用例里的 ``cv_fused_double_benchmark``
由各 plugin 目录自带的比较器注册（见 plugins/<key>/compare_cv_fused_double_benchmark.py）。

输出：``cases/<device>/<key>.json``，``id`` 重排为 0..N-1。
"""

import argparse
import json
from pathlib import Path

import yaml

ACCURACY_DIR = Path(__file__).resolve().parent.parent
MATRIX_PATH = ACCURACY_DIR / "atk_accuracy_matrix.yaml"


def load_interfaces():
    with MATRIX_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["interfaces"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["910", "950"])
    parser.add_argument("--interface", help="接口 key，缺省为矩阵中全部接口")
    parser.add_argument("--limit", type=int, help="每个接口只保留前 N 条，用于冒烟")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = parser.parse_args()

    interfaces = load_interfaces()
    keys = [entry["key"] for entry in interfaces]
    if args.interface:
        if args.interface not in keys:
            raise SystemExit(f"未知接口 {args.interface}，可选: {', '.join(keys)}")
        selected = [entry for entry in interfaces if entry["key"] == args.interface]
    else:
        selected = interfaces

    out_dir = ACCURACY_DIR / "cases" / args.device
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for entry in selected:
        key = entry["key"]
        supported = {str(item) for item in entry.get("devices", ())}
        if args.device not in supported:
            print(f"[skip] {key}: 该接口不支持 Ascend {args.device}")
            continue

        source = (ACCURACY_DIR / entry["case_source"]).resolve()
        if not source.is_file():
            print(f"[error] {key}: 找不到源用例 {source}")
            continue

        with source.open(encoding="utf-8") as handle:
            cases = json.load(handle)

        if args.limit:
            cases = cases[: args.limit]

        for index, case in enumerate(cases):
            case["id"] = index

        print(f"{key}: {len(cases)} cases")

        if not args.dry_run:
            target = out_dir / f"{key}.json"
            with target.open("w", encoding="utf-8") as handle:
                json.dump(cases, handle, ensure_ascii=False, indent=2)
                handle.write("\n")


if __name__ == "__main__":
    main()
