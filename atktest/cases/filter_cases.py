#!/usr/bin/env python3
"""把既有精度用例裁剪成目标设备可直接执行的 memory_check 用例。

用法:
    python3 cases/filter_cases.py --device 950 [--interface func_v3] [--dry-run]

950 会剔除以下用例（910 原样透传）：
  1. 分页 KV cache 且 page_size 不在 {128, 256, 512, 1024}；
  2. softcap > 0。

分页判定沿用生成器 ``generate_flash_attn_with_kvcache_v3.py`` 的约定：
``page_table.range_values`` 不是字符串 "default" 即为分页，此时
``page_size = k_cache.shape[1]``。

输出：``cases/<device>/<key>.json``，``id`` 重排为 0..N-1。
"""

import argparse
import collections
import json
from pathlib import Path

import yaml

ATKTEST_DIR = Path(__file__).resolve().parent.parent
MATRIX_PATH = ATKTEST_DIR / "atk_memory_check_matrix.yaml"
PAGED_PAGE_SIZES_950 = (128, 256, 512, 1024)


def load_interfaces():
    with MATRIX_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["interfaces"]


def find_input(case, name):
    for item in case.get("inputs", ()):
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return None


def is_paged(case):
    page_table = find_input(case, "page_table")
    return bool(page_table) and page_table.get("range_values") != "default"


def drop_reason(case, device):
    """返回 None 表示保留，否则返回剔除原因。"""
    if device != "950":
        return None

    if is_paged(case):
        k_cache = find_input(case, "k_cache")
        shape = k_cache.get("shape") if k_cache else None
        page_size = shape[1] if shape and len(shape) > 1 else None
        if page_size not in PAGED_PAGE_SIZES_950:
            return f"page_size={page_size}"

    softcap = find_input(case, "softcap")
    value = softcap.get("range_values") if softcap else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return "softcap>0"

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["910", "950"])
    parser.add_argument("--interface", help="接口 key，缺省为矩阵中全部接口")
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

    out_dir = ATKTEST_DIR / "cases" / args.device
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for entry in selected:
        key = entry["key"]
        supported = {str(item) for item in entry.get("devices", ())}
        if args.device not in supported:
            print(f"[skip] {key}: 该接口不支持 Ascend {args.device}")
            continue

        source = (ATKTEST_DIR / entry["case_source"]).resolve()
        if not source.is_file():
            print(f"[error] {key}: 找不到源用例 {source}")
            continue

        with source.open(encoding="utf-8") as handle:
            cases = json.load(handle)

        kept = []
        reasons = collections.Counter()
        for case in cases:
            reason = drop_reason(case, args.device)
            if reason:
                reasons[reason] += 1
                continue
            kept.append(case)

        for index, case in enumerate(kept):
            case["id"] = index

        detail = ""
        if reasons:
            detail = "  (" + ", ".join(f"{name}: {count}" for name, count in reasons.most_common()) + ")"
        print(f"{key}: kept {len(kept)}/{len(cases)}{detail}")

        if not args.dry_run:
            target = out_dir / f"{key}.json"
            with target.open("w", encoding="utf-8") as handle:
                json.dump(kept, handle, ensure_ascii=False, indent=2)
                handle.write("\n")


if __name__ == "__main__":
    main()
