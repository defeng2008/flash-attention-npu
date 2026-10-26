"""由 ``../shapes.yaml`` 生成各接口的 ``<key>_perf_cases.jsonl``（msprof 用例）。

jsonl 只负责用例条数与 shape/dtype（供 msprof 报告展示）；真实输入由同目录
``model.py`` 的 ``get_input_groups()`` 按 case 序号提供，两边必须一致。

  python3 gen_cases.py
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SHAPES_PATH = ROOT.parent / "shapes.yaml"

# key -> shapes.yaml 中的分组
KEYS = {
    "func_v2": "func",
    "kvcache_v2": "kvcache",
    "func_v3": "func",
    "kvcache_v3": "kvcache",
    "func_v4": "func",
}

DTYPES = {"fp16": "float16", "bf16": "bfloat16"}


def _tensor(name, shape, dtype):
    return {"name": name, "type": "tensor", "shape": list(shape), "dtype": dtype}


def _func_inputs(item):
    dtype = DTYPES[item["dtype"]]
    b, s, h, h_kv, d = item["b"], item["s"], item["h"], item["h_kv"], item["d"]
    return [
        _tensor("query", (b, s, h, d), dtype),
        _tensor("key", (b, s, h_kv, d), dtype),
        _tensor("value", (b, s, h_kv, d), dtype),
    ]


def _kvcache_inputs(item):
    dtype = DTYPES[item["dtype"]]
    b, sq, s_alloc = item["b"], item["sq"], item["s_alloc"]
    h, h_kv, d = item["h"], item["h_kv"], item["d"]
    return [
        _tensor("query", (b, sq, h, d), dtype),
        _tensor("key", (b, s_alloc, h_kv, d), dtype),
        _tensor("value", (b, s_alloc, h_kv, d), dtype),
        _tensor("cache_seqlens", (b,), "int32"),
    ]


def main():
    groups = yaml.safe_load(SHAPES_PATH.read_text(encoding="utf-8"))
    for key, group in KEYS.items():
        builder = _func_inputs if group == "func" else _kvcache_inputs
        rows = [{"inputs": builder(item)} for item in groups[group]]
        out_path = ROOT / key / f"{key}_perf_cases.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        print(f"{key}: {len(rows)} cases -> {out_path}")


if __name__ == "__main__":
    main()
