#!/usr/bin/env python3
"""Generate the MLA-910 ATK delivery tree under docs/test/atk/.

Facts encoded here were read from:
  * flash-attention-npu/flash_attn_npu_{3,4}/flash_attn_npu_interface.py  (signatures)
  * flash-attention-npu/tests/perf/bench_mla_vs_fia.py                    (candidate + FIA calls)
  * flash-attention-npu/tests/perf/mla_user_scenarios.py + /home/w00811046/code/mla/usershape.txt
  * flash-attention-npu/docs/flash_attention_v2_v3_atk_delivery/**         (ATK schema/format)
  * flash-attention-npu/docs/mla/daoailab_mla_analysis.md                  (daoailab GPU entry)
Run:  python3 docs/test/atk/tools/gen_atk_mla.py

交付形态（2026-11 修订）：
  * 每个接口目录下的 <stem>.yaml 是**唯一真值**，用例由 `atk case -f <stem>.yaml -p .`
    现场生成到 result/<stem>/json/all_<stem>.json，run_*.sh 再把它拷进 cases/。
  * 跨输入的形状耦合由同目录的 generate_<iface>.py 钩子在每条用例上统一改写，
    所以 yaml 只描述"每一维的候选取值"，不需要（也不可能）写死组合。
  * 用户场景 anchor（usershape.txt 的 198 prefill / 1025 decode）单独放在
    cases/user_scenarios_<base>.json，由 run_*.sh 串跑，报告与生成用例分开。
"""
import json
import os
from pathlib import Path

ROOT = Path("/home/w00811046/code/fan/mlatest/flash-attention-npu/docs/test/atk")
SCALE = 192 ** -0.5          # DeepSeek MLA softmax scale
BLOCK = 128                  # paged KV block size
BF16 = "bf16"

IFACES = {
    "fa3_func": dict(pkg="flash_attn_npu_3", fn="flash_attn_func", gpu="fa3", layout="bsnd",
                     paged=False, kvcache=False),
    "fa3_varlen": dict(pkg="flash_attn_npu_3", fn="flash_attn_varlen_func", gpu="fa3", layout="tnd",
                       paged=False, kvcache=False),
    "fa3_kvcache": dict(pkg="flash_attn_npu_3", fn="flash_attn_with_kvcache", gpu="fa3", layout="bsnd",
                        paged=True, kvcache=True),
    "fa4_func": dict(pkg="flash_attn_npu_4", fn="flash_attn_func", gpu="fa4", layout="bsnd",
                     paged=False, kvcache=False),
    "fa4_varlen": dict(pkg="flash_attn_npu_4", fn="flash_attn_varlen_func", gpu="fa4", layout="tnd",
                       paged=True, kvcache=False),
}
IFACE_ORDER = ["fa3_func", "fa3_varlen", "fa3_kvcache", "fa4_func", "fa4_varlen"]

# category -> (kind, nodes file, run file, bm_device, task, multi, 说明)
CATEGORIES = {
    "double_benchmark": dict(kind="acc", nodes="nodes_accuracy_gpu.yaml", run="run_accuracy.sh",
                             bm="cpu", task="accuracy", multi=True,
                             desc="npu 候选 vs GPU(daoailab) + CPU 双标杆精度对比"),
    "single_benchmark_cpu": dict(kind="acc", nodes="nodes_accuracy.yaml", run="run_accuracy.sh",
                                 bm="cpu", task="accuracy", multi=False,
                                 desc="npu 候选 vs CPU fp32 标杆精度对比"),
    "single_benchmark_cann": dict(kind="acc", nodes="nodes_accuracy_cann.yaml", run="run_accuracy.sh",
                                  bm="npu", task="accuracy", multi=False,
                                  desc="npu 候选 vs CANN npu_fused_infer_attention_score 标杆精度对比"),
    "gpu_mem_compare": dict(kind="mem", nodes="nodes_memory_gpu.yaml", run="run_memory.sh",
                            bm="gpu", task="memory_device", multi=True,
                            desc="npu(FA3/FA4) vs GPU(daoailab) 峰值显存对比（需要 GPU 主机）"),
}
CAT_ORDER = ["double_benchmark", "single_benchmark_cpu", "single_benchmark_cann", "gpu_mem_compare"]


def stem_of(iface, cat, paged=False):
    """用例/yaml 的 stem：mem 类加 _mem 后缀，paged 变体加 _paged 后缀。"""
    stem = iface + ("_mem" if CATEGORIES[cat]["kind"] == "mem" else "")
    return stem + ("_paged" if paged else "")


def variants(iface, cat):
    """[(stem, paged), ...]。

    cann 类不出 paged 变体：标杆 FIA 不接受 page_table 入参，出了也无法同口径比对。
    """
    if not IFACES[iface]["paged"] or cat == "single_benchmark_cann":
        return [(stem_of(iface, cat, False), False)]
    return [(stem_of(iface, cat, False), False), (stem_of(iface, cat, True), True)]


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if path.suffix == ".sh":
        path.chmod(path.stat().st_mode | 0o111)


# --------------------------------------------------------------------------------------
# nodes yaml
# --------------------------------------------------------------------------------------
NODES = {
    "nodes_accuracy.yaml": """nodes:
  - backend: npu
    devices: [0]
    task: [accuracy]
  - backend: cpu
    task: [accuracy]
""",
    "nodes_accuracy_gpu.yaml": """nodes:
  - backend: npu
    devices: [0]
    task: [accuracy]
  - backend: cpu
    task: [accuracy]
  - backend: gpu
    host: 192.168.0.100  # TODO: 改为 GPU 主机 IP
    port: 9090
    devices: [0]
    task: [accuracy]
""",
    "nodes_accuracy_cann.yaml": """# 候选 npu 与 FIA 标杆 npu 同设备；标杆节点带 is_bm，由 execute 按节点名判别。
# 注意：不能给候选节点加 is_compare: false，否则 ATK 会把它排除在精度比对之外。
nodes:
  - backend: npu
    name: candidate
    devices: [0]
    task: [accuracy]
  - backend: npu
    name: fia_mla
    devices: [0]
    task: [accuracy]
    is_bm: true
""",
    "nodes_memory_gpu.yaml": """nodes:
  - backend: npu
    devices: [0]
    task: [memory_device]
  - backend: gpu
    host: 192.168.0.100  # TODO: 改为 GPU 主机 IP
    port: 9090
    devices: [0]
    task: [memory_device]
""",
}


# --------------------------------------------------------------------------------------
# run scripts
# --------------------------------------------------------------------------------------
RUN = """#!/usr/bin/env bash
# {cat} —— {desc}
#
# 流程（yaml 是唯一真值，用例全部由 atk case 生成、不再手写）：
#   1) atk case -f <stem>.yaml -p .  ->  result/<stem>/json/all_<stem>.json
#   2) cp 到 <cat>/cases/all_<stem>.json（交付里可直接打开查看）
#   3) atk task -c cases/all_<stem>.json ... 执行并产出精度/显存报告
#   4) 最后补跑用户场景 anchor：cases/user_scenarios_{base}.json（报告与上一步分开）
#
# 环境约定：conda 环境 cann（已装 atk + atk_private_precision）/ CANN set_env.sh / TMPDIR
set -euo pipefail

export CONDA_ENV="${{ATK_CONDA_ENV:-cann}}"
export CANN_ROOT="${{CANN_ROOT:-/home/w00811046/cann/cann-9.2.0-clean/ascend-toolkit}}"
export TMPDIR="${{TMPDIR:-/home/w00811046/tmp}}"
ATK="${{ATK_BIN:-atk}}"

# 加载 CANN 运行环境（torch_npu 需要 libhccl.so 等）
if [ -f "${{CANN_ROOT}}/set_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${{CANN_ROOT}}/set_env.sh"
else
  echo "[warn] 未找到 ${{CANN_ROOT}}/set_env.sh，请用 CANN_ROOT 指定含 set_env.sh 的 CANN 目录" >&2
fi
# 已激活环境里有 atk 就直接用，否则经 conda run 进入 CONDA_ENV
if command -v "${{ATK}}" >/dev/null 2>&1; then
  ATK_CMD=("${{ATK}}")
else
  ATK_CMD=(conda run --no-capture-output -n "${{CONDA_ENV}}" "${{ATK}}")
fi

INTERFACE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
INTERFACE_NAME="$(basename "${{INTERFACE_DIR}}")"
DELIVERY_DIR="$(cd "${{INTERFACE_DIR}}/../../.." && pwd)"
CASE_DIR="${{DELIVERY_DIR}}/{cat}/cases"
NODES_FILE="{nodes}"
SEED="${{ATK_SEED:-20261113}}"
if [ "{tag}" = "multi" ]; then _SP=""; else _SP="--single_process"; fi

mkdir -p "${{CASE_DIR}}"
cd "${{INTERFACE_DIR}}"

# yaml -> cases/all_<stem>.json
gen_case() {{
  local yaml="$1" stem="$2"
  "${{ATK_CMD[@]}}" case -f "${{yaml}}" -p . -s "${{SEED}}" --log warning
  cp "result/${{stem}}/json/all_${{stem}}.json" "${{CASE_DIR}}/all_${{stem}}.json"
  echo "[gen] ${{yaml}} -> ${{CASE_DIR}}/all_${{stem}}.json"
}}

# cases json -> 精度/显存报告
run_case() {{
  "${{ATK_CMD[@]}}" task \\
    -c "$1" \\
    -n "${{NODES_FILE}}" \\
    -p . \\
    --task {task} \\
    --bm_device {bm} \\
    ${{_SP}} \\
    --log warning \\
    --save_data input \\
    --save_data output
}}

STEMS=()
for yaml in "${{INTERFACE_NAME}}"*.yaml; do
  [ -e "${{yaml}}" ] || continue
  stem="$(basename "${{yaml}}" .yaml)"
  gen_case "${{yaml}}" "${{stem}}"
  STEMS+=("${{stem}}")
done
if [ "${{#STEMS[@]}}" -eq 0 ]; then
  echo "[error] ${{INTERFACE_DIR}} 下没有 ${{INTERFACE_NAME}}*.yaml" >&2
  exit 1
fi
for stem in "${{STEMS[@]}}"; do
  run_case "${{CASE_DIR}}/all_${{stem}}.json"
done

# 用户场景 anchor，单独一个 json，报告与上面的生成用例分开
run_case "${{CASE_DIR}}/user_scenarios_{base}.json"
echo "[done] ${{INTERFACE_NAME}} ({cat}) -> ${{CASE_DIR}}"
"""

RUN_CAT = """#!/usr/bin/env bash
# 汇总执行 {cat} —— {desc}
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
for iface in {ifaces}; do
  bash "${{HERE}}/interfaces/${{iface}}/{run}"
done
"""


# --------------------------------------------------------------------------------------
# execute adapter
# --------------------------------------------------------------------------------------
EXECUTE = '''"""ATK cpu/gpu/npu adapter for @@IFACE@@ (@@CAT@@)."""

import inspect
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(PROJECT_ROOT / "flash-attention-npu"))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_@@IFACE@@ import @@FN@@ as cpu_impl

@@NPU_IMPORT@@

FIA_NODE_NAME = "@@BM_NAME@@"
DAOAI_ROOT = os.environ.get("DAOAI_ROOT", "/home/w00811046/code/mla/daoailab")


@register("@@API_TYPE@@")
class @@CLS@@(BaseApi):
    _ACCEPTED = set(inspect.signature(npu_fn).parameters)

    @staticmethod
    def _move(kw, dev):
        return {n: (v.to(dev) if isinstance(v, torch.Tensor) else v) for n, v in kw.items()}

    def __call__(self, input_data, with_output=False):
        kw = dict(input_data.kwargs)
        if self.device == "cpu":
            return cpu_impl(**kw)
        if self.device == "gpu":
            return self._gpu(kw)
        if self.device == "npu":
            if FIA_NODE_NAME and getattr(self, "name", "") == FIA_NODE_NAME:
                return self._fia(kw)
            return npu_fn(**self._move({n: v for n, v in kw.items() if n in self._ACCEPTED},
                                       f"npu:{self.device_id}"))
        raise RuntimeError(f"Unsupported backend: {self.device}")

    def _gpu(self, kw):
        """daoailab GPU benchmark (只读依赖，无 GPU 环境时不执行)。"""
        if DAOAI_ROOT not in sys.path:
            sys.path.insert(0, DAOAI_ROOT)
        if "@@GPU_KIND@@" == "fa3":
            from hopper import flash_attn_interface as gpu_mod
            gpu_fn = gpu_mod.flash_attn_with_kvcache
        else:
            from flash_attn.cute import interface as gpu_mod
            gpu_fn = gpu_mod.flash_attn_func
        accepted = set(inspect.signature(gpu_fn).parameters)
        call = {n: v for n, v in kw.items() if n in accepted}
        return gpu_fn(**self._move(call, f"cuda:{self.device_id}"))

    def _fia(self, kw):
        """CANN npu_fused_infer_attention_score 标杆（映射对齐 tests/perf/bench_mla_vs_fia.py）。"""
        import torch_npu
        fia = torch_npu.npu_fused_infer_attention_score
        dev = f"npu:{self.device_id}"

        def _dt(t):
            # 标杆路径下 ATK 会把 bf16/fp16 输入升精度到 fp32，而 aclnnFusedInferAttentionScore
            # 的 MLA 分支只支持 bf16/fp16，这里按需回落到 bf16。
            if isinstance(t, torch.Tensor) and t.dtype in (torch.float32, torch.float64):
                return t.to(torch.bfloat16)
            return t

        q = _dt(kw["q"])
        k = _dt(kw.get("k", kw.get("k_cache")))
        v = _dt(kw.get("v", kw.get("v_cache")))
        qv = _dt(kw.get("qv"))
        scale = kw.get("softmax_scale")
        causal = bool(kw.get("causal", False))
        if kw.get("cu_seqlens_q") is not None:      # TND 布局
            cu_q, cu_k = kw["cu_seqlens_q"], kw["cu_seqlens_k"]
            if qv is None:
                half, rope = q.shape[-1] - 64, 64
                q_nope, q_rope = q[..., :half].contiguous(), q[..., half:].contiguous()
                k_nope, k_rope = k[..., :half].contiguous(), k[..., half:].contiguous()
                mask = None
                if causal:
                    mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool().to(dev)
                return fia(q_nope.to(dev), k_nope.to(dev), v.to(dev),
                           atten_mask=mask, query_rope=q_rope.to(dev), key_rope=k_rope.to(dev),
                           actual_seq_lengths=cu_q.tolist(), actual_seq_lengths_kv=cu_k.tolist(),
                           num_heads=q.shape[1], scale=scale, input_layout="TND",
                           num_key_value_heads=k.shape[1], sparse_mode=3 if causal else 0)[0]
            return fia(qv.to(dev), v.to(dev), v.to(dev),        # TND decode (absorbed)
                       query_rope=q.to(dev), key_rope=k.to(dev),
                       actual_seq_lengths=cu_q.tolist(), actual_seq_lengths_kv=cu_k.tolist(),
                       num_heads=qv.shape[1], scale=scale, input_layout="TND",
                       num_key_value_heads=v.shape[1], sparse_mode=0)[0]
        if qv is None:                              # BSND prefill
            half, rope = q.shape[-1] - 64, 64
            q_nope, q_rope = q[..., :half].contiguous().to(dev), q[..., half:].contiguous().to(dev)
            k_nope, k_rope = k[..., :half].contiguous().to(dev), k[..., half:].contiguous().to(dev)
            mask = None
            if causal:
                mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool().to(dev)
            return fia(q_nope, k_nope, v.to(dev), atten_mask=mask,
                       query_rope=q_rope, key_rope=k_rope, num_heads=q.shape[2], scale=scale,
                       input_layout="BSND", num_key_value_heads=k.shape[2],
                       sparse_mode=3 if causal else 0)[0]
        return fia(qv.to(dev), v.to(dev), v.to(dev),           # BSND decode (absorbed)
                   query_rope=q.to(dev), key_rope=k.to(dev), num_heads=qv.shape[2], scale=scale,
                   input_layout="BSND", num_key_value_heads=v.shape[2],
                   sparse_mode=0)[0]
'''


# --------------------------------------------------------------------------------------
# CPU golden benchmark
# --------------------------------------------------------------------------------------
BENCHMARK = '''"""MLA fp32 CPU golden for @@IFACE@@ (@@CAT@@)."""

import torch

BLOCK = 128


def _mask(sq, sk, causal, window_size):
    left, right = window_size
    if sk > 0 and left >= sk - 1:
        left = -1
    if sq > 0 and right >= sq - 1:
        right = -1
    if causal:
        right = 0
    if left < 0 and right == 0:
        return torch.triu(torch.ones(sq, sk, dtype=torch.bool), diagonal=sk - sq + 1)
    if left < 0 and right <= 0:
        return None
    q_pos = torch.arange(sq)[:, None]
    k_pos = torch.arange(sk)[None, :]
    center = q_pos + sk - sq
    return ((left >= 0) & (k_pos < center - left)) | ((right >= 0) & (k_pos > center + right))


def _paged(cache, table):
    blocks = [cache[int(p)] for p in table if int(p) >= 0]
    if not blocks:
        return cache[:0]
    return torch.cat(blocks, dim=0)


def _core(q, k, v, qv, scale, causal, window_size, length):
    """q/k/v: [S, H, D] single sequence; length: valid KV rows (None = all)."""
    qh = q.float().transpose(0, 1)
    if qv is not None:
        qnh = qv.float().transpose(0, 1)
        kh = k.float().transpose(0, 1)
        vh = v.float().transpose(0, 1)
        scores = (qnh @ vh.transpose(-1, -2) + qh @ kh.transpose(-1, -2)) * scale
        value = vh
    else:
        kh = k.float().transpose(0, 1)
        vh = v.float().transpose(0, 1)
        scores = (qh @ kh.transpose(-1, -2)) * scale
        value = vh
    if length is not None:
        scores = scores[..., :length]
        value = value[..., :length, :]
    mask = _mask(scores.shape[-2], scores.shape[-1], causal, window_size)
    if mask is not None:
        scores = scores + mask.to(scores.dtype) * -1e4
    probs = torch.softmax(scores, dim=-1)
    # NPU 内核把 P 以原精度(bf16)存回后再与 V 做 fp32 累加；CPU 标杆必须复现该量化，
    # 否则 fp32 的 P 会让近零元素的相对误差被放大（参考交付 benchmark_flash_attn_func_v3.py
    # 的 pvMM2(p_result.to(data_type), ...) 即此口径）。
    probs = probs.to(q.dtype).float()
    if value.shape[0] != probs.shape[0] and probs.shape[0] % value.shape[0] == 0:
        value = value.repeat_interleave(probs.shape[0] // value.shape[0], dim=0)
    return (probs @ value).transpose(0, 1)


def _bsnd(q, k, v, qv, scale, causal, window_size, lengths, page_table):
    outs = []
    for b in range(q.shape[0]):
        kb, vb = k[b], v[b]
        length = int(lengths[b]) if lengths is not None else None
        if page_table is not None:
            kb, vb = _paged(k, page_table[b]), _paged(v, page_table[b])
            kb, vb = kb[:length], vb[:length]
            length = None
        outs.append(_core(q[b], kb, vb, None if qv is None else qv[b], scale, causal, window_size, length))
    return torch.stack(outs).to(q.dtype)


def _tnd(q, k, v, qv, cu_q, cu_k, scale, causal, window_size, page_table=None, seqlens=None):
    outs = []
    for b in range(len(cu_q) - 1):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        kb, vb = k, v
        if page_table is not None:
            # paged KV：内核只接受 page_table + seqused_k，每 batch 的 KV 长度取自 seqused_k。
            kb, vb = _paged(k, page_table[b]), _paged(v, page_table[b])
            seql = int(seqlens[b])
            kb, vb = kb[:seql], vb[:seql]
            ks, ke = 0, seql
        else:
            ks, ke = int(cu_k[b]), int(cu_k[b + 1])
        outs.append(_core(q[qs:qe], kb[ks:ke], vb[ks:ke],
                          None if qv is None else qv[qs:qe], scale, causal, window_size, None))
    return torch.cat(outs, dim=0).to(q.dtype)


def @@FN@@(**kw):
    q = kw["q"]
    k = kw.get("k", kw.get("k_cache"))
    v = kw.get("v", kw.get("v_cache"))
    qv = kw.get("qv")
    scale = kw.get("softmax_scale")
    if scale is None:
        scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** -0.5
    causal = bool(kw.get("causal", False))
    window = tuple(kw.get("window_size") or (-1, -1))
    if kw.get("cu_seqlens_q") is not None:
        return _tnd(q, k, v, qv, kw["cu_seqlens_q"], kw.get("cu_seqlens_k"), scale, causal, window,
                    kw.get("page_table"), kw.get("seqused_k"))
    lengths = None
    if kw.get("cache_seqlens") is not None:
        cs = kw["cache_seqlens"]
        lengths = [cs] * q.shape[0] if isinstance(cs, int) else list(cs)
    if kw.get("seqused_k") is not None:
        lengths = list(kw["seqused_k"])
    return _bsnd(q, k, v, qv, scale, causal, window, lengths, kw.get("page_table"))
'''


# --------------------------------------------------------------------------------------
# case JSON builder
# --------------------------------------------------------------------------------------
def _tensor(name, dtype, shape, rng=(-5.0, 5.0), values=None):
    """values 非空时改用 ATK 的 *_fixed 数据类型，把张量内容固定为给定值。

    ATK 的普通 tensor 输入只接受 [min, max] 取值区间（data_torch.py 对长度为 2 的 list
    走随机生成），因此 cu_seqlens / page_table / seqused_k 这类索引张量必须用 fixed。
    """
    if values is not None:
        return {"name": name, "type": "tensor", "required": True, "dtype": f"{dtype}_fixed",
                "shape": list(shape), "range_values": values, "backward": False,
                "align_32B": None, "outlier_values": None}
    return {"name": name, "type": "tensor", "required": True, "dtype": dtype, "shape": list(shape),
            "range_values": list(rng), "backward": False, "align_32B": None, "outlier_values": None}


def _attr(name, dtype, value):
    return {"name": name, "type": "attr", "required": False, "dtype": dtype, "shape": None,
            "range_values": value, "backward": False, "align_32B": None, "outlier_values": None}


def _attr_tuple(name, value, dtype="int"):
    return [{"name": name, "type": "attr_tuple", "required": False, "dtype": dtype, "shape": None,
             "range_values": int(v), "backward": False, "align_32B": None, "outlier_values": None}
            for v in value]


def _seq(lengths):
    acc, out = 0, [0]
    for n in lengths:
        acc += n
        out.append(acc)
    return out


def _blocks(batch, nblk):
    """page_table：第 b 个 batch 使用全局页号 b*nblk .. b*nblk+nblk-1。"""
    return [[b * nblk + i for i in range(nblk)] for b in range(batch)]


def build_inputs(iface, spec):
    meta = IFACES[iface]
    form = spec["form"]
    B, Sq = spec["B"], spec["Sq"]
    Sk = spec["Sk"]
    paged = spec.get("paged", False)
    causal = spec["causal"]
    window = spec["window"]
    layout = meta["layout"]
    inputs = []
    if form == "prefill":
        Hq = Hk = 32
        Dq = Dk = 192
        Dv = 128
        qv = None
    else:
        Hq, Hk, Dq, Dk, Dv = 32, 1, 64, 64, 512
        qv = 512

    if meta["kvcache"]:
        cache_len = Sk
        if paged:
            nblk = (cache_len + BLOCK - 1) // BLOCK
            inputs.append(_tensor("q", BF16, (B, Sq, Hq, Dq)))
            inputs.append(_tensor("k_cache", BF16, (B * nblk, BLOCK, Hk, Dk)))
            inputs.append(_tensor("v_cache", BF16, (B * nblk, BLOCK, Hk, Dv)))
            inputs.append(_tensor("page_table", "int32", (B, nblk), values=_blocks(B, nblk)))
        else:
            inputs.append(_tensor("q", BF16, (B, Sq, Hq, Dq)))
            inputs.append(_tensor("k_cache", BF16, (B, cache_len, Hk, Dk)))
            inputs.append(_tensor("v_cache", BF16, (B, cache_len, Hk, Dv)))
        if qv is not None:
            inputs.append(_tensor("qv", BF16, (B, Sq, Hq, Dv)))
        inputs.append(_attr("cache_seqlens", "int", cache_len))
        inputs.append(_attr("softmax_scale", "float", SCALE))
        inputs.append(_attr("causal", "bool", causal))
        inputs.append(_attr_tuple("window_size", window))
        return inputs

    if layout == "tnd":
        Tq = B * Sq
        Tk = B * Sk
        cu_q, cu_k = _seq([Sq] * B), _seq([Sk] * B)
        inputs.append(_tensor("q", BF16, (Tq, Hq, Dq)))
        inputs.append(_tensor("k", BF16, (Tk, Hk, Dk)))
        inputs.append(_tensor("v", BF16, (Tk, Hk, Dv)))
        if qv is not None:
            inputs.append(_tensor("qv", BF16, (Tq, Hq, Dv)))
        if paged:
            nblk = (Sk + BLOCK - 1) // BLOCK
            inputs[1] = _tensor("k", BF16, (B * nblk, BLOCK, Hk, Dk))
            inputs[2] = _tensor("v", BF16, (B * nblk, BLOCK, Hk, Dv))
            inputs.append(_tensor("page_table", "int32", (B, nblk), values=_blocks(B, nblk)))
            inputs.append(_tensor("seqused_k", "int32", (B,), values=[Sk] * B))
        inputs.append(_tensor("cu_seqlens_q", "int32", (B + 1,), values=cu_q))
        if not paged:
            # paged KV 的契约是 page_table + seqused_k；同时给 cu_seqlens_k 会被内核拒绝
            # （"If cu_seqlens_k is passed in, then paged table is not supported"）。
            inputs.append(_tensor("cu_seqlens_k", "int32", (B + 1,), values=cu_k))
        inputs.append(_attr("max_seqlen_q", "int", Sq))
        inputs.append(_attr("max_seqlen_k", "int", Sk))
        inputs.append(_attr("softmax_scale", "float", SCALE))
        inputs.append(_attr("causal", "bool", causal))
        inputs.append(_attr_tuple("window_size", window))
        return inputs

    # BSND plain (fa3_func / fa4_func)
    inputs.append(_tensor("q", BF16, (B, Sq, Hq, Dq)))
    inputs.append(_tensor("k", BF16, (B, Sk, Hk, Dk)))
    inputs.append(_tensor("v", BF16, (B, Sk, Hk, Dv)))
    if qv is not None:
        inputs.append(_tensor("qv", BF16, (B, Sq, Hq, Dv)))
    inputs.append(_attr("softmax_scale", "float", SCALE))
    inputs.append(_attr("causal", "bool", causal))
    inputs.append(_attr_tuple("window_size", window))
    return inputs


def build_user_scenarios(iface, cat):
    meta = IFACES[iface]
    kind = CATEGORIES[cat]["kind"]
    api_type = f"{iface}_mla_{'mem' if kind == 'mem' else 'accuracy'}"
    # 用户场景 anchor：只保留 usershape.txt 里的两条真实业务 shape。
    # 这些用例**不走 atk case 生成**，是手写的固定 JSON，由 run_*.sh 串跑、报告单独出。
    # (form, B, Sq, Sk, causal, window, paged, label)
    rows = [
        ("prefill", 1, 198, 198, True, (-1, -1), False, "user_prefill_198"),
        ("decode", 1, 1, 1025, False, (-1, -1), False, "user_decode_1025"),
    ]
    cases = []
    for i, (form, B, Sq, Sk, causal, window, paged, label) in enumerate(rows):
        cases.append({
            "id": i, "default_seed": None, "name": f"{meta['pkg']}.{meta['fn']}",
            "aclnn_name": None, "triton_name": None, "kernel_name": None, "version": "v2.1",
            "expected_error_msg": None, "api": "pytorch", "api_type": api_type,
            "aclnn_api_type": "aclnn_function", "triton_api_type": "triton_function",
            "fusion_api_type": "fusion_function", "fusion_mode": None,
            "dist_api_type": "dist_function", "kernel_api_type": "kernel_function",
            "backward": False, "memo": label,
            "standard": {"acc": {"cv_fused_double_benchmark": {
                "max_re_ratio": 2.0, "avg_re_ratio": 1.2, "root_mean_squared_ratio": 1.2}},
                "perf": "not_key", "mem": 1.1},
            "outputs": 1,
            "inputs": build_inputs(iface, dict(form=form, B=B, Sq=Sq, Sk=Sk, causal=causal,
                                              window=window, paged=paged)),
            "acl_json": "", "method_inputs": None, "tensor_input": None, "compute_times": None,
            "save_name": None, "uuid": None, "downloaded": False, "is_boundary": False,
            "xrun_cs_name": None, "xrun_data": None, "strategy": None,
        })
    return cases


# --------------------------------------------------------------------------------------
# yaml：ATK 用例设计文件（本接口的唯一真值）
# --------------------------------------------------------------------------------------
# 每一维的候选取值：序列维 + 末维是"形状多样性"的来源；头数/hidden 会被
# generate_*.py 按 MLA 契约改写，这里给出只是为了 ATK 能随机出一个起点。
_Q_BSND = [[1, 2], [1, 64, 198, 512], [32], [64, 192]]
_K_BSND = [[1, 2], [1, 64, 198, 256, 384, 512, 1025], [1, 32], [64, 192]]
_V_BSND = [[1, 2], [1, 64, 198, 256, 384, 512, 1025], [1, 32], [128, 512]]
_QV_BSND = [[1, 2], [1, 64, 198, 512], [32], [512]]
_Q_TND = [[1, 64, 198, 512], [32], [64, 192]]
_K_TND = [[1, 64, 198, 256, 384, 512, 1025], [1, 32], [64, 192]]
_V_TND = [[1, 64, 198, 256, 384, 512, 1025], [1, 32], [128, 512]]
_QV_TND = [[1, 64, 198, 512], [32], [512]]
# paged 的 KV 存成 (num_blocks, block_size, H, D)；num_blocks = batch * ceil(kv_len/BLOCK)
_K_PAGED = [[1, 2], [BLOCK], [1, 32], [64, 192]]
_V_PAGED = [[1, 2], [BLOCK], [1, 32], [128, 512]]


def _in_tensor(name, dims, comment=None):
    """rank = len(dims) 的 bf16 张量输入；dims 是每一维的候选取值。"""
    lines = [f"  - name: {name}", "    type: tensor", "    required: false",
             "    dtypes: {values: [bf16]}", "    shapes:",
             f"      dim_numbers: {{values: [{len(dims)}]}}", "      dim_values:"]
    lines += [f"        - values: [{', '.join(str(v) for v in vs)}]" for vs in dims]
    lines += ["    ranges:", "      valid:", "        values: [[-5.0, 5.0]]"]
    return ([f"  # {comment}"] if comment else []) + lines


def _in_index(name, comment=None, dim_values=None, max_length=4096):
    """int32 索引张量输入；真实取值由 generate_*.py 用 *_fixed 写死。"""
    lines = [f"  - name: {name}", "    type: tensor", "    required: false",
             "    dtypes: {values: [int32]}", "    shapes:"]
    if dim_values:
        lines += [f"      dim_numbers: {{values: [{len(dim_values)}]}}", "      dim_values:"]
        lines += [f"        - values: [{', '.join(str(v) for v in vs)}]" for vs in dim_values]
    else:
        lines += ["      dim_numbers: {values: [1]}", f"      max_length: {max_length}"]
    return ([f"  # {comment}"] if comment else []) + lines


def _in_attr(name, dtype, values, comment=None):
    v = "[" + ", ".join(str(x) for x in values) + "]"
    lines = [f"  - name: {name}", "    type: attr", "    required: false",
             f"    dtypes: {{values: [{dtype}]}}", "    ranges:", "      valid:",
             f"        values: {v}"]
    return ([f"  # {comment}"] if comment else []) + lines


def _in_window():
    lines = ["  - name: window_size", "    type: attr_tuple", "    required: false",
             "    tuple_numbers:", "      values: [2]", "    dtypes:", "      values: [int]",
             "    ranges:", "      valid:", "        values: [-1, 64, 128]"]
    return ["  # window_size: (left, right) 二元组，-1 表示该侧不限制；两个元素独立取值",
            "# 只有候选和标杆都支持该入参的类别才声明（FIA 标杆没有）"] + lines


def inputs_for(iface, cat, paged):
    """按接口形态拼出 yaml 的 inputs 段。"""
    meta = IFACES[iface]
    can_window = cat != "single_benchmark_cann"
    blocks = []
    if meta["layout"] == "tnd":
        blocks.append(_in_tensor("q", _Q_TND, "[total_q, Hq, Dq]，TND 打包布局（本文件按单序列打包）"))
        if paged:
            blocks.append(_in_tensor("k", _K_PAGED, "paged KV: [num_blocks, block_size, Hk, Dk]"))
            blocks.append(_in_tensor("v", _V_PAGED, "paged KV: [num_blocks, block_size, Hk, Dv]"))
        else:
            blocks.append(_in_tensor("k", _K_TND, "k: [total_k, Hk, Dk]"))
            blocks.append(_in_tensor("v", _V_TND, "v: [total_k, Hk, Dv]"))
        blocks.append(_in_tensor("qv", _QV_TND, "qv: 吸收态专用 [total_q, Hq, 512]；prefill 用例会被钩子摘掉"))
        blocks.append(_in_index("cu_seqlens_q", "每个 sequence 的 q 累计长度（钩子写死成 [0, total_q]）"))
        if paged:
            blocks.append(_in_index("page_table", "page_table: [batch, num_blocks]（钩子写死成连续块）",
                                    dim_values=[[1], [2, 3, 8]]))
            blocks.append(_in_index("seqused_k", "每个 sequence 实际用到的 KV 长度（钩子写死）"))
        else:
            blocks.append(_in_index("cu_seqlens_k", "每个 sequence 的 kv 累计长度（钩子写死成 [0, total_k]）"))
        blocks.append(_in_attr("max_seqlen_q", "int", [1, 64, 198, 512], "最长的 q 序列（钩子改成 total_q）"))
        blocks.append(_in_attr("max_seqlen_k", "int", [64, 198, 256, 384, 512, 1025],
                               "最长的 kv 序列（钩子改成 total_k / num_blocks*BLOCK）"))
    elif meta["kvcache"]:
        blocks.append(_in_tensor("q", _Q_BSND, "q: [batch, seqlen_q, Hq, Dq]"))
        if paged:
            blocks.append(_in_tensor("k_cache", _K_PAGED, "paged KV: [num_blocks, block_size, Hk, Dk]"))
            blocks.append(_in_tensor("v_cache", _V_PAGED, "paged KV: [num_blocks, block_size, Hk, Dv]"))
        else:
            blocks.append(_in_tensor("k_cache", _K_BSND, "k_cache: [batch, kv_len, Hk, Dk]"))
            blocks.append(_in_tensor("v_cache", _V_BSND, "v_cache: [batch, kv_len, Hk, Dv]"))
        blocks.append(_in_tensor("qv", _QV_BSND, "qv: 吸收态专用 [batch, seqlen_q, Hq, 512]；prefill 用例会被钩子摘掉"))
        if paged:
            blocks.append(_in_index("page_table", "page_table: [batch, num_blocks]（钩子写死成连续块）",
                                    dim_values=[[1, 2], [2, 3, 8]]))
        blocks.append(_in_attr("cache_seqlens", "int", [198, 384, 1025],
                               "KV cache 有效长度（钩子改成实际 kv_len）"))
    else:
        blocks.append(_in_tensor("q", _Q_BSND, "q: [batch, seqlen_q, Hq, Dq]"))
        blocks.append(_in_tensor("k", _K_BSND, "k: [batch, kv_len, Hk, Dk]"))
        blocks.append(_in_tensor("v", _V_BSND, "v: [batch, kv_len, Hk, Dv]"))
        blocks.append(_in_tensor("qv", _QV_BSND, "qv: 吸收态专用；prefill 用例会被钩子摘掉"))
    blocks.append(_in_attr("softmax_scale", "float", [SCALE], "MLA 固定 192^-0.5"))
    blocks.append(_in_attr("causal", "bool", ["false", "true"], "causal / 非 causal 各一半"))
    if can_window:
        blocks.append(_in_window())
    lines = []
    for block in blocks:
        lines += block
    return lines


def build_yaml(iface, cat, paged=False):
    meta = IFACES[iface]
    cfg = CATEGORIES[cat]
    stem = stem_of(iface, cat, paged)
    api_type = f"{iface}_mla_{'mem' if cfg['kind'] == 'mem' else 'accuracy'}"
    paired = "" if cat != "single_benchmark_cann" else "（该类别不出 paged 变体：标杆 FIA 没有 page_table 入参）"
    # 形状注释按接口形态/是否 paged 变化，避免给测试同事看时出现无关内容
    if meta["layout"] == "bsnd":
        shapes = [
            "# 生成后每条用例的形状（钩子改写之后）：",
            "#   BSND prefill : q[B,Sq,32,192]  k[B,Sk,32,192]   v[B,Sk,32,128]",
            "#   BSND decode  : q[B,Sq,32,64]   qv[B,Sq,32,512]  k[B,Sk,1,64]  v[B,Sk,1,512]",
        ]
    else:
        shapes = [
            "# 生成后每条用例的形状（钩子改写之后；TND 按单序列打包）：",
            "#   TND prefill  : q[Tq,32,192]  k[Tk,32,192]  v[Tk,32,128]  cu_seqlens_x=[0,Tx]",
            "#   TND decode   : q[Tq,32,64]   qv[Tq,32,512] k[Tk,1,64]    v[Tk,1,512]",
        ]
    if meta["paged"]:
        shapes.append("#   paged        : KV 变 [num_blocks,128,H,D]，另加 page_table（页表写成连续块）")
    head = [
        f"# {stem}.yaml —— {meta['pkg']}.{meta['fn']} 的 ATK 用例设计文件",
        f"# 类别：{cat}（{cfg['desc']}）{paired}",
        "#",
        "# 这份 yaml 是本接口的**唯一真值**，用例不再手写，而是由 run 脚本现场生成：",
        f"#   atk case -f {stem}.yaml -p . -s <seed>",
        f"#     -> result/{stem}/json/all_{stem}.json",
        f"#     -> 拷成 cases/all_{stem}.json（交付里可直接打开查看）",
        "#",
        "# 两个必须知道的约定：",
        "#  1) ATK 会按下面的取值范围**逐维独立**随机每个输入，所以 k/v/qv/cache/索引张量",
        f"#     之间不可能自洽；跨输入的耦合全部由 generate_{iface}.py 里的钩子按 MLA 的",
        "#     数学契约统一改写（头数、末维、cache 与索引张量的内容/长度等）。",
        "#     因此这里只写「每一维的候选取值」，不描述输入之间的对应关系。",
        "#  2) 形态由 q 的最后一维决定：",
        "#       64  -> decode 吸收态（qv 生效，Hk=1，Dq=64，Dk=64，Dv=512）",
        "#       192 -> prefill（Dq=Dk=192=nope128+rope64，Dv=128，钩子会把 qv 摘掉）",
        "#",
    ] + shapes
    body = [
        f"name: {meta['pkg']}.{meta['fn']}",
        "api: pytorch",
        f"api_type: {api_type}",
        "version: v2.1",
        f"generate: {iface}_mla_constraints",
        "",
        "# 用例条数：本文件所有输入都只有 1 个 dtype，所以总条数 = dtype_numbers = 100",
        "dtype_numbers: 100",
        "# 关掉边界/空 tensor 用例：q.shape 为空时钩子无法判断 prefill/decode",
        "extra_numbers: 0",
        "outputs: 1",
        "",
        "# 判据：cv_fused_double_benchmark（混合容差 atol/rtol）。bf16 的 eps 是 2^-8，",
        "# 阈值 2.0 约等于「最大相对误差 <= 0.78%」；mem 只在 gpu_mem_compare 用得到。",
        "standard:",
        "  acc:",
        "    cv_fused_double_benchmark:",
        "      max_re_ratio: 2.0",
        "      avg_re_ratio: 1.2",
        "      root_mean_squared_ratio: 1.2",
        "  perf: not_key",
        "  mem: 1.1",
        "shape_distributions: [[0, 1.0]]",
        "",
        "# 输入声明。tensor 用 dim_values 给每一维的候选取值；attr 用 ranges.valid.values 给离散值。",
        "inputs:",
    ]
    return "\n".join(head + body + inputs_for(iface, cat, paged)) + "\n"


# --------------------------------------------------------------------------------------
# generate_*.py：ATK 跨输入约束钩子
# --------------------------------------------------------------------------------------
GENERATE = '''"""@@IFACE@@ (MLA) 的 ATK 跨输入约束钩子。

ATK 每条用例都会逐维独立随机各个输入，本钩子把它们改写成 MLA 的合法组合：

* prefill（q 末维 192）：Dq=Dk=192(nope128+rope64)，Dv=128，Hk=Hq=32，**摘掉 qv**
* decode （q 末维 64 ）：q 只留 rope64，qv=512，k 只留 rope64，v=512，Hk=1（吸收态）
* paged  ：KV 存成 (num_blocks, block_size, H, D)；page_table 固定成
           "第 b 个 batch 用第 b*nblk..b*nblk+nblk-1 页"；seqused_k / cache_seqlens
           取实际 KV 长度（= num_blocks * block_size）
* 索引张量（cu_seqlens_* / page_table / seqused_k）改成 int32_fixed 写死取值，
  否则 ATK 会随机出一组非法的（非单调的）累计长度。

注意：range_values 会被 ATK 拿去 hash 去重，嵌套结构必须用 tuple 而不是 list。
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY

BLOCK = 128   # paged KV 的 block_size，与交付用例保持一致
ROPE = 64     # MLA 的 rope 维度
NOPE = 128    # MLA 的 nope 维度
LATENT = 512  # 吸收态 v / qv 的维度
HEADS = 32    # q 的头数


def _entries(case_config):
    """按名字取输入；attr_tuple 在输入列表里是嵌套 list，这里按类型过滤掉。"""
    return {i.name: i for i in case_config.inputs if not isinstance(i, list)}


def _drop(case_config, *names):
    case_config.inputs = [i for i in case_config.inputs
                          if getattr(i, "name", None) not in names]


def _fixed(entry, values, shape=None):
    """把索引张量的取值写死：换成 ATK 的 *_fixed 数据类型。"""
    entry.dtype = "int32_fixed"
    if values and isinstance(values[0], (list, tuple)):
        entry.range_values = tuple(tuple(v) for v in values)
    else:
        entry.range_values = tuple(values)
    entry.shape = list(shape) if shape else [len(values)]


def _blocks(batch, nblk):
    """page_table：第 b 个 batch 占用全局页号 b*nblk .. b*nblk+nblk-1。"""
    return [tuple(b * nblk + i for i in range(nblk)) for b in range(batch)]


@GENERATOR_REGISTRY.register("@@IFACE@@_mla_constraints")
class @@CLS@@Constraints(CaseGenerator):
    """把 ATK 随机出来的用例改写成合法的 MLA 组合。"""

    def after_case_config(self, case_config):
        ins = _entries(case_config)
        q = ins["q"]
        decode = q.shape[-1] == ROPE        # 形态判据：q 末维
        bsnd = len(q.shape) == 4
        qv_entry = ins.get("qv") if decode else None
        if not decode:
            _drop(case_config, "qv")        # qv 只在吸收态出现

        # ---- q / qv ------------------------------------------------------------
        if decode:
            dq, hk, dk, dv = ROPE, 1, ROPE, LATENT
        else:
            dq, hk, dk, dv = NOPE + ROPE, HEADS, NOPE + ROPE, NOPE
        if bsnd:
            batch, seq_q = q.shape[0], q.shape[1]
            q.shape = [batch, seq_q, HEADS, dq]
        else:
            total_q = q.shape[0]
            q.shape = [total_q, HEADS, dq]
        if qv_entry is not None:
            qv_entry.shape = ([batch, seq_q, HEADS, LATENT] if bsnd
                              else [total_q, HEADS, LATENT])

        # ---- KV（普通 / paged）--------------------------------------------------
        # 非 paged：k 的 seq 维就是 KV 长度；paged：页数取自 page_table 的第 1 维。
        # 硬约束：KV 长度必须 >= Q 长度。causal / 滑窗掩码在 KV<Q（或窗口过窄）时会
        # 把尾部整行 mask 掉，而这些行在真实 MLA 场景并不存在；softmax 对全 -inf
        # 行的取值无定义，候选与参考必然对不上（实测 21/100 失败全部由此导致）。
        q_len = seq_q if bsnd else total_q
        k_entry = ins["k_cache"] if "k_cache" in ins else ins["k"]
        v_entry = ins["v_cache"] if "v_cache" in ins else ins["v"]
        paged = "page_table" in ins
        if paged:
            pt = ins["page_table"]
            nblk = max(pt.shape[1], -(-q_len // BLOCK))   # 向上取整，保证 nblk*BLOCK >= q_len
            kv_len = nblk * BLOCK
            batch_pt = batch if bsnd else 1        # TND 按单序列打包
            _fixed(pt, _blocks(batch_pt, nblk), shape=[batch_pt, nblk])
        elif bsnd:
            batch_pt = batch
            kv_len = max(k_entry.shape[1], q_len)
        else:
            batch_pt = 1
            kv_len = max(k_entry.shape[0], q_len)
        if paged:
            k_entry.shape = [batch_pt * nblk, BLOCK, hk, dk]
            v_entry.shape = [batch_pt * nblk, BLOCK, hk, dv]
        elif bsnd:
            k_entry.shape = [batch, kv_len, hk, dk]
            v_entry.shape = [batch, kv_len, hk, dv]
        else:
            k_entry.shape = [kv_len, hk, dk]
            v_entry.shape = [kv_len, hk, dv]

        # ---- 索引张量 / 长度 attr ----------------------------------------------
        if not bsnd:
            _fixed(ins["cu_seqlens_q"], [0, total_q], shape=[2])
            if "cu_seqlens_k" in ins:
                _fixed(ins["cu_seqlens_k"], [0, kv_len], shape=[2])
            if "max_seqlen_q" in ins:
                ins["max_seqlen_q"].range_values = total_q
            if "max_seqlen_k" in ins:
                ins["max_seqlen_k"].range_values = kv_len
        if "seqused_k" in ins:
            _fixed(ins["seqused_k"], [kv_len] * batch_pt, shape=[batch_pt])
        if "cache_seqlens" in ins:
            ins["cache_seqlens"].range_values = kv_len
        return case_config
'''


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def iface_files(iface, cat):
    meta = IFACES[iface]
    cfg = CATEGORIES[cat]
    kind = cfg["kind"]
    api_type = f"{iface}_mla_{'mem' if kind == 'mem' else 'accuracy'}"
    bm_name = "fia_mla" if cat == "single_benchmark_cann" else ""
    cls = "".join(p.capitalize() for p in iface.split("_")) + ("Mem" if kind == "mem" else "Api")
    return dict(api_type=api_type, cls=cls, bm_name=bm_name, meta=meta, kind=kind)


def main():
    for cat in CAT_ORDER:
        cfg = CATEGORIES[cat]
        for iface in IFACE_ORDER:
            f = iface_files(iface, cat)
            idir = ROOT / cat / "interfaces" / iface
            # 1) yaml（唯一真值）：每个变体一份（paged 变体只在候选支持 page_table 时出现）
            for stem, paged in variants(iface, cat):
                write(idir / f"{stem}.yaml", build_yaml(iface, cat, paged=paged))
            # 2) 跨输入约束钩子
            write(idir / f"generate_{iface}.py",
                  GENERATE.replace("@@IFACE@@", iface)
                          .replace("@@CLS@@", "".join(p.capitalize() for p in iface.split("_"))))
            # 3) CPU fp32 golden
            write(idir / f"benchmark_{iface}.py",
                  BENCHMARK.replace("@@IFACE@@", iface).replace("@@CAT@@", cat)
                          .replace("@@FN@@", f["meta"]["fn"]))
            # 4) ATK cpu/gpu/npu adapter
            npu_import = f"from {f['meta']['pkg']} import {f['meta']['fn']} as npu_fn"
            write(idir / f"execute_{iface}.py",
                  EXECUTE.replace("@@IFACE@@", iface).replace("@@CAT@@", cat)
                         .replace("@@FN@@", f["meta"]["fn"]).replace("@@NPU_IMPORT@@", npu_import)
                         .replace("@@API_TYPE@@", f["api_type"]).replace("@@CLS@@", f["cls"])
                         .replace("@@BM_NAME@@", f["bm_name"]).replace("@@GPU_KIND@@", f["meta"]["gpu"]))
            # 5) nodes + run 脚本
            write(idir / cfg["nodes"], NODES[cfg["nodes"]])
            write(idir / cfg["run"],
                  RUN.format(cat=cat, nodes=cfg["nodes"], bm=cfg["bm"],
                             tag="multi" if cfg["multi"] else "single",
                             task=cfg["task"], base=stem_of(iface, cat, False),
                             desc=cfg["desc"]))
            # 6) 用户场景 anchor：手写固定 JSON（不走 atk case），由 run 脚本串跑
            write(ROOT / cat / "cases" / f"user_scenarios_{stem_of(iface, cat, False)}.json",
                  json.dumps(build_user_scenarios(iface, cat), ensure_ascii=False, indent=1))
        # 类别级汇总脚本
        write(ROOT / cat / cfg["run"],
              RUN_CAT.format(cat=cat, desc=cfg["desc"], run=cfg["run"],
                             ifaces=" ".join(IFACE_ORDER)))
    print("generated under", ROOT)
    print("提示：cases/all_*.json 由 atk case 现场生成，跑 run_*.sh 即可刷新")


if __name__ == "__main__":
    main()
