# docs/test 用例说明

本目录是 `flash-attention-npu` 在 Ascend 910 上 MLA（FA3 / FA4）的**测试与验收用例集合**，按用途分为四个子目录：

| 子目录 | 定位 | 判定依据 | 用例数 |
|---|---|---|---|
| [`atk/`](atk) | ATK 平台的精度 / 显存比对，4 个套件 | 精度阈值（`re_ratio`）、显存比 | **2640** |
| [`funcs/`](funcs) | pytest 功能正确性（MLA 专项）| 与 CPU golden 的 rel-L2 | **58** |
| [`consistency/`](consistency) | pytest 接口间**二进制一致性** | `torch.equal`（严格逐位） | **8** |
| [`perf/`](perf) | FA 与 CANN FIA 的**内核耗时**对比 | `candidate / FIA <= 1.05` | **14** 次测量 / **10** 条门限 |

```mermaid
flowchart LR
  subgraph ATK["atk/ — 平台化比对（ATK 框架）"]
    A1["single_benchmark_cpu<br/>npu 候选 vs CPU fp32"]
    A2["single_benchmark_cann<br/>npu 候选 vs CANN FIA"]
    A3["double_benchmark<br/>npu vs GPU + CPU 双标杆"]
    A4["gpu_mem_compare<br/>峰值显存对比"]
  end
  subgraph PY["pytest 直跑"]
    B1["funcs/<br/>功能正确性"]
    B2["consistency/<br/>接口间二进制一致"]
    B3["perf/<br/>内核耗时对比"]
  end
  SRC["atk/tools/gen_atk_mla.py<br/>（生成整棵 atk 树）"] --> ATK
```

---

## 1. 环境准备

所有用例都跑在真实 Ascend 910（910B3）+ CANN 9.2 上，`torch_npu` 必需。

```bash
# 1) CANN 运行环境（torch_npu 需要 libhccl.so 等）
source /home/w00811046/cann/cann-9.2.0-clean/ascend-toolkit/set_env.sh

# 2) conda 环境（atk 用例需该环境已装 atk + atk_private_precision）
conda activate cann

# 3) 通用环境变量
export ASCEND_RT_VISIBLE_DEVICES=0
export LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/11:/usr/lib/aarch64-linux-gnu
export TMPDIR=/home/w00811046/tmp
```

- `perf/` 脚本强制要求 CANN 前缀为 `9.2`，否则直接 abort。
- `funcs/`、`consistency/`、`perf/` 的模块顶层 import `flash_attn_npu_3/4`，必须在**仓库根目录**执行（`fast-attention-npu/`）。
- `atk` 脚本支持 `CANN_ROOT` / `ATK_CONDA_ENV` / `ATK_SEED` / `ATK_BIN` 覆盖默认值。

---

## 2. atk/ — ATK 平台用例（2640 条）

### 2.1 四个套件与对比基准

| 套件 | 对比基准 | `nodes_*.yaml` | 执行参数 |
|---|---|---|---|
| `single_benchmark_cpu` | CPU fp32 标杆 | `nodes_accuracy.yaml` | `--task accuracy --bm_device cpu --single_process` |
| `single_benchmark_cann` | CANN `npu_fused_infer_attention_score` | `nodes_accuracy_cann.yaml` | `--task accuracy --bm_device npu --single_process` |
| `double_benchmark` | GPU(daoailab) + CPU 双标杆 | `nodes_accuracy_gpu.yaml` | `--task accuracy --bm_device cpu`（多进程） |
| `gpu_mem_compare` | GPU 峰值显存 | `nodes_memory_gpu.yaml` | `--task memory_device --bm_device gpu`（多进程） |

### 2.2 接口与 paged 变体

每个套件默认覆盖 5 个接口：`fa3_func` / `fa3_varlen` / `fa3_kvcache` / `fa4_func` / `fa4_varlen`。

- `*_paged.yaml` 分页变体只出现在 **`fa3_kvcache`** 与 **`fa4_varlen`**（其 `paged=True`）。
- `single_benchmark_cann` **不含** paged 变体（CANN FIA 无 `page_table` 入参）。
- 因此 yaml 数量：`single_benchmark_cann` = 5，其余三个套件各 = 7。
- `gpu_mem_compare` 的文件名带 `_mem` 后缀（如 `fa3_func_mem.yaml`）。

### 2.3 用例范围与组合方式

**yaml 是唯一真值**：所有用例由 `atk case` 从 yaml 生成，不再手写，便于给测试人员逐条讲解。

yaml 结构（以 `single_benchmark_cann/interfaces/fa3_func/fa3_func.yaml` 为例）：

```yaml
name: flash_attn_npu_3.flash_attn_func   # 被测接口
api: pytorch
api_type: fa3_func_mla_accuracy
version: v2.1
generate: fa3_func_mla_constraints       # 用例生成约束钩子
dtype_numbers: 100                       # ← 生成 100 条用例
outputs: 1
standard:
  acc:
    cv_fused_double_benchmark:           # 精度判定阈值
      max_re_ratio: 2.0
      avg_re_ratio: 1.2
      root_mean_squared_ratio: 1.2
  perf: not_key
  mem: 1.1
shape_distributions: [[0, 1.0]]
inputs:
  - name: q
    type: tensor
    required: true
    dtypes: {values: [bf16]}
    shapes:
      dim_numbers: {values: [4]}
      dim_values:                        # ← 每一维的候选取值池
        - values: [1, 2]
        - values: [1, 64, 198, 512, 1025]
        - values: [32]
        - values: [64, 192, 512]
```

组合维度：

| 维度 | 取值 | 说明 |
|---|---|---|
| dtype | `bf16` | 当前 MLA 用例只跑 bf16 |
| 张量形状 | `dim_values` 逐维候选池 | ATK **逐维独立随机**取样（非笛卡尔积） |
| 数值范围 | `ranges.valid.values` = `[-5.0, 5.0]` | 随机数分布区间 |
| 属性 | `causal`、`softmax_scale`、`window_size`、`cache_seqlens`、`max_seqlen_q/k` | attr / attr_tuple |
| 索引张量 | `cu_seqlens_q/k`、`page_table`、`seqused_k`（`int32`，`max_length: 4096`） | 真实取值由钩子写死 |
| 用例条数 | `dtype_numbers: 100` | 本用例所有输入只有 1 个 dtype ⇒ **总条数 = 100** |

**跨输入的 MLA 契约**由 `interfaces/<iface>/generate_<iface>.py` 里的 `after_case_config` 钩子统一改写（这是保证用例合法性的关键）：

- prefill：q/k 末维 192，**摘除 qv**；
- decode：q 末维 64 + `qv` 末维 512，`Hk = 1`；
- paged：`page_table` 写连续块，`seqused_k` 与真实 KV 长度对齐；
- 索引张量统一改为 `int32_fixed`。

### 2.4 用户场景 anchor（与常规用例分开管理）

除 100 条生成用例外，每个接口额外有 **2 条手写用户场景**用例
（`cases/user_scenarios_<base>.json`，报告单独产出）：

| memo | 形状 | 说明 |
|---|---|---|
| `user_prefill_198` | q/k `[1,198,32,192]`、v `[1,198,32,128]`、`causal=True` | 训练 / 未吸收 prefill |
| `user_decode_1025` | q `[1,1,32,64]`、k `[1,1025,1,64]`、v `[1,1025,1,512]`、qv `[1,1,32,512]`、`causal=False` | 推理 / 吸收 decode |

### 2.5 执行步骤

整棵树由生成器产出（仅在需要重建模板时执行）：

```bash
cd /home/w00811046/code/fan/mlatest/flash-attention-npu
python3 docs/test/atk/tools/gen_atk_mla.py
```

**套件级**（一次跑完该套件全部接口）：

```bash
cd docs/test/atk
CANN_ROOT=/home/w00811046/cann/cann-9.2.0-clean/ascend-toolkit \
  bash single_benchmark_cann/run_accuracy.sh          # 其余套件换成 double_benchmark / single_benchmark_cpu
CANN_ROOT=/home/w00811046/cann/cann-9.2.0-clean/ascend-toolkit \
  bash gpu_mem_compare/run_memory.sh
```

**接口级**（调试单个接口）：`cd docs/test/atk/<suite>/interfaces/<iface> && bash run_accuracy.sh`（显存套件为 `run_memory.sh`）。

每个接口脚本内部按固定 4 步执行：

```mermaid
flowchart LR
  S1["1. atk case -f &lt;stem&gt;.yaml -p . -s 20261113<br/>→ result/&lt;stem&gt;/json/all_&lt;stem&gt;.json"] --> S2["2. cp 到 &lt;suite&gt;/cases/all_&lt;stem&gt;.json<br/>（交付可直接查看）"]
  S2 --> S3["3. atk task -c cases/all_&lt;stem&gt;.json -n &lt;nodes&gt;.yaml<br/>--task accuracy --bm_device npu --single_process"]
  S3 --> S4["4. atk task -c cases/user_scenarios_*.json<br/>（用户场景 anchor，报告分开）"]
```

对应命令（`single_benchmark_cann/interfaces/fa3_func/run_accuracy.sh`）：

```bash
atk case -f fa3_func.yaml -p . -s 20261113 --log warning
cp result/fa3_func/json/all_fa3_func.json ../../../single_benchmark_cann/cases/all_fa3_func.json
atk task -c ../../../single_benchmark_cann/cases/all_fa3_func.json \
  -n nodes_accuracy_cann.yaml -p . \
  --task accuracy --bm_device npu --single_process \
  --log warning --save_data input --save_data output
atk task -c ../../../single_benchmark_cann/cases/user_scenarios_fa3_func.json \
  -n nodes_accuracy_cann.yaml -p . --task accuracy --bm_device npu --single_process \
  --log warning --save_data input --save_data output
```

产出位置：`interfaces/<iface>/result/<stem>/`（csv / excel / json）与 `interfaces/<iface>/atk_output/<case>_<ts>/report/*.xlsx`。

### 2.6 用例数量

| 套件 | `all_*.json` 条数 | `user_scenarios_*.json` 条数 | 小计 |
|---|---|---|---|
| `single_benchmark_cann` | 5 × 100 = 500 | 5 × 2 = 10 | **510** |
| `single_benchmark_cpu` | 7 × 100 = 700 | 5 × 2 = 10 | **710** |
| `double_benchmark` | 7 × 100 = 700 | 5 × 2 = 10 | **710** |
| `gpu_mem_compare` | 7 × 100 = 700 | 5 × 2 = 10 | **710** |
| **合计** | | | **2640** |

---

## 3. funcs/ — 功能正确性用例（58 条）

针对 MLA 的**数值正确性**，用本地 golden 比对（rel-L2 / 容差），覆盖 prefill（未吸收）、decode（吸收）、GQA、paged、BNSD/BSND/TND 等各种组合。

```bash
cd /home/w00811046/code/fan/mlatest/flash-attention-npu
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest \
  docs/test/funcs/test_flash_attn_npu_v3_mla.py \
  docs/test/funcs/test_flash_attn_npu_v4_mla.py \
  docs/test/funcs/test_flash_attn_npu_mla_consistency.py -q
```

| 文件 | 覆盖 | 收集用例数 |
|---|---|---|
| `test_flash_attn_npu_v3_mla.py` | FA3 全部接口 | 25（15 个 `def test_*`；其中 2 个 `@parametrize batch[1,2] × kv_len[129,513,1025]` 各展开 6） |
| `test_flash_attn_npu_v4_mla.py` | FA4 全部接口 | 17（12 个 `def test_*`；1 个 parametrize 展开 6） |
| `test_flash_attn_npu_mla_consistency.py` | 训练(prefill) ↔ 推理(decode) 一致性、以及参数校验 | 16（9 个 `def test_*`；7 个 `@parametrize backend=[flash_attn_npu_3, flash_attn_npu_4]` 各展开 2，另有 2 个拒绝非法入参的用例） |

覆盖范围（用例名摘要）：`prefill_bsnd_func` / `prefill_tnd_varlen` / `prefill_bnsd` / `prefill_gqa` / `decode_bsnd_no_paged` / `decode_bsnd_paged` / `decode_bnsd_cross_batch` / `decode_bnsd_sq1_cross_batch` / `decode_kv_multicore` / `decode_kv_single_core` / `decode_tnd_varlen` / `decode_paged_varlen` / `decode_rejects_append_kv` / `rejects_gather_kv_indices` / `rejects_invalid_layout`。

> `docs/test/` 下没有 `conftest.py` / `pytest.ini`，仓库根的 `tests/conftest.py`（提供 `--random-sample`、`CI_RANDOM_SEED`）不会自动生效，直接 `pytest <文件>` 即可。

---

## 4. consistency/ — 接口间二进制一致性用例（8 条）

验证**同一版本内各公开接口对等价输入返回逐位相同（`torch.equal`）的结果**，覆盖 prefill 与 decode 两种用户场景（含 `heads_q = 128` 变体）。

```bash
cd /home/w00811046/code/fan/mlatest/flash-attention-npu
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest \
  docs/test/consistency/test_interface_consistency_mla.py -q
```

| 用例 | 比较对象 |
|---|---|
| `test_fa3_three_interfaces_bitwise_identical` | `flash_attn_func` vs `flash_attn_varlen_func` vs `flash_attn_with_kvcache` |
| `test_fa4_two_interfaces_bitwise_identical` | `flash_attn_func` vs `flash_attn_varlen_func` |

场景组合（`FORMS`，`@parametrize` 展开 4 × 2 = **8** 条）：

| form | (Sq, Skv, Hq, Hkv, Dq, Dv, causal) |
|---|---|
| `prefill` | (198, 198, 32, 32, 192, 128, True) |
| `prefill_qh128` | (198, 198, **128**, 32, 192, 128, True) |
| `decode` | (1, 1025, 32, 1, 64, 512, False) |
| `decode_qh128` | (1, 1025, **128**, 1, 64, 512, False) |

保证逐位一致的关键开关（用例内已固定）：

- `num_splits=1` —— 关闭 FlashDecode 的 split-KV，使 KV 方向的每行归约顺序在各入口间一致；
- KV 非分页 —— 避免 page 间接打乱处理顺序；
- 各接口跑**同一数学形式**（BSND dense / TND varlen / cache 只是入口布局不同）；
- `softmax_scale` **不显式传**，由接口默认值 `(q.last + qv.last) ** -0.5` 决定，各接口公式相同。

> `decode_qh128` 因 `heads_q = 128 > 32` 不再命中 GEMV 闸门，实际走 FAInfer 主路径。

---

## 5. perf/ — 性能对比用例（14 次测量 / 10 条门限）

对比 FA3/FA4 与 CANN `npu_fused_infer_attention_score`（FIA）的**设备计算内核耗时**。

```bash
cd /home/w00811046/code/fan/mlatest/flash-attention-npu
python docs/test/perf/bench_mla_vs_fia.py          # 依赖 CANN 9.2，否则 abort
python docs/test/perf/mla_user_scenarios.py        # 可选：仅打印两个场景摘要
```

测量口径：

- 只统计**计算内核**（FA 白名单 `FAInfer`、`MlaDecodeGemv`；FIA 白名单 `FusedInferAttentionScore`），调度/清洗类内核（`ComputeFAMetadata*`、`Fill`）过滤后单独打印；
- `WARMUP = 5`、`STEPS = 10`，取 `torch_npu.profiler` `op_summary` 的 `Task Duration` 中位数 × 每调用内核数；
- 门限 `KERNEL_GATE = 1.05`：每条候选行 `candidate / 同布局 FIA <= 1.05`，任一超限则退出码非 0。

组合情况：

| 组 | FIA 基线 | 候选实现 |
|---|---|---|
| `MLA-BSND` | `fia_mla_bsnd` | `fa3_func_bsnd`、`fa3_kvcache_bsnd`、`fa4_func_bsnd` |
| `MLA-TND` | `fia_mla_tnd` | `fa3_varlen_tnd`、`fa4_varlen_tnd` |

`main()` 对 **2 个场景（prefill / decode）× 7 个实现（2 基线 + 5 候选）** 各测一次 = **14 次内核测量**，其中 **10 条**候选行参与门限判定。

场景定义（`mla_user_scenarios.py`，源自 `usershape.txt`）：

| 场景 | 形态 | 形状 |
|---|---|---|
| `prefill` | 训练 / 未吸收 | q/k head_dim 192、v head_dim 128、198 token、32 heads、causal |
| `decode` | 推理 / 吸收 | q rope 64 + nope 512、v 512、KV 1025、`heads_kv = 1` |

产出：`docs/test/perf/perf_outputs/kernel_bench_<scenario>_<impl>_<时间戳>/`，汇总 CSV 为 `docs/test/perf/perf_outputs/bench_kernel_summary.csv`。

---

## 6. 用例数量汇总

| 分类 | 用例数 | 备注 |
|---|---|---|
| `atk/`（4 套件） | 2640 | 其中 2600 条为生成的常规用例，40 条为用户场景 anchor |
| `funcs/`（3 文件） | 58 | pytest 静态统计 |
| `consistency/`（1 文件） | 8 | 逐位一致，最严格 |
| `perf/`（1 脚本） | 14 次测量 / 10 条门限 | 内核耗时 |
| **合计** | **2720** | 不含 perf 的测量次数 |

---

## 7. 注意事项与已知差异

1. **历史文档里的路径已过期**：早期设计文档写的是 `tests/test_flash_attn_npu_*`、`tests/perf/...`，实际文件已迁到 `docs/test/` 下，请以本说明为准。
2. **yaml 与生成器的版本差异**：已入库的 `fa3_func.yaml` 与当前 `gen_atk_mla.py` 产出的模板不完全一致（缺 `extra_numbers: 0`，且多出 `k_cache`/`v_cache`/索引输入），重建前请先确认。
3. **`softmax_scale` 口径不一致**：`perf/mla_user_scenarios.py` 中的 `DEEPSEEK_SOFTMAX_SCALE = 0.14467962580268923` 与其注释「`192 ** -0.5`」对不上（`192 ** -0.5 ≈ 0.0721688`）；ATK 的用户场景用例（`user_scenarios_*.json`）里用的是 `0.07216878364870322`。建议核对 `usershape.txt` 后统一。
4. **用例数量口径**：`atk` 的用例数来自 `cases/*.json`（`atk case` 的实际产出），`funcs`/`consistency` 来自 pytest 静态统计（`def test_*` × `parametrize` 展开），二者定义不同但都可用 `--collect-only` 复核。
