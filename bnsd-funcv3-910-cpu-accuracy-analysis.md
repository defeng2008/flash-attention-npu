# BNSD `flash_attn_func` v3（910）CPU 精度问题分析

## 1. 结论摘要

- 失败**不是 910 kernel 的问题**，而是 ATK 精度插件的 **CPU 标杆 causal 掩码实现有缺陷**。
- 根因：`benchmark_flash_attn_func_v3.py` 用「加 -1e4」实现 causal 掩码。测试 yaml 的 `outlier_values: [0.001, 1000]` 会把部分元素放大 1000 倍，q·k·scale 可达 1e5\~1e6，`-1e4` 无法再压制被掩码位置，CPU 标杆整体偏掉。
- 修复：掩码改为 `masked_fill(..., float("-inf"))`（单文件、6 行）。
- 验证：
  - 三方对比（npu / cpu 标杆 / fp64）：修复前 npu 与 fp64 一致、cpu 偏离；修复后 **cpu 与 npu 逐元素完全一致（0 差异）**。
  - ATK 受影响子集 100 条：96 通过；剩余 4 条为「参考值接近 0 时 max 相对误差失效」的度量边界，不是 kernel 问题。

## 2. 环境

| 项     | 值                                                                                          |
| ----- | ------------------------------------------------------------------------------------------ |
| 设备    | Ascend 910B3（card 6）                                                                       |
| CANN  | `/home/w00811046/cann/cann-9.2.0-clean`                                                    |
| conda | `cann`                                                                                     |
| 扩展构建  | `FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python setup.py build_ext --inplace` |
| 运行期   | 需 `PYTHONPATH=/home/w00811046/code/fan/bnsdissue/flash-attention-npu`                      |

## 3. 失败特征（cc.csv）

- 共 **33** 条失败 shape。
- **全部 `causal=True`**。
- 每条用例 **q 或 k 至少一个是 `nd` 分布**（mean ∈ \[-100,100]，std ∈ \[1,25]）；只有 v 为 nd 的用例不失败。
- head\_dim 集中在 8/16/32/64，seqlen 含 9、99 等非对齐值，包含 GQA（num\_heads ≠ num\_heads\_k）。

## 4. 已排除项：qv / descales

- `func_v3.yaml` 中 `qv / q_descale / k_descale / v_descale` 的 `range_values` 都是 `default`，ATK 实际生成的数据是 **`None`**（已从落盘 `input.bin` 验证）。
- 因此既不触发 910 host 的 `TORCH_CHECK(!q_descale_.has_value(), ...)`，也不进入 CPU 标杆的 `NotImplementedError` 分支，**与本次失败无关**。

## 5. 根因与证据

### 5.1 旧实现（问题代码）

`atktest/accuracy/plugins/func_v3/benchmark_flash_attn_func_v3.py`

```python
if mask is not None:
    qk_result += mask[:query.shape[1], kv_start:kv_start + sub_key.shape[2]].to(interm_dtype) * -1e4
```

### 5.2 触发链

1. ATK 对 `nd` 输入执行 `generate_outlier_data`：0.1% 元素放大 1000 倍，并 clamp 到 dtype 上限。
2. nd 本身均值可达 ±100、std 25；放大后 fp16 可达 65504、bf16 可达 1e5 量级。
3. `q·k·softmax_scale`（D=8\~128）→ logit 可达 1e5\~1e6。
4. 被 causal 掩码的位置只加了 `-1e4`，仍是 1e5 量级 → 未被压制，进入 softmax → CPU 标杆结果整体错误。
5. 非 causal 没有掩码，所以不失败。

### 5.3 三方对比（npu vs cpu 标杆 vs fp64 参考）

同分布随机数据（含 `outlier_values: [0.001, 1000]` 与1个放大元素）：

| 场景                          | 旧标杆 cpu vs fp64    | npu vs fp64 | npu vs cpu         |
| --------------------------- | ------------------ | ----------- | ------------------ |
| fp16 causal，无 outlier       | 1.95e-3            | 1.95e-3     | **0**              |
| fp16 causal，+outlier        | **4.96（rel 3.20）** | 1.91e-3     | **4.96（rel 47.6）** |
| bf16 causal，+outlier        | **4.95（rel 3.21）** | 1.52e-2     | **4.95（rel 47.4）** |
| bf16 non-causal，+outlier    | 4.19               | 4.19        | **0**              |
| fp16 causal GQA D8，+outlier | **4.80**           | 1.94e-3     | **4.80**           |
| bf16 causal GQA D8，+outlier | **80.6**           | 1.79e-2     | **80.6**           |

修复后，上表所有 `npu vs cpu` 场景均为 **0.00e+00**，且 cpu/npu 与 fp64 保持一致。

## 6. 修复

文件：`atktest/accuracy/plugins/func_v3/benchmark_flash_attn_func_v3.py`

```python
if mask is not None:
    # Mask with -inf instead of adding a fixed -1e4: the outlier ranges in
    # func_v3.yaml (0.1% of elements scaled by 1000) push qk logits far past
    # 1e4, so an additive constant no longer suppresses the masked entries.
    qk_result = qk_result.masked_fill(
        mask[:query.shape[1], kv_start:kv_start + sub_key.shape[2]], float("-inf")
    )
```

说明：

- 此处 `qk_result` 已 cast 到 fp32（`interm_dtype=torch.float32`），`-inf` 是精确掩码。
- 小 logit 场景下 `exp(logit - 1e4)` 本就下溢为 0，与 `-inf` 等价，因此不影响仓库默认用例。
- causal 且 `seqlen_q <= seqlen_k` 时每行至少有一个可见 key，不会出现整行 `-inf`。

## 7. ATK 验证结果

- 子集：80 条「causal + q/k 为 nd」+ 20 条 causal 对照（单 tensor ≤ 200k 元素）。
- 结果：**96 通过 / 4 精度失败**。

### 残留 4 条（均 fp16、大 logit、参考值接近 0）

| case | shape                        | 现象                                                           | 判断             |
| ---- | ---------------------------- | ------------------------------------------------------------ | -------------- |
| 351  | q(1,99,8,16)，q/k=nd，v=±0.001 | npu 与 cpu **逐位相同**，参考值 \~9.5e-6                              | max 相对误差在近零处失效 |
| 434  | q(8,9,2,128)                 | npu 误差小于 cpu 标杆                                              | 非 kernel 问题    |
| 524  | q(4,16,4,128)                | npu max/avg 优于 cpu，rms 比 1.37（阈值 1.2）                        | 数值极限           |
| 567  | q(16,9,2,128)                | 单个元素参考值 \~-4.7e-4，两者绝对误差均 \~5e-3；avg/rms 比 0.145/0.42，npu 更好 | 抵消点的相对误差失真     |

这 4 条均不在 `cc.csv` 的失败形态中，属于精度度量的边界情况。

## 8. 两套 yaml 的差异

| 维度              | 仓库默认 `cases/gen/func_v3.yaml` | 测试人员 `func_v3.yaml`                                                             |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------- |
| `dtype_numbers` | 8（×2 dtype = 16 条）            | 700（×2 dtype = 1400 条）                                                          |
| B               | `[1,2]`                       | `[1,2,4,8,9,16,32,64,99,128,256]`                                               |
| S               | `[2,8,64]`                    | `[2,8,9,16,32,64,99,128,256]`                                                   |
| H / D           | `[2,4,8]` / `[16,32,64,128]`  | 同左，D 增加 `8`                                                                     |
| `max_length`    | 8 MB                          | 32 MB                                                                           |
| 数据分布            | 仅均匀 `[-5,5]`                  | `nd` 正态（mean ±100、std 1~25）+ 均匀 `[-0.001,0.001]`/`[-5,5]` + `custom_dist_ratio` |
| s               | 无                             | `[0.001, 1000]`                                                                 |
| `boundary`      | 无                             | 显式开 `has_infnan / has_lower_border` 等                                           |
| `perf`          | `not_key`                     | `[0.95,0.95]`                                                                   |

相同点：接口、输入清单、attr 取值、精度标准（max 2.0 / avg 1.2 / rms 1.2）完全一致。
因此仓库默认 16 条不会复现该问题，测试人员 1400 条才会暴露。

## 9. 复现与回归

```bash
cd /home/w00811046/code/fan/bnsdissue/flash-attention-npu/atktest/accuracy
source /home/w00811046/miniconda3/etc/profile.d/conda.sh && conda activate cann
export PYTHONPATH=/home/w00811046/code/fan/bnsdissue/flash-attention-npu:$PYTHONPATH
export CANN_SET_ENV=/home/w00811046/cann/cann-9.2.0-clean/ascend-toolkit/set_env.sh
bash run_accuracy.sh --device 910 --interface func_v3 --mode single --card 6
```

注意：

- `run_accuracy.sh` 为 CRLF 换行，直接执行会报 `$'\r'`；用同目录 LF 副本运行（脚本按自身路径推导 `ACCURACY_DIR`，不能用 `/dev/fd` 方式）：
  `tr -d '\r' < run_accuracy.sh > .run_accuracy_lf.sh && bash .run_accuracy_lf.sh --device 910 --interface func_v3 --mode single --card 6`
- 用例从 `cases/source/func_v3.json` 过滤生成；需要先按目标 yaml 重新生成 source。
- 机器负载高时单用例可能 20\~60s，1400 条全量耗时较长。

## 10. 后续建议

1. 同样的 `-1e4` 写法还存在于 `func_v2`、`func_v4`、`kvcache_v2`、`kvcache_v3` 的 benchmark 以及 `tests/common/attention_ref.py`；如这些测试也会用到 nd/outlier 数据，建议同步改为 `-inf`。
2. 残留 4 条建议与测试方确认度量的近零处理（是否排除 `|ref|` 很小的元素），目前看不是 kernel 精度缺陷。
3. 全量回归建议在机器空闲时执行，并保留修复前后的 `reports/910/func_v3/single/*/report/*.xlsx` 作为对照。
