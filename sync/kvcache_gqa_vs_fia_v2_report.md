# FA3 `flash_attn_with_kvcache` vs CANN 9.2 `aclnnFusedInferAttentionScoreV2 / V4` 性能对比报告

## 1. 结论

在 Ascend910B3、CANN 9.2.0、bf16、单 BSND GQA 场景下，**FA3 `flash_attn_with_kvcache` 的纯 device 计算内核耗时显著低于 FIA V2 和 V4**：

| 实现 | 计算内核耗时（us/call，两轮中位数） | 相对 FA3 |
|---|---:|---:|
| **FA3 `flash_attn_with_kvcache`** | **56.12** | 1.00x |
| **`aclnnFusedInferAttentionScoreV2`（直接 aclnn C API）** | **79.05** | FA3 快 **29.0%**（V2 是 FA3 的 **1.41x**） |
| **`aclnnFusedInferAttentionScoreV4`（直接 aclnn C API）** | **78.40** | FA3 快 **28.4%**（V4 是 FA3 的 **1.40x**） |
| 参考：`torch_npu.npu_fused_infer_attention_score_v2`（实际下发 V4） | 80.43 | FA3 快 30.2%（1.43x） |

- FA3 单次调用只启动 1 个计算内核 `SplitFuse::FAInfer`，两轮 56.96 / 55.28 us。
- FIA V2 单次调用只启动 1 个计算内核 `aclnnFusedInferAttentionScoreV2_*`，两轮 79.21 / 78.89 us。
- FIA V4 单次调用只启动 1 个计算内核 `aclnnFusedInferAttentionScoreV4_*`，两轮 78.81 / 77.99 us。
- **V4 比 V2 略快约 0.8%**（78.40 vs 79.05 us），两者性能基本同一档，都明显慢于 FA3。
- `torch_npu` 封装入口（同为 V4 内核）比直接 aclnn V4 慢约 2.6%（80.43 vs 78.40 us），差异来自封装侧下发的 tiling/参数组合，不影响“FA3 更快”的结论。
- 三个 FIA 路径的输出完全对齐：`max_abs_diff = 2.44e-4`、`rel_l2 ≈ 1.69e-3`（bf16），说明比较的是同一数学题。

## 2. 环境

| 项 | 值 |
|---|---|
| 设备 | Ascend910B3 |
| CANN | 9.2.0（`ASCEND_HOME_PATH=/home/w00811046/cann/cann-9.2.0-clean/cann-9.2.0`） |
| torch_npu | 2.9.0.post2 |
| conda 环境 | `cann` |
| dtype | bf16 |
| 执行模式 | eager（未开图模式） |
| 可见设备 | `ASCEND_RT_VISIBLE_DEVICES=0` |

运行命令：

```bash
source /home/w00811046/miniconda3/etc/profile.d/conda.sh && conda activate cann
source /home/w00811046/cann/cann-9.2.0-clean/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TMPDIR=/home/w00811046/tmp
cd /home/w00811046/code/fan/mlatest/flash-attention-npu
python docs/test/perf/bench_kvcache_gqa_vs_fia_v2.py
```

## 3. 形状与计算口径

| 张量 | 形状 | 说明 |
|---|---|---|
| Query | `[1, 293, 16, 128]` | BSND，B=1，Sq=293，Hq=16，D=128 |
| Key cache | `[1, 1024, 8, 128]` | BSND，Skv=1024，Hkv=8，D=128 |
| Value cache | `[1, 1024, 8, 128]` | 同上 |
| Mask | `[1, 16, 293, 1024]` 语义 | 右下角 causal：query 第 i 行可见 key `j <= i + (1024-293)` |

- GQA 映射：16 个 query head 共享 8 个 KV head（每组 2 个 query head）。
- `softmax_scale = 128 ** -0.5 = 0.08838834765`，所有实现显式传入同一个值。
- FA3 侧：`causal=True`，其文档明确为**右下角对齐**，与 `[1,16,293,1024]` 右下角 causal mask 等价。
- FIA 侧：`sparse_mode=3`（rightDownCausal）。按 CANN 约束，该模式只接收优化的 `(2048, 2048)` bool mask（上三角为 True 表示屏蔽），语义与上面的完整 mask 等价。

## 4. 测量方法

- 只统计 **device 计算内核**：`torch_npu.profiler`（Level0，Text 导出）的 `op_summary*.csv` 中 `Task Duration(us)`。
- `WARMUP=5`，采集 `STEPS=10`，取内核耗时中位数 × 每调用内核数 / 10，得到 `us/call`。
- 每个实现单独跑一次 profiler，串行执行，避免同卡并发干扰。
- 调度/metadata/fill 等非计算内核打印为 `[excluded]`，不计入结果。
- 复测一轮，两轮差异：FA3 3.0%，V2 0.4%，V4 1.1%，torch_npu 封装入口 2.5%，均在 ±5% 以内；报告采用两轮中位数。

## 5. 详细结果

### 5.1 逐内核明细（us/call）

| 实现 | 计算内核 | Run 1 | Run 2 | 中位数 |
|---|---|---:|---:|---:|
| FA3 | `SplitFuse::FAInfer<bf16, bf16, ... MaskType=1 ...>` | 56.96 | 55.28 | 56.12 |
| FIA V2 | `aclnnFusedInferAttentionScoreV2_FusedInferAttentionScore_FusedInferAttentionScore` | 79.21 | 78.89 | 79.05 |
| FIA V4 | `aclnnFusedInferAttentionScoreV4_FusedInferAttentionScore_FusedInferAttentionScore` | 78.81 | 77.99 | 78.40 |
| FIA V4（torch_npu 入口） | `aclnnFusedInferAttentionScoreV4_FusedInferAttentionScore_FusedInferAttentionScore` | 81.42 | 79.44 | 80.43 |

FA3 每调用还有 1 个被排除的 `aclnnInplaceFillScalar_FillAiCore_Fill`（约 1.00 us），属于 fill 类内核，不计入计算耗时；即使计入，FA3 约 57.1 us，仍明显快于 FIA。

### 5.2 汇总对比

| 指标 | Run 1 | Run 2 | 中位数 |
|---|---:|---:|---:|
| FA3（us/call） | 56.96 | 55.28 | 56.12 |
| FIA V2（us/call） | 79.21 | 78.89 | 79.05 |
| FIA V4（us/call） | 78.81 | 77.99 | 78.40 |
| FA3 / FIA V2 | 0.719x | 0.701x | **0.710x** |
| FA3 相对 FIA V2 节省 | 28.1% | 29.9% | **29.0%** |
| FIA V2 / FA3（耗时倍数） | 1.391x | 1.427x | **1.409x** |
| FA3 / FIA V4 | 0.723x | 0.709x | **0.716x** |
| FA3 相对 FIA V4 节省 | 27.7% | 29.1% | **28.4%** |
| FIA V4 / FA3（耗时倍数） | 1.384x | 1.411x | **1.397x** |
| V4 / V2 | 0.995x | 0.989x | **0.992x** |

### 5.3 正确性 sanity check（不计时）

| 实现 | max_abs_diff | rel_l2 |
|---|---:|---:|
| FIA V2（直接 aclnn） | 2.441406e-04 | 1.694e-03 |
| FIA V4（直接 aclnn） | 2.441406e-04 | 1.694e-03 |
| `torch_npu.npu_fused_infer_attention_score_v2`（V4） | 2.441406e-04 | 1.694e-03 |

以 FA3 输出为参考，三条 FIA 路径误差完全一致，符合 bf16 正常误差范围，说明 benchmark 的数学口径一致。

## 6. 关于 aclnn FIA 版本入口的说明

本机 torch_npu 2.9.0.post2 + CANN 9.2 下，公开 Python 入口的实际下发版本与名字并不一一对应：

| Python 入口 | 实际 device 内核 |
|---|---|
| `torch_npu.npu_fused_infer_attention_score_v2` | `aclnnFusedInferAttentionScoreV4_*` |
| `torch_npu.npu_fused_infer_attention_score`（旧入口） | `aclnnFusedInferAttentionScoreV3_*` |

因此，基准脚本通过 CANN aclnn C API 直接调用 V2 与 V4，两者共用同一套参数装配（V4 额外补 `query_rope`/`key_rope` 等可选参数与 `query_quant_mode`）：

```
aclnnFusedInferAttentionScoreV2GetWorkspaceSize(...) / aclnnFusedInferAttentionScoreV2(...)
aclnnFusedInferAttentionScoreV4GetWorkspaceSize(...) / aclnnFusedInferAttentionScoreV4(...)
```

同时保留 `torch_npu.npu_fused_infer_attention_score_v2`（V4）作为入口参考行，用于区分“API 名字”和“实际内核版本”。

## 7. 注意事项

1. **mask 约束**：FIA `sparse_mode=3` 只接受 `(2048, 2048)`（或 `(1,2048,2048)` / `(1,1,2048,2048)`）的优化 mask；`[1,16,293,1024]` 只作为语义等价参考，不能直接传给该模式。
2. **只统计 device 内核**：不含 host 下发、tiling、workspace 查询、调度/metadata 内核；端到端耗时可能不同。
3. **GQA 与 causal**：16/8 GQA、Sq=293、Skv=1024，FA3 的 `causal=True` 为右下角对齐，与 FIA `sparse_mode=3` 等价。
4. **测量噪声**：单次 profiler 结果存在约 ±3% 波动，本报告用两轮中位数；结论在该噪声范围内稳定。
5. **V2 与 V4 差异**：V4 比 V2 中位数快约 0.8%，在测量噪声附近，可视为同一性能档位。
6. **FA3 额外内核**：`aclnnInplaceFillScalar` 约 1.0 us/call 已被排除；若按“总 device 时间”口径统计，FA3 约 57.1 us，结论不变。
7. **可复现产物**：
   - 脚本：`docs/test/perf/bench_kvcache_gqa_vs_fia_v2.py`
   - 汇总 CSV：`docs/test/perf/perf_outputs/bench_kvcache_gqa_vs_fia_v2_summary.csv`
   - profiler trace：`docs/test/perf/perf_outputs/kvcache_gqa_bench_*`
