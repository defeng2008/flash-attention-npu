# FA3 `flash_attn_with_kvcache` vs CANN 9.2 `aclnnFusedInferAttentionScoreV2 / V4` 性能对比报告

## 1. 结论

在 Ascend910B3、CANN 9.2.0、bf16、BSND GQA、纯 device 计算内核口径下，**FA3 与 FIA 的相对性能随 batch 明显变化**：

**Batch = 1：FA3 更快**

| 实现 | 计算内核耗时（us/call，两轮中位数） | 相对 FA3 |
|---|---:|---:|
| **FA3 `flash_attn_with_kvcache`** | **56.12** | 1.00x |
| **`aclnnFusedInferAttentionScoreV2`（直接 aclnn C API）** | **79.05** | FA3 快 **29.0%**（V2 是 FA3 的 **1.41x**） |
| **`aclnnFusedInferAttentionScoreV4`（直接 aclnn C API）** | **78.40** | FA3 快 **28.4%**（V4 是 FA3 的 **1.40x**） |
| 参考：`torch_npu.npu_fused_infer_attention_score_v2`（实际下发 V4） | 80.43 | FA3 快 30.2%（1.43x） |

**Batch = 32：FIA V2/V4 更快**

| 实现 | 计算内核耗时（us/call，两轮中位数） | 相对 FA3 |
|---|---:|---:|
| **FA3 `flash_attn_with_kvcache`** | **950.67** | 1.00x |
| **`aclnnFusedInferAttentionScoreV2`（直接 aclnn C API）** | **810.89** | 比 FA3 **快 14.7%**（FA3 是 V2 的 1.17x） |
| **`aclnnFusedInferAttentionScoreV4`（直接 aclnn C API）** | **811.15** | 比 FA3 **快 14.7%**（FA3 是 V4 的 1.17x） |
| 参考：`torch_npu.npu_fused_infer_attention_score_v2`（实际下发 V4） | 811.76 | 比 FA3 快 14.6% |

关键观察：

- **batch 扩展性不同**：B=1 → B=32 时，FA3 从 56.12 us 增至 950.67 us（**16.9x**），FIA 从约 78.4 us 增至约 811 us（**10.3x**）。FA3 在 B=32 下明显没有随 batch 线性扩展，导致排名反转。
- **B=32 时 V2 与 V4 基本持平**（810.89 vs 811.15 us，差异 <0.1%）。
- 两个 batch 下 `torch_npu` 封装入口（V4 内核）与直接 aclnn V4 均只差约 1–3%，属于同一档。
- 所有 FIA 路径输出与 FA3 完全对齐（B=1 `rel_l2≈1.69e-3`，B=32 `rel_l2≈2.63e-4`），说明比较的是同一数学题。
- 两个 batch 的两轮测量差异均 ≤3%，结论稳定。
- V4 的深度剖析见第 9 节：kernel 已接近自身调度上限（20/20 核、cube 95% 占用、非访存瓶颈），API 参数层面空间约 1%，主要余量在 Sq 128 对齐/序列打包（10%–15%）。

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
- 复测一轮：B=1 下两轮差异 FA3 3.0%、V2 0.4%、V4 1.1%、torch_npu 封装入口 2.5%；B=32 下差异均 ≤1.7%；两层结论都在 ±5% 噪声内。

## 5. Batch = 1 详细结果

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

1. **mask 约束**：FIA `sparse_mode=3` 只接受 `(2048, 2048)`（或 `(1,2048,2048)` / `(1,1,2048,2048)`）的优化 mask；`[1,16,293,1024]`（B=32 时为 `[32,16,293,1024]`）只作为语义等价参考，不能直接传给该模式。
2. **只统计 device 内核**：不含 host 下发、tiling、workspace 查询、调度/metadata 内核；端到端耗时可能不同。
3. **GQA 与 causal**：16/8 GQA、Sq=293、Skv=1024，FA3 的 `causal=True` 为右下角对齐，与 FIA `sparse_mode=3` 等价。
4. **测量噪声**：单次 profiler 结果存在约 ±3% 波动，本报告用两轮中位数；结论在该噪声范围内稳定。
5. **V2 与 V4 差异**：两个 batch 下 V4 与 V2 中位数差异都 <1%，可视为同一性能档位。
6. **FA3 额外内核**：`aclnnInplaceFillScalar`（B=1 约 1.0 us，B=32 约 1.2 us）已被排除；若按“总 device 时间”口径统计，结论不变。
7. **可复现产物**：
   - 脚本：`docs/test/perf/bench_kvcache_gqa_vs_fia_v2.py`（`--batch` 控制 batch）
   - 深采驱动：`docs/test/perf/profile_fia_v4_metrics.py`（单实现，供 msprof 包裹）
   - B=1 汇总 CSV：`docs/test/perf/perf_outputs/bench_kvcache_gqa_vs_fia_v2_b1_summary.csv`
   - B=32 汇总 CSV：`docs/test/perf/perf_outputs/bench_kvcache_gqa_vs_fia_v2_b32_summary.csv`
   - profiler trace：`docs/test/perf/perf_outputs/kvcache_gqa_bench_b{1,32}_*`
   - msprof 深采原始数据：`docs/test/perf/perf_outputs/msprof_v4_b32_*`、`msprof_fa3_b32_*`、`msprof_v4_b1_*`、`msprof_v4_b32_sq*`

## 8. Batch = 32 详细结果

命令：

```bash
python docs/test/perf/bench_kvcache_gqa_vs_fia_v2.py --batch 32
```

形状：`q=[32,293,16,128]`、`k/v=[32,1024,8,128]`，mask 语义 `[32,16,293,1024]` 右下角 causal；其余口径与 B=1 完全一致。

### 8.1 逐内核明细（us/call）

| 实现 | 计算内核 | Run 1 | Run 2 | 中位数 |
|---|---|---:|---:|---:|
| FA3 | `SplitFuse::FAInfer<...>` | 951.93 | 949.40 | 950.67 |
| FIA V2 | `aclnnFusedInferAttentionScoreV2_*` | 815.64 | 806.14 | 810.89 |
| FIA V4 | `aclnnFusedInferAttentionScoreV4_*` | 817.99 | 804.30 | 811.15 |
| FIA V4（torch_npu 入口） | `aclnnFusedInferAttentionScoreV4_*` | 818.42 | 805.09 | 811.76 |

FA3 每调用另有约 1.2 us 的 `aclnnInplaceFillScalar`（已排除）。

### 8.2 汇总对比

| 指标 | Run 1 | Run 2 | 中位数 |
|---|---:|---:|---:|
| FA3（us/call） | 951.93 | 949.40 | 950.67 |
| FIA V2（us/call） | 815.64 | 806.14 | 810.89 |
| FIA V4（us/call） | 817.99 | 804.30 | 811.15 |
| FA3 / FIA V2 | 1.167x | 1.178x | **1.172x** |
| FA3 相对 FIA V2 慢 | 16.7% | 17.8% | **17.2%** |
| FA3 / FIA V4 | 1.164x | 1.180x | **1.172x** |
| FA3 相对 FIA V4 慢 | 16.4% | 18.0% | **17.2%** |
| V4 / V2 | 1.003x | 0.998x | **1.000x** |

### 8.3 正确性 sanity check（不计时）

| 实现 | max_abs_diff | rel_l2 |
|---|---:|---:|
| FIA V2（直接 aclnn） | 2.441406e-04 | 2.630e-04 |
| FIA V4（直接 aclnn） | 2.441406e-04 | 2.630e-04 |
| `torch_npu.npu_fused_infer_attention_score_v2`（V4） | 2.441406e-04 | 2.630e-04 |

### 8.4 解读

- B=32 时 FA3 的单次耗时是 B=1 的 16.9 倍，而 FIA 只有约 10.3 倍；说明该 shape 下 FA3 的 batch 并行扩展性弱于 FIA，B=32 反转为 FIA 快约 17%。
- V2 与 V4 在 B=32 下几乎完全同速（差异 <0.1%），选择哪个版本对性能影响可忽略。
- 该结论仅针对 `Sq=293 / Skv=1024 / Hq=16 / Hkv=8 / D=128` 的纯 device kernel 口径，不代表其他 shape 或端到端口径。

## 9. V4 深度性能剖析（msprof 深采，B = 32）

### 9.1 采集方法

- 驱动脚本：[profile_fia_v4_metrics.py](file:///home/w00811046/code/fan/mlatest/flash-attention-npu/docs/test/perf/profile_fia_v4_metrics.py)，每次只跑一个实现，保证 trace 里只有一个计算内核。
- 采集命令（7 组 `aic-metrics` 分开跑，msprof 每次只支持一组）：

```bash
msprof --output=./perf_outputs/msprof_v4_b32_PipeUtilization \
       --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
       --task-time=on --ascendcl=on \
       python docs/test/perf/profile_fia_v4_metrics.py --impl v4 --batch 32
```

- 每轮 5 次 warmup + 10 次计时，取后 10 次的 `Task Duration(us)` 与各 pipe 指标中位数。
- 注意：CANN profiler 依赖进程正常退出才会 flush 设备侧数据，驱动在 msprof 下用 `sys.exit`（`MSPROF_PROFILE_RUN=1`），裸跑时仍用 `os._exit` 规避 aclnn 析构 double free。

### 9.2 关键指标（V4，B = 32，Sq = 293）

| 指标 | 值 | 解读 |
|---|---:|---|
| Kernel 时长 `Task Duration` | **811 us** | 与 torch_npu profiler 口径一致 |
| `Block Num` / `Mix Block Num` | 20 / 40 | 20 个 AI Core 全部占满 |
| AIC 占用 `aicore_time` | 774.3 us | 矩阵/搬运/标量主路径 |
| AIV 占用 `aiv_time` | **811.8 us** | 与总时长几乎相等 → **向量侧是临界路径** |
| Cube 实测算力 `aic_cube_fops` | 89.29 GFLOP | 平均 **110 TFLOPS**，MFU ≈ **37%**（20 核 × 4096 MAC × 2 × 1.8 GHz ≈ 295 TFLOPS 峰值） |
| `aic_mac_ratio` | 0.38（294.7 us） | cube MAC 阵列只有 38% 时间在算 |
| `aic_mte2_ratio` / `aic_fixpipe_ratio` / `aic_scalar_ratio` / `aic_mte1_ratio` | 0.43 / 0.43 / 0.41 / 0.34 | 搬运、搬出、标量各占四成，cube 在等数据/等 softmax |
| `aiv_vec_ratio` / `aiv_scalar_ratio` | 0.57（465.7 us） / 0.49（397.9 us） | 向量侧 softmax 与标量寻址开销很重 |
| `cube_utilization(%)` | 95.0 | AI Core 几乎无调度空转，但不等价于 MAC 利用率 |
| L2 读命中 | AIC 82.8%（1.99M hit / 0.41M miss）、AIV 99.8% | K/V 复用充分 |
| HBM 读/写带宽 | AIC 0.84 / 0.73 GB/s（每核均值） | 远未饱和，**不是访存瓶颈** |
| `aiv_vec_bankgroup_cflt_ratio` / `bank_cflt_ratio` | 4.0% / 2.2% | 无 bank 冲突问题 |

### 9.3 瓶颈判定

1. **临界路径在 AIV**：aiv_time 811.8 us ≈ kernel 时长，AIC 只要 774.3 us。即使把 AIC 侧完全优化掉，收益上限也只有 4.6%。
2. **AIC 侧不是算力打满**：MAC 仅 38%，MTE2/fixpipe/scalar 各 40% 左右。典型 FA 的 QK→softmax→PV 串行依赖导致 cube 空转，属于「依赖等待 + 标量开销」型，而不是单一 pipe 打满。
3. **不是访存瓶颈**：算术强度 89.29 GFLOP / 211 MB ≈ 423 FLOP/B，远高于该机器算力/带宽平衡点（≈184 FLOP/B）；L2 命中 83%–99.8%，HBM 带宽占用极低。
4. **AIV 标量占比 49%（397.9 us）异常高**，说明 mask 索引、行偏移、softmax 归约等按元素标量代码偏多，是 AIV 侧最值得压的部分。

对比 FA3（B=32，941 us）：`aic_scalar_ratio` 0.74（688 us）、`aic_mte2_ratio` 0.68（631 us），实测算力 87.48 GFLOP / 941 us ≈ 92.9 TFLOPS（MFU 31.5%）。FA3 的标量与搬运开销约为 V4 的两倍，这正是 B=32 下 V4 反超的结构性原因。

### 9.4 冗余计算与形状量化

按右下角 causal 语义，有效 key 对为 `293×732 + 293×292/2 = 257,254` 个/（batch, head），理想计算量：

```
257,254 × 32 × 16 × 512 FLOP = 67.4 GFLOP
```

而 V4 实际执行 89.29 GFLOP，即 **1.32x 冗余**（相对「无 mask 密集计算」78.6 GFLOP 也有 1.14x）。其中一部分来自 causal 块不做细分跳过，另一部分来自 Sq 尾部量化。

固定 B=32、只改 Sq 的实测（同一驱动 `--seq-q`）：

| Sq | Kernel 时长（us） | Cube FLOPs（GFLOP） | 实测算力（TFLOPS） | 说明 |
|---:|---:|---:|---:|---|
| 256 | 595.2 | 77.85 | **130.8** | 128 对齐，效率最高 |
| 288 | 801.0 | 88.92 | 111.0 | 非 128 对齐 |
| **293（正式 shape）** | **809.6** | **89.32** | **110.3** | 非 128 对齐，处于低效区间 |
| 320 | 832.2 | 97.21 | 116.8 | 非 128 对齐 |
| 384 | 905.9 | 116.56 | **128.7** | 128 对齐，效率回到最高档 |

结论：**Sq 是否 128 对齐直接决定 12%–16% 的 cube 效率**（130 TFLOPS vs 110 TFLOPS）。Sq=293 落在低效区间，是当前最明确的、可回收的浪费来源。

### 9.5 API 参数层面的实测空间

| 试验 | 结果 | 结论 |
|---|---|---|
| `inner_precise=1`（高性能模式） | 805.7 us vs 815.6 us（**-1.2%**） | 精度模式几乎无收益，且会牺牲精度，不建议 |
| `aclnnFusedInferAttentionScoreV2` vs `V4` | 810.89 vs 811.15 us（B=32） | 同档，版本选择不影响性能 |
| B=1 | 81.3 us，`Block Num=16`（**4 个核空闲**） | 小 batch 有 20% 核闲置；B=32 已占满 20 核，继续加 batch 只增时长 |
| `softmax_lse_flag=False`、`antiquant_mode=0`、`block_size=0` | 已是零开销配置 | 无可回收项 |

### 9.6 优化空间结论

| 层级 | 可回收空间 | 依据 | 能否由调用方控制 |
|---|---:|---|---|
| API 参数（精度模式、版本、开关） | **≈1%** | `inner_precise=1` 仅 -1.2%，V2/V4 同档 | 可，但收益可忽略 |
| 形状/打包（Sq 128 对齐、TND 序列打包） | **10%–15%** | Sq=293 110 TFLOPS vs 128 对齐 129–131 TFLOPS；89.3 GFLOP 对理想 causal 67.4 GFLOP 有 1.32x 冗余 | 可，需上游调整 token 组织 |
| 算子内部实现（降 AIV 标量、softmax 融合、更大 tile 喂满 cube） | 理论最高 **~2.6x**（MFU 37% → 100%） | AIC MAC 仅 38%，AIV scalar 49%，AIV 为临界路径 | 不可，属 CANN 算子内部 |

**总结**：在当前 shape 下，V4 的 kernel 调度已经比较饱满（20/20 核、cube 95% 占用、非访存瓶颈），**API 参数层面基本没有优化空间（≈1%）**；真正的余量在 10%–15% 的形状量化/序列打包，以及算子内部实现（标量开销、softmax 融合）带来的更大理论空间。若只看「调用方可用手段」，最值得做的是让 Sq 尽量 128 对齐（或改用 TND 打包多序列），而不是换接口或调精度模式。
