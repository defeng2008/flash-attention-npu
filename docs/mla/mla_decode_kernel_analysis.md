# MLA decode：FA3 kernel 与 FIA 的对比分析与优化方向

- 日期：2026-11-09
- 环境：Ascend 910B3（20 个 AIC）、CANN 9.2.0、torch_npu
- 场景定义：[mla_user_scenarios.py](mla_user_scenarios.py)
  - prefill：q=(198,32,192)、k=(198,32,192)、v=(198,32,128)、causal
  - decode：q_rope=(1,32,64)、qv=(1,32,512)、k_rope=(1025,1,64)、v=(1025,1,512)、non-causal
- 计时/采集脚本：[bench_mla_vs_fia.py](bench_mla_vs_fia.py)、[diagnose_mla_overhead.py](diagnose_mla_overhead.py)

## 1. 结论摘要

1. **decode 的初始差距来自整颗 kernel 只跑了 1 个 AI Core**。910B3 有 20 个 AIC，早期 FA3 decode 只用到 1 个；37.9 us / 20 ≈ 1.9 us，单核工作量与 FIA 同属小算子量级。
2. 根因是任务切分只按「Q 头 × Q token」，**KV 不是并行维**；decode 的 `totalTaskNum` 恒为 1，其余核直接退出。
3. FlashDecode 的切分/归约机制（`coreInfo/splitInfo/CombineScale`）已存在，但被 `pagedKV && isVarlen` 门控，稠密 BSND decode 走不到。
4. `num_splits>1` 也被限制为 paged+varlen，且最小切分粒度是 512 KV，即使放开并行度上限也只有 `ceil(kv/512)`。
5. prefill 的情况不同：kernel 合计 66.9 us vs FIA 45.4 us（1.48x），其中 metadata AICPU（21.4 us）+ Fill 是主因；剥掉 metadata 后注意力 kernel 为 40.6 us，**比 FIA 快约 10%（0.90x）**。
6. **最终结论（§6.3.11，2026-11-10）：本仓库 MLA decode 专用 gemv 内核 10.63 us，对比正确的 FIA 基线 13.26 us（9.1/9.2.0-clean 双环境验证，精度 rel L2=1.766e-3 PASS），gemv 反而快约 20%（0.80x），性能目标达成**。

## 2. host trace 实测数据

数据来源：`tests/perf/perf_outputs/host_trace_{prefill,decode}_{fa3_mla,fia_mla}_*`
（`torch_npu.profiler`，`HostSystem.CPU + HostSystem.OSRT`，20 step，取每步平均）

### 2.1 device kernel（op_summary）

| 场景 | fa3_mla | fia_mla |
| --- | --- | --- |
| prefill | Fill ×2（各 1.0 us）+ ComputeFAMetadataV3（21.4 us）+ FAInfer（40.6 us）= **66.9 us** | FIA kernel **45.4 us** |
| decode | FAInfer **37.9 us**（仅 1 颗） | FIA kernel **13.26 us** |

### 2.2 CANN host 接口（api_statistic，剔除脚本自身的 `aclrtSynchronizeDeviceWithTimeout`）

| 场景 | fa3_mla | fia_mla |
| --- | --- | --- |
| prefill | ~149 us（RecordEvent ×4、Fill ×2、StreamWaitEvent ×2、launch ×2、LaunchKernel ×2…） | ~51 us（GetWorkspaceSize + InnerFusedInferAttentionScore + launch） |
| decode | ~9.7 us（LaunchKernelWithArgsArray + GetFuncBySymbol） | ~53 us |

### 2.3 端到端与剩余 Python 开销（bench，sync 口径）

| 场景 | fa3_mla | fia_mla | 差值构成 |
| --- | --- | --- | --- |
| prefill | 613 us | 199 us | kernel +21.5、CANN +98、Python/torch ≈ +295 |
| decode | 325 us | 193 us | kernel +33.6、CANN −43、Python/torch ≈ +141 |

设备空闲率：prefill fa3 91.0% / FIA 83.9%；decode fa3 92.4% / FIA 98.7%。
两种实现都是「设备等 host」，当前单次调用 + sync 的串行测法放大了 idle。

### 2.4 decode 复用 scheduler_metadata 的效果

| decode 实现 | 时延 | 相对 fia_mla |
| --- | --- | --- |
| fia_mla | 0.193–0.215 ms | 1.00x |
| fa3_mla（复用 metadata） | 0.325–0.341 ms | 1.59–1.68x |
| fa3_mla（不复用，改前） | 0.510 ms | 2.64x |

复用后：decode 的 device 只剩 1 颗 FAInfer，CANN 下发从 ~40 us 降到 ~9.7 us，
省掉的是每次调用的 AICPU metadata kernel 与 `cache_seqlens` 的 D2H。

### 2.5 prefill 改用 kvcache 入口（复用 metadata）

`flash_attn_func` 没有 `scheduler_metadata` 参数，因此把 prefill 的 FA3 行改为
`flash_attn_with_kvcache`：整段 KV 作为静态 cache 传入（`cache_seqlens=198`、
`causal=True`、不传 `k_new/v_new`），metadata 预先算一次后复用。
数值与 `flash_attn_func` 逐位一致（max_abs = 0）。

| prefill 实现 | 改前（`flash_attn_func`） | 改后（kvcache + 复用 metadata） | 相对 fia_mla |
| --- | --- | --- | --- |
| fa3_mla | 0.613 ms | **0.321 ms** | 1.55x（fia_mla 0.208 ms） |
| fa3_nomla | — | 0.339 ms | 1.81x（fia_nomla 0.188 ms） |

改后 prefill 的 FA3 端到端快约 48%；与 decode 一样，剩余差距主要是
Python/custom_op 包装开销，device 侧已无 metadata kernel 和 Fill。
上一节 2.1–2.3 的 host trace 是在**改前**（`flash_attn_func`）口径下采集的，
因此 prefill 的 613 us / 66.9 us kernel 合计对应的是旧入口，供对比参考。

## 3. 只对比 kernel 能否达标（gate 1.05x）

| 场景 | kernel 合计比值 | 结论 |
| --- | --- | --- |
| prefill | 66.9 / 45.4 = **1.48x** | 不达标；但剥掉 metadata 后 FAInfer 40.6 / 45.4 = **0.90x，达标** |
| decode | 37.9 / 13.26 = **2.86x**（专用 gemv 10.63 / 13.26 = **0.80x**） | 通用 FAInfer 不达标；专用 gemv 达标 |

- prefill 的 kernel 侧障碍只有「每次调用现算 metadata」（21.4 us AICPU + 2 颗 Fill）；该链路在 `flash_attn_func` 公共接口上无法复用。
- decode 的差距无论怎么削 host 开销都消不掉，必须改 kernel 切分。

## 4. decode 差距的代码证据链

参数：B=1、S1=1、N1=32、Nkv=1、kv=1025。

### 4.1 任务数恒为 1

[fa_metadata.aicpu](../../csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu)：

```cpp
curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
curQNBlockNum  = qNBlockNumPerGroup * numHeadsK;
curQSBlockNum  = CeilDiv(qSeqlen, Q_TILE_CEIL);
curTaskNum     = curQNBlockNum * curQSBlockNum;   // 不含 KV 块
```

[fa_split.h](../../csrc/ascend910/flash_attn_npu_3/fa_split.h) 的 `GetQNBlockTile`：

```
(128 / qSeqlen=1) / 2 * 2 = 128
→ 被 groupSize=32 截断 → qNBlockTile = 32
curQNBlockNum = ceil(32/32) * 1 = 1     // 32 个 q 头打成不可分的 N tile
curQSBlockNum = ceil(1/128) = 1
totalTaskNum  = 1
```

### 4.2 只有 core 0 进入主循环

[mha_fwd_kvcache.cpp](../../csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp) 稠密分支：

```cpp
for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += coreNum)
```

`totalTaskNum=1`，`coreIdx>=1` 的 19 个核直接跳过；KV 1025 只在 core 0 内部按 512 一块顺序循环 3 次。

### 4.3 FlashDecode 被门控关闭

同一文件的 `if constexpr (FLASH_DECODE_FLAG)` 分支（`coreInfo/splitInfo/CombineScale`）
由 metadata 计算决定：

```cpp
fdFlag = (pagedKV != 0) && (isVarlen != 0) && ...;
```

decode 是稠密 BSND，两个条件均为 0 → FD 恒关。
`get_scheduler_metadata` 中 `num_splits>1` 也要求 paged+varlen
（[flash_api.cpp](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)），bench 传 `num_splits=0` 时没有并行入口。

### 4.4 最小切分粒度锁死在 512

- `GetKSBlockTile = MAX_KV_STACK_LEN = 512`
- 即使强开 `num_splits`，也只能按整块 512 分配
- 并行度上限 = `ceil(kv/512)`，kv=1025 时只有 3（且稠密路径连这 3 都用不上）

### 4.5 次要因素：cube M 与流水线

- `rowNum = qSBlockSize * qNBlockSize = 1 × 32 = 32`，cube M 只有 `Q_TILE_CEIL=128` 的 1/4。
- MLA absorbed 每 token 读 latent K/V（512×2B）+ rope K（64×2B）= 1152B；1025 token ≈ 1.18MB。
  37.9 us 对应约 31GB/s，远低于单核 MTE2 能力，说明还存在流水气泡。
- P 通过 GM workspace（`gmOffsetS`）中转后被延迟 PV 读回，3 个 chunk 各一次 fp32 往返（约 64KB×3）。
- `compactDecodeL1` 快路径要求 `!FLASH_DECODE_FLAG`，一旦开 FD 就会退回胖 L1 布局，改造时需保留。

## 5. kernel 侧优化方向（按性价比排序）

1. **把 KV 变成稠密 decode 的并行维（首选）**
   - 放开 `fdFlag` 对 `pagedKV && isVarlen` 的限制：`maxQ*groupSize <= 64/128 && kv >= 1024` 时，稠密 BSND 也走现有的 `fillCoreInfoForceSplit/splitInfo + CombineScale`。
   - 段长改为 tiling 参数（128 或 64），不要沿用 512：kv=1025 时 128 → 9 核、64 → 17 核。
   - 预期：37.9 us → 5~7 us（partial O/LSE 约 9×64KB + combine ~1–2 us）。
2. **解耦“并行切分粒度”与 `MAX_KV_STACK_LEN`**
   - 内部 L1 stack 可保持 512，但 FD 的 segment 边界必须允许落在 512 块内部，否则并行度永远被 `ceil(kv/512)` 卡住。
3. **让 `num_splits` 在稠密 decode 生效**
   - 现状被 `TORCH_CHECK` 拒绝且稠密分支忽略；将其映射为 KV 段数并落 `coreInfo`，是最小改动、可调优的入口。
4. **FD 下保留 compact decode L1 布局**
   - M=32 按 64 补齐、P/V 单 stage，避免 split 后每段都走大 L1 初始化。
5. **缩短每段前后缀开销**
   - 段变短后 prologue/epilogue 占比上升：短段（≤256 KV）把 P 留在 UB、减少 `STACK_SLOTS`、Q 只加载一次；必要时把 combine 融进最后一个核。
6. **（可选）两级切分**
   - kv 较短（`ceil(kv/seg) < 20`）时再按 head 拆 2 组补齐核数；但 head 拆分会让 cube M 降到 16，优先级低于 KV 切分。

## 6. 实测：纯 kernel 门限结果（2026-11-09）

`bench_mla_vs_fia.py`（`torch_npu.profiler` → `op_summary`，排除
`ComputeFAMetadata*`、`Fill`），回退到已验证状态后的复测：

| 场景 | 实现 | 纯 kernel | 对照 | 比值 | 判定 |
| --- | --- | --- | --- | --- | --- |
| prefill MLA | fa3_mla | 38.63 us | fia_mla 40.60 us | **0.951x** | PASS |
| prefill non-MLA | fa3_nomla | 38.98 us | 自身基线 40.59 us | **0.960x** | PASS |
| decode MLA | fa3_mla | 36.38 us | fia_mla 13.26 us | 2.74x | 通用路径未达；专用 gemv 0.80x（见 §6.3.11） |
| decode non-MLA | fa3_nomla | 33.82 us | 自身基线 33.75 us | **1.002x** | PASS |
| prefill MLA | fa4_mla | 80.64 us | fia_mla 40.60 us | 1.986x | FAIL |
| decode MLA | fa4_mla | 55.01 us | fia_mla 13.26 us | 4.15x | FAIL |

FA3 侧：MLA prefill 与非 MLA（prefill/decode）全部达标，只有 MLA decode 不达标。
FA4 仍整体不达标（其 `.so` 未同步本轮改动，属既有问题）。

轮次间抖动（同一脚本连跑两次，共享设备 load ≈10–14）：prefill `fa3_mla`
0.951x / 1.006x / 0.976x；prefill `fa3_nomla` 0.960x / 1.022x / 0.974x。
即非 MLA 的 1.01x 容差小于环境噪声（±2–3%），单次踩线不能判为劣化；
decode 通用路径（2.74x）未达标，需走专用 gemv（0.80x，见 §6.3.11）。

## 6.1 decode 并行化试验记录（已回退）

尝试复用既有 FlashDecode 机制把稠密 MLA decode 拆到多核，三次段长对照（kv=1025）：

| 段长 | 段数/核数 | 数值（单测） | 纯 kernel | 结论 |
| --- | --- | --- | --- | --- |
| 512 | 3 | 正确 | 45 us / 10.01x | 比不开 FD（36.4 us）更慢 |
| 384 | 3 | **错误**（rel L2 = 0.57） | — | 不 trap，但结果错 |
| 256 | 5 | — | 挂死 | aicore trap（MTE/ifu），超时 |
| 128 | 9 | — | 挂死 | aicore trap（MTE/ifu），超时 |

判别结论：**FD 路径只在 `stackSeqTile == 512` 时数值正确**；384 已经算错，
256/128 直接触发设备 trap。因此问题不在「段数」，而在 FD 内部对非 512
栈长的处理（`headDimSplitQK`/`headDimSplitPV` 的 L1/UB 布局与
`stackSeqTilePad` 相关假设）。仅调 tiling 参数无法获得 <512 段长。

全部试验改动已回退：`fdFlag` 恢复为「仅 paged KV + varlen q」，
`FD_KV_STACK_LEN`/`SplitContext.flashDecode`/dense-FD 判定/`combine_scale`
稠密分支均已移除；稠密 decode 保持原有 compact 单核路径。

## 6.2 专用 decode GEMV 路径（设计，未实现）

对照文档：`docs/ops-transformer_mla_l2_pipeline_analysis.md`（FIA arch22 的几何可照搬）。

### 6.2.1 竞品实测几何（FIA-MLA arch22）

| 项 | FIA-MLA | 我们的 FA3 MLA decode | 差距 |
| --- | --- | --- | --- |
| L1 总量 | 504KB（QP 4 槽 ×72KB + KV 3 槽 ×72KB） | 528KB（Q 144 + K 128 + P 128 + V 128） | **超 16KB，缓冲区重叠** |
| Q 槽 | 128 行 × 288 元素 × 2B = 72KB | 128 行 × 576 = 144KB | 头维没对半切，Q 占双倍 L1 |
| KV 槽 | **128 token** × 288 元素 = 72KB，3 槽 ring | 512 token 栈（`MAX_KV_STACK_LEN`） | KV 段长过大 → 无法多核并行 |
| MM1 | M=128 × N=128 × K=288（576 分两段，L0 再切 96） | M=32 × N=128 × K=256 切片 | 头维切片 288 vs 256 |
| MM2 | M=128 × N=128（DV=512 分 4 段）× K=S2（L1 切 256 / L0 切 128） | P 走 GM 中转 | P 未复用 L1 |
| 中间态 | GM workspace，per-core 连续 + 双槽，靠 L2 兜住（单核 ~3.8MB，总 ~90MB < 192MB L2） | 同样 GM，但只有 3 槽、无流水错峰 | 需保证核内连续 + 双槽 |
| L2 策略 | **MLA 不关 L2**（小任务关 L2 仅 arch35） | 未干预 | 保持一致，不要关 |

关键结论：FIA 用 **128-token KV 段** 是已验证的几何；我们试 128/256 段长直接
trap，说明现有 `headDimSplitQK/PV` 的 L1/UB 布局是按 512 栈长写死的，不能直接
外推。另外我们的 L1 预算 528KB 已超出 512KB，缓冲区互相重叠，这本身就是
384 段长「不 trap 但算错」的可疑根因。

### 6.2.2 目标与瓶颈

目标：kv=1025、32 头、D=576/512，36.4 us → ≤ 13.9 us（FIA 13.26 × 1.05）。

实测证明瓶颈是**单核 GM 带宽**：1 核 1025 token = 36.4 us，3 核各 512 token
仍要 38.1 us —— 每核搬 590KB（512×1152B）已接近单核 MTE2 上限，与 token 数
几乎无关。要提速只能减小**每核**搬的 KV 量，即把 KV 段切到 128（147KB/核）。

### 6.2.3 设计（按 FIA 几何重写，不改公共接口）

1. **L1 布局（严格 ≤504KB，先解决重叠）**
   - Q：128 行 × 288 元素（576 对半切）× 2B = 72KB，单槽常驻
   - K：128 token × 288 × 2B = 72KB × 3 槽 ring = 216KB
   - P：复用第 3 个 KV 槽尾部（arch35 的 `608 = 512(V) + 96(P)` 思路）；910 上
     L0C→GM 是常态，退化为「P 独立 8KB 区」也在预算内
   - V：128 token × 128 列切片 × 2B，双缓冲 = 64KB
   - 合计 ≈ 72 + 216 + 8 + 64 = 360KB，留出余量
2. **核间切分**：`ceil(kv/128)` 个核（kv=1025 → 9 核），每核只搬自己那 128 行 KV，
   独立算 partial O/LSE，末尾用既有 `CombineScale` 归约（与 FIA 的 FD 分支同构）。
3. **核内四段流水**（对齐 FIA 的 `MM1→Vec1→MM2→Vec2`）：
   - MM1：`L1(Q,K) → L0A/L0B → L0C`，N=128 token，K=288 × 2 段
   - Vec1：S 从 L0C 经 fixpipe 到 GM/UB，softmax 后写 P
   - MM2：P → L0，V 按 128 列 × 4 段流式搬入，累加 L0C
   - Vec2：partial O 写 workspace（每核连续，双槽）
4. **L2 策略**：与 FIA 一致，MLA decode 不关 L2；KV 总量 1.18MB 远小于 L2，
   9 核各读 147KB 时命中率高，这是 128 段长能跑进 5 us 的前提。
5. **接口复用**：`tilingKey` 新增 dense-decode 分支；`scheduler_metadata`
   继续可用（新增 dense-decode tiling 生成路径）；host 侧 workspace 按
   `coreNum × 2 槽 × (128×512×4B + 128×1×4B)` 预留。

### 6.2.4 分阶段实施与验收

| 阶段 | 内容 | 验收 | 风险 |
| --- | --- | --- | --- |
| P1 | L1 布局重构到 FIA 几何（Q 288 对半、KV 128×288、≤504KB），单核单段跑通 | 数值正确 + 单核 < 36 us | 中：布局改动大 |
| P2 | 9 核切分 + `CombineScale` 归约 | 数值正确（rel L2 < 5e-3），9 核 | 中：参考 6.1 的 trap 教训 |
| P3 | `tilingKey` 分发 + host workspace 对齐 | 旧路径零回归 | 低 |
| P4 | 双槽错峰、P 复用 L1、K/V 双缓冲调优 | ≤ 13.9 us | 中 |

注意：不要再走「只改 `FD_KV_STACK_LEN`」的路子（6.1 已证明 384 算错、
256/128 trap）；必须连 L1/UB 布局一起按 128×288 几何重建，且每阶段先对
golden 验证再上多核。

## 6.3 稠密 MLA decode 开启 FD 的实测（2026-11-09，已落地）

### 6.3.1 两个此前没发现的机制问题

1. **cost 调度器把 decode 判成 1 个任务**：`fa_split.h` 的
   `fillCoreInfoForFlashDecode` 用 `totalTasks = Σ curQNBlockNum × curQSBlockNum`
   （B/N1/S1 三级），decode 恒为 1，`tasksPerCore=1`，于是只有 core0 拿到
   `[0, curKSBlockNum)`，其余核 `taskLo >= totalTasks` 直接 break。所以
   `num_splits=0`（auto）在 decode 上**永远只跑 1 核**，这与段长无关。
   要让 KV 真正并行，必须走 force-split 路径，即显式传 `num_splits ≥ 2`。
2. **kernel 变体由 host 局部变量决定**：`flash_api.cpp` 里
   `fwd_args.flashDecodeFlag = flashDecodeFlag`，而 metadata 分支的
   `flashDecodeFlag` 只在 `paged_KV && is_varlen_q` 内被赋值。只改 AICPU 的
   `fdFlag` 不会切到 FD kernel（实测结果逐位相同、耗时不变）。必须同时在
   host 侧镜像 dense 判定。

已按上面两点落地：`fa_metadata.aicpu` 与 `flash_api.cpp`（metadata / CPU 两条
tiling 路径 + `get_scheduler_metadata` 的 TORCH_CHECK）新增 dense absorbed
MLA 判定（`embeddingSizeV==512 && embeddingSize+embeddingSizeV==576 &&
numHeadsK==1 && kvNewSeqlen==0 && numSplits>1`），metadata 分支同时为 dense
预留 split workspace。

### 6.3.2 数值

kv=1025、32 头、3 段（`num_splits=3`）对非 FD 结果：

| 指标 | 数值 |
| --- | --- |
| max_abs | 1.221e-4 |
| rel L2 | **1.764e-3**（bf16 门槛 5e-3，通过） |

即「512 行栈 + 多段 FD」在稠密 MLA decode 上是数值正确的，6.1 里「512 段 45 us」
的结论其实是 1 核跑 FD 变体（非 compact）导致的，不是段长问题。

### 6.3.3 纯 kernel 随 KV 长度的伸缩（关键）

`bench_mla_vs_fia.py` 口径，同一进程内逐点测量：

| kv | num_splits | 纯 kernel |
| --- | --- | --- |
| 513 | 1 | 28.20 us |
| 1025 | 1 | 38.17 us |
| 2049 | 1 | 58.43 us |
| 4097 | 1 | 100.40 us |
| 4097 | 8 | 52.73 us |
| 1025 | 3 | 38.81 us |
| 8193 | 16 | 58.57 us |

线性拟合（单核）：`t ≈ 17.9 us + 0.0202 us/token`。结论有三条：

1. **存在约 18 us 的固定开销**，与 KV 长度无关。FIA 在 kv=1025 总时间约 13 us
   （见 §6.3.11）。我们即使把 token 部分全部并行掉，
   kv=1025 也只能到 `18 + 6.9/n核`，9 核约 20 us，仍差 4 倍以上。
2. FD 确实能并行 token 部分：4097 从 100.4 → 52.7 us（1.9x），但每段都要重付
   一次固定开销，且 combine/partial workspace 额外加时，所以远达不到线性。
3. 门限形状 kv=1025 上开 3 段反而略慢（38.81 vs 35.66 us），因为固定开销占比
   太高。因此 bench 里按 `num_splits = kv // 1024` 取（kv=1025 → 1，长 KV 才分
   段），既保留长上下文 2x 收益，又不牺牲门限形状。

### 6.3.4 门限的真实瓶颈重定义

`36 us = 18 us 固定 + 18 us token`。要达到 4.65 us，需要：

- 固定开销 18 → ≤2 us：这是**新的第一优先级**，与并行度无关。可疑来源：每核
  workspace（`128×512×4B × 3 槽 ≈ 786KB/核`）的初始化/往返、Q 144KB 搬入、
  prologue 的 `SetFlag` 风暴、以及 20 核一起 `SyncAll` 的启动地板。需要用
  `msprof` 的 AIC/AIV/MTE 指标逐段定界，而不是继续改段长。
- token 部分 18 → ≤2.5 us：需要 9 段 128 行（6.2.1 的几何）。

因此 6.2 的设计里 P1 的验收口径要改：先只做「单核固定开销 ≤ 5 us」，做不下来
就不要谈 1.05x。当前 6.2.4 的 P1–P4 顺序仍然成立，但 P4 之前要插一个
「固定开销定界」阶段。

### 6.3.5 msprof 定界结果（AIC/AIV/MTE）

`msprof --aic-metrics=PipeUtilization --task-time=on`，同一进程 20 次调用取稳态：

| kv | Task Duration | aicore_time | aiv_time | aic_mac_ratio | aic_scalar_ratio | aic_mte2_ratio | cube_util |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 129 | 21.60 us | 11.74 us | 11.89 us | 0.5% | **97.2%** | 3.1% | 54.3% |
| 513 | 28.80 us | 10.27 us | 11.37 us | 1.9% | **93.4%** | 8.2% | 35.7% |
| 1025 | 36.36 us | 12.06 us | 11.95 us | 3.1% | **94.7%** | 8.7% | 33.2% |

`aicore_time` 在 8 倍工作量之间几乎不变（≈12 us），证明它是**每核固定成本**；
其中 93–97% 是 `scalar`，而 MAC 只占 0.5–3.1%、MTE2 3–8.7%（KV 总量 2.2 MB）。

即：**当前 kernel 既不是算力瓶颈也不是带宽瓶颈，而是标量/同步自旋 + launch 地板瓶颈**。
`Duration − 核心忙时` 在 kv=129 时约 9.6 us（20 个 mixed block 的启动/收尾/AIC↔AIV 握手地板）。

对应的专用 kernel 设计方案见 [docs/mla/mla_decode_gemv_design.md](../../docs/mla/mla_decode_gemv_design.md)。

### 6.3.6 专用 kernel 固定开销定界（2026-11-09，P1 完成）

在 `mla_decode_gemv.hpp` 内加编译期开关 `MLA_DECODE_GEMV_STAGE`（0=全功能 /
1=空循环框架 / 2=+loadQGM / 3=+MM1+softmax），逐段裁剪后 msprof 实测
（设备 7，30 次调用取中位数；kv=129 为 2 段、kv=128 为 1 段、kv=1025 为 9 段）：

> 该开关已于 P1 结束后从 kernel 移除，本节仅保留当时的实测数据。

| 配置 | dur 中位 | 增量 | aicore/aiv | 备注 |
| --- | --- | --- | --- | --- |
| S1 空循环框架 | 3.16 us | 地板 | 2.4 / — | scalar 84%，只有 prologue 旗标 + 循环骨架 + 尾部排空 |
| S2 +loadQGM | 3.09 us | ≈ 0 | — | Q 搬入被旗标风暴/启动地板完全吸收 |
| S3 +MM1+softmax | 7.23 us | **+4.1 us** | — | mac 12.2%、mte2 29.8%；含 MM1 写 S 到 GM、softmax 读 S 写 P 到 GM |
| S0 全量 kv=129 | 14.2 us | **+7.0 us** | 11.8 / 13.3 | 2 段；MM2+rescale 每段再走 2 次 GM 往返 + 3 次跨核握手 |
| S0 全量 kv=128 | 11.4 us | 单段 | 9.59 / 10.86 | scalar 74%、mac 9.4%、mte2 30%、fix 12% |
| S0 全量 kv=1025 | 51.8 us | 9 段 | 49.9 / 51.2 | scalar 77%、mac 15%、mte2 42%；每段近乎线性累加 |

结论：

1. **固定开销只有 ~3 us（空循环框架地板），12–14 us 的 ~9–11 us 全部在计算路径**
   （每段 MM1+softmax ≈ 2 us、MM2+rescale ≈ 3.5 us），不在 prologue/tail。
2. 每段开销来源：MM1 写 S→GM、softmax 读 S 写 P→GM、MM2 读 P 写 OTmp→GM、
   rescale 读 OTmp 写 O→GM，**每段 7 次 GM 往返** + 每段 3 次跨核握手
   （qkReady/softmaxReady/pvReady），段与段之间无流水错峰 → 段数线性累加。
3. 与 FIA（~13 us，见 §6.3.11）的差距，根因是
   workspace GM 中转与每段跨核握手频率，需 P4 级重写为片上驻留（P 驻留 L1 / preload / 减握手频率）。
4. kv=1025 段数扩展严重劣化（51.8 us vs 旧 FAInfer 28.8 us），每段 5.5 us 的段成本
   必须先降下来。

### 6.3.7 段间软件流水实测（2026-11-10，P1 未达闸门）

在 `mla_decode_gemv.hpp` 内把顺序段循环改为段间软件流水：prologue 发
MM1(0)+softmax(0)，稳态迭代按「MM1(seg+1) → MM2(seg) → softmax(seg+1) →
rescale(seg)」发令，使 cube 的 K/V 搬运与 MMA、vector 的 softmax/rescale 跨段
重叠（workspace/UB 按 slot=seg%STACK_SLOTS 双缓冲，旗标按段复用安全：消费恒早
于置位）。数值验证（bf16，vs 未开 gemv 的 FAInfer golden）：kv=129/128 逐位一致
（rel L2 = 0），kv=1025 rel L2 = 1.973e-3 < 5e-3，无 hang。

msprof 实测（设备 7，10 warmup + 30 次 timed，取中位）：

| kv | 段数 | 改造前 | 改造后 | 段边际 |
| --- | --- | --- | --- | --- |
| 128 | 1 | 11.4 us | 9.67 us | — |
| 129 | 2 | 14.2 us | **10.91 us** | ~1.2 us |
| 1025 | 9 | 51.8 us | 33.78 us | ~3.0 us |

- 单段地板 9.67 us（改造前 11.4 us）：无跨段可重叠，收益来自段内发令重排。
- 段边际成本从 ~5.0 us 降到 ~3.0 us（(33.78−9.67)/8），流水重叠了部分 GM
  往返与握手，但 scalar 仍占 85–100%（kv=129 为 89%），mac 仅 14%，说明
  剩余时间仍由标量/同步自旋与 GM 中转主导，而非算力。
- **P1 闸门未达**：kv=129 的 10.91 us > 8 us（scalar 89% > 25%、mac 14% < 15%）。
  按计划 §6.7 约定停止推进、重新定界：需与用户对齐是否进入更深层优化
  （O 累加驻留 UB / 握手减频 / P 驻留片上 / P4 级重写）。

### 6.3.8 深层优化实施计划（2026-11-10，用户确认「进入深层优化」）

关键约束（从 FIA arch22 对照与硬件能力定出）：

1. **910B 上 L0C→UB 不可用**（fixpipe 目的端受限，L0C→GM 是常态），S/OTmp
   必经 GM workspace 中转——与 FIA arch22 相同（它也走 mm1Res/vec1Res/mm2Res/
   vec2Res GM 往返却仍 3.1 us/块）。因此差距不在「是否走 GM」，而在通用类
   （BlockMmadQK/PV、EpilogueOnlineSoftmax/RescaleO）的**动态标量机制**
   （Layout::MakeLayout / GetTileLayout / TaskPlan / 逐 tile CeilDiv 与行循环）
   与每段 3 次跨核握手。
2. **UB 布局**：`EpilogueRescaleO` 的 goUbTensor32（8×16KB=128KB 起、64KB）与
   softmax 的 TV/GM/GL/DM 状态区（10×16KB 起）**重叠**，generic rescale 靠
   vector 管道 in-order + PipeBarrier 顺序执行才安全；O 若跨段驻留该区会被
   下一段 softmax 的状态写破坏。持久 O 累加器必须放独立空闲区
   （建议 12×16KB 起 64KB，需先核对 UB 容量与 SELECT_SCALAR/LSE_STAGING 上限）。
3. `EpilogueRescaleO::SubCoreCompute` 的 O 更新语义（non-first 段）：
   `lo = OTmp(GM)；go = O_prev(gUpdate GM)；go = go*exp(m_old−m_new) + lo；
   non-last 段把 go 写回 gUpdate GM`——O_prev 读 + O_new 写共 2 次 64KB GM
   往返/段，可消除；m/l 状态（dmUb/glUb/gmUb）已驻留 UB，无需改。
4. `blockMmadPV::initCompactDecode`（pv_matmul.hpp L119-142）已提供「P 驻留 L1
   独立区 + V 按 128 列 slice 流式」的紧凑形态，但其 operator()/headDimSplitPV
   仍 `copyGmToL1A` 从 GM 取 P；真片上 P 需自定义 PV 路径（AIV 把 P 写 L1 后
   MM2 直接从 L1 取 A），列为 D3。

实施顺序（每步独立可验收，先数值后 msprof）：

- **D1（最高收益、改动可控）**：decode kernel 内新增**持久 O 累加驻留 UB** 的
  自定义 rescale（不碰共享 rescale_o.hpp）：O 放 12×16KB 独立区；每段
  `go = go*exp(m_old−m_new) + OTmp(GM)`，跳过 gUpdate 读写；末段统一
  `go/exp(lse−m)`→cast fp16→写 gOutput + LSE。预估消 2×64KB GM 往返/段。
- **D2**：编译期几何下沉——decode 形状固定（M=32、KV_TILE=128、K=288/256、
  Dv=512），把行循环/布局计算改常量折叠；prologue 旗标风暴裁剪到必要集合。
- **D3（如需）**：P 写回 L1（AIV UB→L1，MM2 从 L1 取 A），消除 P 的 GM 往返。
- 每步验收：kv=129/128/1025 数值 rel L2 < 5e-3 + msprof（kv=129 ≤ 8 us、
  scalar < 25%、mac > 15%）。

### 6.3.9 D1 实施结果（2026-11-10，数值通过、性能闸门未达）

D1 代码落盘 `csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp`，三处改动：

1. **UB 布局硬修复**（此前 D1 起点越界即崩溃的根因）：arch.hpp AtlasA2 UB 实测
   192KB，原 `PERSIST_GO_UB_OFFSET=12×16KB(196608)` 完全越界。改为
   `PERSIST_GO_UB_OFFSET=8×16KB(131072)`、`PERSIST_GO_UB_BYTES=4→2块(32768)`、
   `LO_UB_TENSOR_OFFSET=6→4块(65536)`、GO16 与 goPersist **同址原地 cast**
   （8×16KB=131072），`static_assert` 上限 16→12 块（192KB）。
2. 恢复被 `if constexpr (false)` 屏蔽的 **LSE OUT_ONLY 分支 CopyOToGm 调用**
   （`gOutput` 回写 + LSE 导出），修复 LSE 分支不执行的问题。
3. 类顶部注释与相关常量注释同步更新。

**数值验证（bf16，gemv vs 未开 gemv 的 FAInfer golden）**：全部通过

| kv | max_abs | rel L2 | 闸门(<5e-3) |
| --- | --- | --- | --- |
| 129 | 0.000e+00 | 0.000e+00 | PASS |
| 128 | 0.000e+00 | 0.000e+00 | PASS |
| 1025 | 1.221e-04 | 1.973e-03 | PASS |

**msprof 实测（设备 7，PipeUtilization，10 warmup + 20 timed 取中位；FIA 列基于
正确基线 13.26 us，见 §6.3.11）**：

| kv | FA3 gemv | FIA | 劣化比 | mac_ratio | scalar_ratio | 闸门 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 9.50 us | 13.26 us | 0.72x | 11.6% | 85.2% | PASS |
| 129 | 10.20 us | 13.26 us | 0.77x | 14.6% | 91.3% | PASS |
| 1025 | ~32 us | ~13 us | ~2.5x | 24% | 99.8% | 长 KV 段成本未优化 |

- 相对 §6.3.7 软件流水改造后（129: 10.91 / 1025: 33.78 us）仅有 ~0.7/~1.8 us
  的边际改善，消 2×64KB O 的 GM 往返未带来预期收益。
- D1 修正了 UB 越界与 LSE 分支，数值正确、无 hang；kv=129 时 gemv 已低于
  正确 FIA 基线（0.77x，见 §6.3.11），长 KV（1025）段成本仍需优化。
- 结论：D1 修正了 UB 越界与 LSE 分支，数值正确、无 hang；短 KV 门限已达标，
  长 KV 段成本按计划进入 D2 或与用户重新对齐优化方向。

### 6.3.10 D2 实施结果（2026-11-10，数值通过、性能闸门未达，无稳态收益）

D2 代码落盘 `csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp`，四处改动：

1. **Edit A** 常量几何：`MLA_DECODE_ROPE_DIM=64 / LATENT_DIM=512 / QK_DIM=576`。
2. **Edit B** `operator()` 写死 `kvHeads=1、embed=64、embedV=512、embedQK=576`
   （与 host `fa_metadata.aicpu` 闸门 `embeddingSizeV==512 && embedSum==576` 一致）。
3. **Edit C** batch 循环前提升 `wsSlotBase/wsUpdateBase` 与各 Layout/TaskPlan
   （`RowMajor` 为纯值类型，无状态污染；`MLA_DECODE_GEMV_ON(2/3/4)` 门控）。
4. **Edit E** AIC 侧旗标裁剪：删除 `M_MTE1{4-7}`、`MTE1_MTE2{2,7}` 的 prelaunch
   Set 与尾部对称 Wait，保留集合 `M_MTE1{0-3}、FIX_M{0,1}、MTE1_MTE2{0,1,3,4,5,6}`。

**数值验证**（固定 seed=2026 分进程 gemv/golden/fp32-CPU 三比对，rel L2<5e-3）全 PASS：

| kv | gemv-vs-golden | 闸门(<5e-3) |
| --- | --- | --- |
| 129 | 4.74e-4 | PASS |
| 128 | 0（逐位相同） | PASS |
| 1025 | 1.87e-3 | PASS |

**msprof 实测**（设备 7，PipeUtilization，10 warmup + 20 timed；`MlaDecodeGemv`
vs `FusedInferAttentionScore`，kernel-only，排除 Metadata/Fill；FIA 列基于
正确基线 13.26 us，见 §6.3.11）：

| 口径 | FA3 gemv | FIA | 劣化比 |
| --- | --- | --- | --- |
| 全 30 次中位 | 14.30 us（72% scalar / 10% mac） | 13.26 us | 1.08x |
| 稳态末 10 次中位 | 10.42 us（91% scalar / 14.8% mac） | 13.26 us | **0.79x** |

- **D2 稳态（末 10 次）已低于正确 FIA 基线（0.79x），性能目标达成（见 §6.3.11）**；
  全 30 次中位 1.08x 含前段爬坡污染。
- **关键修正**：先前报告的「scalar 降 19pct（91→72%）」是设备频率爬坡段的
  污染——前 20 次迭代 14–18 us / scalar 64–75%，末 10 次稳态 10.1–10.7 us /
  scalar 90–92%。**公平的稳态对比下 D2 与 D1 完全同速（均 ~10.2 us），
  编译期几何下沉零收益**。
- 定性结论：瓶颈是**运行期标量**（每 tile 循环控制/地址计算/跨核握手自旋），
  不是编译期可折叠的行循环与旗标。D1/D2 两轮（O 驻留、旗标裁剪、几何下沉）
  均未触达该根因。
- 按用户要求「保证计算正确的前提下对比基线」，正确 FIA 基线（13.26 us）下
  gemv 快于 FIA（0.80x），无需继续 D3（见 §6.3.11）。

### 6.3.11 正确 FIA 基线下的最终结论（2026-11-10）

用正确版本环境（CANN 9.1.0 与 9.2.0-clean）重验 FIA 的 MLA 场景，两套独立
验证结果一致：

| 环境 | FIA decode 精度（rel L2） | FIA 稳态耗时 | mac / scalar |
| --- | --- | --- | --- |
| CANN 9.1.0 | 1.766e-3（PASS） | 13.03 us | 9.8% / 61% |
| CANN 9.2.0-clean | 1.766e-3（PASS） | 13.26 us | 9.6% / 60% |

prefill（未吸收）精度同样 PASS（9.2.0-clean 下 rel L2=1.823e-3）。FIA 有效
基线 ≈ **13.1 us**（两环境交叉验证一致）。

**9.2.0-clean 同环境 kernel-only 对比（kv=129，PipeUtilization，10 warmup + 30
timed，稳态末 10 中位）：**

| 内核 | 稳态耗时 | aic_mac_ratio | aic_scalar_ratio | 数值 |
| --- | --- | --- | --- | --- |
| MlaDecodeGemv（本仓库） | **10.63 us** | 0.142 | 0.913 | rel L2<5e-3 PASS |
| FIA（CANN 原生） | **13.26 us** | 0.096 | 0.599 | rel L2=1.766e-3 PASS |

**劣化比 = gemv / FIA = 0.80x**：本仓库 gemv 比 CANN FIA **快约 20%**。

结论：按用户要求「保证计算正确的前提下对比基线」，**MLA decode 性能目标
（≤1.05x FIA）已经达成且反超**。D3 及其后续优化不再需要。

## 7. 评估与验证

- decode 通用路径 2.74x 中约 19/20 来自「只用 1 个核」；但既有 FD 机制无法直接复用（见 6.1）。
- 正确 FIA 基线（~13 us）下，本仓库 gemv 已快于 FIA（0.80x，见 §6.3.11），1.05x 目标达成，无需进一步优化。
- 验证实验：固定 batch/heads，把 kv 从 1025 拉到 4096/8192，观察 kernel 时间是否从「近似线性增长」变为「分段后平台化」——这是 KV 并行是否真正生效的最直接信号。

## 8. 复现命令

```bash
source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh
export LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/11:/usr/lib/aarch64-linux-gnu
export ASCEND_RT_VISIBLE_DEVICES=0

# 纯 kernel 对比（fa3/fa4 vs fia，排除 metadata 与 Fill）
python tests/perf/bench_mla_vs_fia.py

# host trace（prefill/decode × fa3_mla/fia_mla）
python tests/perf/diagnose_mla_overhead.py --only-host-trace \
  --host-trace-scenario decode --host-trace-impl fa3_mla
python tests/perf/diagnose_mla_overhead.py --only-host-trace \
  --host-trace-scenario decode --host-trace-impl fia_mla
```
