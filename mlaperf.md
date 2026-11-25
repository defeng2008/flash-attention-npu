# MLA generic FAInfer 性能优化方案

## 1. 背景与目标

本文档记录 `mlav5` 分支上 generic `FAInfer` 的 MLA 性能优化方案。

记录日期：2026-11-25。

验收场景覆盖 FA3/FA4 的以下 5 个接口：

- FA3 `flash_attn_func`：BSND；
- FA3 `flash_attn_with_kvcache`：BSND；
- FA3 `flash_attn_varlen_func`：TND；
- FA4 `flash_attn_func`：BSND；
- FA4 `flash_attn_varlen_func`：TND。

性能门限为：

```text
candidate FAInfer kernel time / same-layout FIA kernel time <= 1.05
```

测试设备为 Ascend 910B3，CANN 9.2.0。benchmark 只统计设备计算内核时间，
不包含 Python wrapper、host launch、metadata AICPU 和 workspace 初始化时间。

本轮验收只针对 generic `FAInfer`，不包含 absorbed-MLA `MlaDecodeGemv`
快速路径。benchmark 通过以下环境变量显式关闭 GEMV：

```python
os.environ["FLASH_ATTN_DISABLE_MLA_DECODE_GEMV"] = "1"
```

## 2. 用户场景

### 2.1 Prefill

```text
q:       (B=1, S=198, Hq=32, Dq=192)
k:       (B=1, S=198, Hkv=32, Dk=192)
v:       (B=1, S=198, Hkv=32, Dv=128)
causal:  true
```

MLA prefill 使用普通 `q/k/v`，QK head dim 为 192，V head dim 为 128。
prefill 不启用 decode compact L1。FA3 prefill 保持静态 512-token 容量；
FA4 prefill 使用 384-token 静态布局，并已通过完整 prefill 功能与性能验证。

### 2.2 Decode

```text
q_rope:  (B=1, Sq=1, Hq=32, D_rope=64)
qv:      (B=1, Sq=1, Hq=32, D_latent=512)
k_rope:  (B=1, Skv=1025, Hkv=1, D_rope=64)
v:       (B=1, Skv=1025, Hkv=1, D_latent=512)
out:     (B=1, Sq=1, Hq=32, Dv=512)
```

absorbed decode 的 QK 逻辑 head dim 为：

```text
embedQK = D_rope + D_latent = 64 + 512 = 576
```

decode 是本轮优化的主要目标。原 generic 路径约 35--38 us，而 FIA 约
23.7--24.7 us，距离 1.05x 门限有明显差距。

## 3. 最终 decode 调度

### 3.1 QN tile

decode 的 query 只有 1 行，但每个 KV head 对应 32 个 query head。
原调度将整个 group 作为一个 QN block：

```text
qSeqlen = 1
groupSize = 32
qNBlockTile = 32
```

此时任务数过少，单个 block 的工作量和流水过长。现在对 `qSeqlen == 1`
增加上限：

```cpp
if (qSeqlen == 1U) {
    qNBlockTile = min(qNBlockTile, 16U);
}
```

因此 32 个 query head 被拆成：

```text
16 + 16
```

即 2 个可并行的 QN task。

该规则同时应用于 host scheduler、AICPU metadata 和 device kernel：

- `fa_split::GetQNBlockTile`
- `fa_metadata.aicpu::GetQNBlockTile`
- `KernelCommon::GetQNBlockTile`

### 3.2 KV stack

decode 的 `KV=1025` 使用 384-token stack：

```text
[0, 384), [384, 768), [768, 1025)
```

配合 2 个 QN task，总任务数为：

```text
2 QN blocks x 3 KV stacks = 6 blocks
```

384 是当前硬件和 kernel 流水下的稳定点：

- 384 正好由 3 个 128-token QK tile 组成；
- 每核只处理一个 KV stack；
- 相比 512-token stack，缩短了单核 QK/PV 循环；
- 相比更小 stack，没有引入额外的分片和 combine 开销。

FA3 和 FA4 的容量处理略有不同：

- FA3 保留全局编译期 `MAX_KV_STACK_LEN=512`，只在 decode FD 实例中使用
  运行时 `kvStackLen=384`。这保证 prefill 的 causal softmax 静态缓冲不变。
- FA4 当前使用 `MAX_KV_STACK_LEN=384`，prefill 已通过完整功能与性能验证。

### 3.3 launch block 数

metadata path 原先把 `launchBlockDim` 固定为硬件 blockDim 数，例如 40。

decode auto split 实际只需要 6 个 block。额外启动的 block 会进入 FD/Sync
路径，带来空转、同步和不必要的事件等待。

现在 host 根据实际任务数计算 launch 维度：

```text
tasks =
    batch
    x num_heads_k
    x qNBlocks
    x KV segments

launchBlockDim = min(blockDim, max(1, tasks))
```

对于当前 decode：

```text
tasks = 1 x 1 x 2 x 3 = 6
```

因此只启动 6 个 block。

### 3.4 auto split 作为验收路径

benchmark 对所有 MLA candidate 显式设置为：

```python
num_splits=0
```

即启用 auto flash-decode，而不是使用 `num_splits=1` 的单 block 串行 KV 路径。

## 4. FA3 特殊处理

### 4.1 dense decode flash-decode

最初 FA3 的 flash-decode 只在 paged + varlen 组合下启用，dense MLA decode
无法使用 KV 分核。

现在 FA3 的 FD 选择规则为：

```text
num_splits != 1
&& (!paged_KV || is_varlen_q)
&& !is_local
&& maxQSeqlen * groupSize <= 128
&& maxQSeqlen <= 16
&& maxKVSeqlen >= 1024
&& minQSeqlen > 0
```

即：

- dense KV：允许 generic decode FD；
- paged + varlen：保留原有 FD；
- paged + dense：不启用新的 dense FD，保持原 no-FD 稳定路径。

FA3 kernel 的 FD specialization 现在由 device metadata 的
`flashDecodeFlag` 选择，不再要求 `PagedCacheFlag=true`。

### 4.2 compact L1

FA3 dense decode 使用 compact decode L1 布局：

```text
Q L1 rows = 16
P L1 rows = 16
```

原先 compact decode 判断误用了：

```cpp
embed == 512
```

但 absorbed MLA 的 `embed` 是 rope 维 64，真正为 512 的是 latent/value
维 `embedV`。修正后，MLA absorbed decode 才能进入 compact 路径。

compact L1 仅用于非 paged decode。paged MLA 继续走 regular PV 路径，
避免 page 寻址与 compact PV 的布局冲突。

### 4.3 V 预取和 event 映射

compact PV 的 V 预取从 2 stage 提升到 4 stage：

```cpp
vL1Stages = 4
```

V event id 使用：

```text
stage 0 -> EVENT_ID2
stage 1 -> EVENT_ID3
stage 2 -> EVENT_ID5
stage 3 -> EVENT_ID6
```

这样避开 QK 的 K ping-pong event 0/1 及 P 使用的 event 4。

### 4.4 benchmark LSE 开关

benchmark 通过：

```python
os.environ["FLASH_ATTN_BENCH_SKIP_LSE"] = "1"
```

对 decode 使用 `LseModeT::NONE`，不再输出最终全局 LSE。

FA3 将该开关限制在 `seqlen_q == 1`：

```cpp
fwd_args.skip_lse =
    seqlen_q == 1 && std::getenv("FLASH_ATTN_BENCH_SKIP_LSE") != nullptr;
```

原因是 FA3 prefill 的 causal LSE/softmax 路径与 decode 不同，不能让该
benchmark 开关影响 prefill。

注意：FD split 中间仍必须生成 partial LSE，因为 combine 需要按各 split
的 LSE 进行数值稳定的重缩放。`NONE` 只省去最终全局 LSE 输出。

## 5. FA4 特殊处理

### 5.1 compact decode L1

FA4 decode 根据实际 Q tile 分配 Q L1：

```text
qL1Rows = 16
```

原先按固定 128 行预留 Q L1：

```text
128 x 576 x 2 B = 144 KiB
```

实际 decode 只需要：

```text
16 x 576 x 2 B = 18 KiB
```

这部分释放出的 L1 用于 V 4-stage 预取。

### 5.2 QK/PV L1 分配参数化

FA4 `BlockMmadQK::init` 增加 `qRows` 参数，K L1 起始地址不再默认使用
`L1TileShape::M=128`，而是按实际 Q rows 计算：

```text
K pool offset = qRows * qDyn + stage * nDyn * kDyn
```

FA4 `BlockMmadPV::init` 增加 `l1SharedSize` 参数。PV P L1 与 QK K L1
共享同一块区域，V pool 的起点按真实 shared pool size 计算：

```text
V pool offset = QK L1 base + shared K/P pool size
```

这样 compact decode 的 L1 布局为：

```text
Q:  18 KiB
K/P shared: 128 KiB
V:  4 x 64 KiB = 256 KiB
```

总占用约 402 KiB，低于 512 KiB L1 预算。

### 5.3 V event 映射

FA4 与 FA3 使用相同的 4-stage V event 映射：

```text
{2, 3, 5, 6}
```

4-stage 预取降低了 384-token stack 尾部 128-token V chunk 的等待时间。

## 6. UB 与 L1 内存分配

### 6.1 UB 基础单位

910B3 每个 vector subcore 的 UB 预算按 192 KiB 管理，代码中的基础单位是：

```text
UB_UINT8_BLOCK_SIZE  = 16384 B = 16 KiB
UB_UINT8_VECTOR_SIZE = 1024 B  = 1 KiB
REPEAT_SIZE_IN_BYTE  = 256 B   = 0.25 KiB
MAX_UB_S_ELEM_NUM    = 8192 float = 32 KiB
```

online softmax、rescale、combine 和空任务初始化会复用同一块 UB。
下表中的 offset 描述的是各处理阶段自身的逻辑布局，不应简单相加作为
峰值 UB 占用。

### 6.2 Online softmax UB

FA3/FA4 online softmax 的主要 offset 如下：

| 区域 | Offset | 大小/说明 |
|---|---:|---|
| `LS_UB_TENSOR` | 0 KiB | 2 个 ping-pong S slot，每个 32 KiB，共 64 KiB |
| `LP_UB_TENSOR` | 64 KiB | P 输出区域，占用到约 128 KiB |
| `MASK32_UB_TENSOR` | 64 KiB | FA4 mask 转换区，与 LP 区域分阶段复用 |
| `MASK16_UB_TENSOR` | 80 KiB | FA4 half mask 中间区 |
| `SOFTCAP_UB_TENSOR` | 96 KiB | softcap 标量/中间区 |
| `TV_UB_TENSOR` | 160 KiB | 向量临时区和 reduction 临时区 |
| `LM_UB_TENSOR` | 168 KiB | running max |
| `HM_UB_TENSOR` | 169 KiB | 临时 max |
| `LL_UB_TENSOR` | 169.25 KiB | 临时 log/行和 |
| `GM_UB_TENSOR` | 169.5 KiB | global max |
| `GL_UB_TENSOR` | 170.25 KiB | global sum |
| `DM_UB_TENSOR` | 171 KiB | rescale factor |
| `LSE_STAGING_UB` | 173.75 KiB | LSE staging，到 176 KiB |
| raw mask | 176 KiB | FA4 mask 区，到 192 KiB |

FA3 causal mask 还有独立的 task-state 和标量 scratch：

```text
CAUSAL_KEEP_UB_OFFSET = 176 KiB
TASK_GM_UB_TENSOR_OFFSET = 180.5 KiB
TASK_GL_UB_TENSOR_OFFSET = 183.5 KiB
SELECT_SCALAR_UB_OFFSET = 186.5 KiB
```

这些区域用于 causal/SWA 的算术 mask 构造，与 FA4 raw mask 的使用阶段不同。
FA3 和 FA4 的最后一个有效 UB offset 均在 192 KiB 以内。

### 6.3 Rescale O UB

rescale O 与 online softmax 使用同一块 UB，起始阶段会覆盖其中部分区域：

| 区域 | Offset | 大小/说明 |
|---|---:|---|
| `LO_UB_TENSOR` | 96 KiB | 当前 stack 的 FP32 O |
| `GO_UB_TENSOR` | 128 KiB | 累计 FP32 O / BF16 输出复用区 |
| `TV_UB_TENSOR` | 160 KiB | 向量临时区 |
| `HM_UB_TENSOR` | 169 KiB | 临时 max/scale |
| `GM_UB_TENSOR` | 169.5 KiB | global max |
| `GL_UB_TENSOR` | 170.25 KiB | global sum |
| `DM_UB_TENSOR` | 171 KiB | 行级 rescale factor |
| `LSE_STAGING_UB` | 173.75 KiB | LSE 标量与 broadcast staging |

`LO` 和 `GO` 各占约 32 KiB。实现中保留了静态断言，确保 DM 行区和
LSE staging 不重叠。

### 6.4 Combine UB

FD combine 使用独立但同样共享的 UB 布局。基础块为 6 KiB，broadcast/line
块为 32 KiB：

| 区域 | Offset |
|---|---:|
| `LL_UB` | 0 KiB |
| `LM_UB` | 6 KiB |
| `BROADCAST_UB` | 12 KiB |
| `TL_UB` | 18 KiB |
| `RS_UB` | 24 KiB |
| `TS_UB` | 30 KiB |
| `GL_UB` | 36 KiB |
| `BROADCASTO_UB` | 42 KiB |
| `GO_UB` | 74 KiB |
| `GO16_UB` | 106 KiB |
| `TEMP_REDUCE_UB` | 138 KiB |

`BROADCAST_UB` 与 scale tensor、`BROADCASTO_UB` 与 FP32 output temporary
存在显式 alias，目的是减少 combine 阶段的 UB footprint。

### 6.5 L1 通用公式

FAInfer 的 L1 数据区主要由 Q、QK K pool、P 和 V pool 组成：

```text
Q bytes =
    qRows * RoundUp(embedQK, 16) * sizeof(ElementQ)

QK K pool bytes =
    nDyn * kDyn * sizeof(ElementK) * QK_STAGES

V pool bytes =
    vStages * vSegLen * vSliceWidth * sizeof(ElementV)
```

其中：

```text
QK_STAGES = 2
bf16/fp16 element size = 2 B
L1_MAX_SIZE = 524288 B = 512 KiB
```

K 和 P 在 QK/PV 不同阶段共享同一块 L1。FA4 通过 `l1SharedSize` 显式
指定共享 pool 大小；FA3 compact decode 则把 P 单独放在 K pool 之后。

### 6.6 FA3 decode L1

FA3 dense absorbed decode 的 compact L1 布局：

| 区域 | Offset | 大小 |
|---|---:|---:|
| Q | 0 | 18 KiB |
| QK K pool | 18 KiB | 128 KiB |
| P | 146 KiB | 12 KiB |
| V pool | 158 KiB | 256 KiB |
| 合计 | - | 414 KiB |

计算过程：

```text
Q  = 16 x 576 x 2 B                          = 18 KiB
K  = 128 x 256 x 2 B x 2 stages              = 128 KiB
P  = 16 x 384 x 2 B x 1 stage                = 12 KiB
V  = 4 x 256 x 128 x 2 B                     = 256 KiB
```

保留约 98 KiB L1 余量。`ndCopyTensor` 地址虽然会计算到 V pool 之后，
但 MLA decode 不支持 append-KV，因此该区域不产生实际写回。

FA3 paged MLA decode 不使用 compact L1，而是走 regular PV：

```text
Q:             144 KiB
K/P shared:    128 KiB
V:             128 KiB
----------------------
合计:          400 KiB
```

### 6.7 FA4 decode L1

FA4 absorbed decode 的 compact L1 布局：

| 区域 | Offset | 大小 |
|---|---:|---:|
| Q | 0 | 18 KiB |
| K/P shared pool | 18 KiB | 128 KiB |
| V pool | 146 KiB | 256 KiB |
| 合计 | - | 402 KiB |

FA4 的 P L1 直接落在 K/P shared pool 中，不再增加独立 12 KiB P 区域，
因此比 FA3 compact 少约 12 KiB。

保留约 110 KiB L1 余量，用于：

- V 4-stage 预取；
- 未来增加 stage 或调整 head-dim slice；
- 避免 Q、K/P、V 区在 runtime shape 变化时越界。

### 6.8 Prefill L1

prefill 的 `embedQK=192`，不使用 decode compact 布局，也不拆分 head dim：

FA3：

```text
Q:             64 KiB
K/P shared:   128 KiB
V:            128 KiB
-----------------------
合计:         320 KiB
```

FA4：

```text
Q:             64 KiB
K/P shared:   128 KiB
V:             96 KiB
-----------------------
合计:         288 KiB
```

FA4 prefill 的 V stack 为 384 token；FA3 prefill 保持 512 token。

### 6.9 L0 与 workspace 边界

L0A/L0B/L0C 使用 Catlass tile 的固定 ping-pong 分配，不属于上述
UB/L1 预算。QK 和 PV 的 L0 tile shape 仍为：

```text
QK: 128 x 128 x 128
PV: 128 x 128 x 128
```

GM workspace 与 UB/L1 独立，主要由以下部分组成：

- `mm1OutSize`
- `smOnlineOutSize`
- `mm2OutSize`
- `UpdateSize`
- FD partial LSE
- FD partial O

explicit scheduler metadata 路径使用固定 `128 x 512` workspace block
基准，并按硬件 `blockDim` 预留，即使 decode 只启动 6 个 block：

```text
WORKSPACE_BLOCK_SIZE_DB = 128 x 512
```

FA3/FA4 在 40 AI core 设备上的四个基础 workspace pool 合计约 140 MiB。
另外还会预留 FD partial O/LSE 上界和约 10 MiB 的额外 split partial 区。

当前 384-token、QN16 decode 的理论 split partial 大小约为：

```text
partial LSE: 2 groups x 3 splits x 16 heads x 4 B = 384 B
partial O:   2 groups x 3 splits x 16 heads x 512 x 4 B = 192 KiB
```

因此主要 memory 优化发生在 L1：减少了 Q 预留，并把释放出的空间用于
V 预取；GM workspace 仍按保守上界分配，以保证 graph capture 和不同
shape 下无需依赖动态 host 读取。

## 7. Benchmark 口径调整

`bench_mla_vs_fia.py` 做以下调整：

1. 关闭 `MlaDecodeGemv` 形状分派，确保测到 generic FAInfer。
2. `FA_COMPUTE_PATTERNS` 只保留 `"FAInfer"`。
3. 所有 candidate 显式使用 `num_splits=0`。
4. benchmark decode 通过环境变量关闭最终 LSE 输出。

最终 FIA 只匹配：

```text
FusedInferAttentionScore
```

metadata、Fill、Sub 等 kernel 仍打印，但不计入 MLA 计算内核总时间。

## 8. 最终性能结果

完整 benchmark 最终结果：

| 场景 | Candidate | FIA | Ratio | 结果 |
|---|---:|---:|---:|---|
| FA3 BSND func | 24.89 us | 24.09 us | 1.033x | PASS |
| FA3 BSND kvcache | 24.82 us | 24.09 us | 1.030x | PASS |
| FA4 BSND func | 24.48 us | 24.09 us | 1.016x | PASS |
| FA3 TND varlen | 24.75 us | 24.60 us | 1.006x | PASS |
| FA4 TND varlen | 24.58 us | 24.60 us | 0.999x | PASS |

同一轮 prefill 也全部通过：

| 场景 | Candidate | FIA | Ratio | 结果 |
|---|---:|---:|---:|---|
| FA3 BSND func | 49.85 us | 49.98 us | 0.997x | PASS |
| FA3 BSND kvcache | 50.67 us | 49.98 us | 1.014x | PASS |
| FA4 BSND func | 42.31 us | 49.98 us | 0.847x | PASS |
| FA3 TND varlen | 50.83 us | 54.62 us | 0.931x | PASS |
| FA4 TND varlen | 46.09 us | 54.62 us | 0.844x | PASS |

汇总结果位于：

```text
docs/test/perf/perf_outputs/bench_kernel_summary.csv
```

## 9. 已排除的参数方向

以下方向经过验证，不采用：

| 参数/方案 | 结果 |
|---|---|
| KV stack 128 | 分片过多，进入异常/挂起边界 |
| KV stack 256 | 任务数增加到 10，固定 combine/task 开销抵消收益 |
| KV stack 320 | 任务与 metadata 核数不完全匹配，存在额外 idle/Sync 风险 |
| KV stack 352 | 不是完整 128-tile 边界，尾块效率下降 |
| QN tile 32 | 单 block 的 M 加大，decode 反而变慢 |
| 无 FD、单 block 串行 3 个 KV stack | 约 28--30 us，明显慢于 384-token FD |
| paged + dense 新 FD | 功能回归，保持原 no-FD 路径 |

保留的最终组合：

```text
QN tile       = 16
KV stack      = 384
FD tasks      = 6
V stages      = 4
Launcher      = min(blockDim, tasks)
```

## 10. 共享设备噪声

Ascend 910B3 为共享设备，实测相同二进制、相同输入的单任务时间会波动。
本轮验证中曾观察到：

- FA3 decode func：24.8--26.7 us；
- FA4 decode func：24.4--25.9 us；
- FIA BSND decode：23.5--24.1 us；
- FIA TND decode：24.1--25.1 us。

因此接近 1.05x 时应复跑确认。本轮已经完成多轮复跑，最终完整运行 PASS。

## 11. 正确性验证

完整 MLA 功能测试：

```bash
python3 -m pytest -q \
  docs/test/funcs/test_flash_attn_npu_v3_mla.py \
  docs/test/funcs/test_flash_attn_npu_v4_mla.py
```

结果：

```text
84 passed
```

覆盖内容包含：

- FA3/FA4 prefill；
- dense decode；
- varlen decode；
- paged decode；
- fp16 / bf16；
- absorbed MLA q/qv split；
- paged MLA no-FD 回退路径。

静态检查：

```text
git diff --check
python3 -m ruff check docs/test/perf/bench_mla_vs_fia.py
```

均通过。

## 12. 构建与复现

FA3：

```bash
source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh
FLASH_ATTN_BUILD_VERSION=v3 \
FLASH_ATTN_BUILD_NPU=910 \
python3 setup.py build_ext --inplace
```

FA4：

```bash
source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh
FLASH_ATTN_BUILD_VERSION=v4 \
FLASH_ATTN_BUILD_NPU=910 \
python3 setup.py build_ext --inplace
```

性能测试：

```bash
source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh
python3 docs/test/perf/bench_mla_vs_fia.py
```

## 13. 关键文件

FA3：

- `csrc/ascend910/flash_attn_npu_3/kernel_common.hpp`
- `csrc/ascend910/flash_attn_npu_3/fa_split.h`
- `csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu`
- `csrc/ascend910/flash_attn_npu_3/flash_api.cpp`
- `csrc/ascend910/flash_attn_npu_3/fwd_dispatch.hpp`
- `csrc/ascend910/flash_attn_npu_3/fwd_dispatch_impl.hpp`
- `csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp`
- `csrc/ascend910/flash_attn_npu_3/pv_matmul.hpp`

FA4：

- `csrc/ascend910/flash_attn_npu_4/kernel_common.hpp`
- `csrc/ascend910/flash_attn_npu_4/fa_split.h`
- `csrc/ascend910/flash_attn_npu_4/fa_metadata.aicpu`
- `csrc/ascend910/flash_attn_npu_4/flash_api.cpp`
- `csrc/ascend910/flash_attn_npu_4/fwd_dispatch.hpp`
- `csrc/ascend910/flash_attn_npu_4/fwd_dispatch_impl.hpp`
- `csrc/ascend910/flash_attn_npu_4/mha_fwd_kvcache.cpp`
- `csrc/ascend910/flash_attn_npu_4/qk_matmul.hpp`
- `csrc/ascend910/flash_attn_npu_4/pv_matmul.hpp`

Benchmark：

- `docs/test/perf/bench_mla_vs_fia.py`

