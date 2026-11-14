# MLA decode GEMV kernel：同步机制走读与 host 保护排查

- 日期：2026-11-11
- 对象：[mla_decode_gemv.hpp](../../csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp)（FA4 的 `flash_attn_npu_4/` 同名文件同构）
- 性质：**静态代码走读结论**，未做上机实测；所有行号对应当前工作区快照
- 相关文档：[mla_decode_gemv_design.md](mla_decode_gemv_design.md)（设计权威口径）、[mla_l1_layout.md](mla_l1_layout.md)

本文回答四类问题：调用链、段间软流水、HardEvent 同步体系（含 prelaunch 配平与 L840 归还点位）、容量与分核（含 host 保护缺失）。

---

## 1. 调用链

```
Python flash_attn_*
  → mha_fwd
  → launch_fwd<kLayout>
  → launch_fwd_dtype            fwd_dispatch_impl.hpp L67
  → SplitFuse::MlaDecodeGemv<<<launchBlockDim, nullptr, aclStream>>>
                                fwd_dispatch_impl.hpp L100-105（命中即 return，不进 FAInfer）
  → KernelType kernel(params); kernel(params);
                                mla_decode_gemv.hpp L1101
  → MlaDecodeGemv::operator()   mla_decode_gemv.hpp L571
```

选路闸门 `IsMlaDecodeGemvShape`（[flash_api.cpp L102-113](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/flash_api.cpp#L102-L113)）在两个位置各判一次，两处必须一致：

1. **早闸门** L518：预处理 `scheduler_metadata`，命中则跳过 metadata 的 host 重路由；
2. **主 tiling** L725：分核、workspace、`fwd_args.mlaDecodeGemv` 赋值。

命中后 `mlaDecodeGemv = true`、`flashDecodeFlag = false`，走 `fa_split::splitKvForGemvDecode`，`launchBlockDim = needCoreNum`（无任务的核不发射，L756）。

---

## 2. 段间软流水

Cube(AIC) 与 Vector(AIV) 在同一段号上错位一段工作，`STACK_SLOTS = 3`（[L567](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L567)）三槽滚动，workspace 与 UB 均按 `slot = seg % STACK_SLOTS` 双/三缓冲。

```
段号          seg0      seg1      seg2      seg3      ...
AIC  MM1(QK)  ███       ███       ███
AIC  MM2(PV)            ███       ███       ███
AIV softmax             ███       ███       ███     ← 消费 MM1 的 S（smOnline 区）
AIV rescale                       ███       ███     ← 消费 MM2 的 OTmp，持久 O 在线 rescale
```

三条跨核握手 flag（`Arch::CrossCoreFlag`，[L1025-1027](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L1025-L1027)）：

| flag | Set 端 | 含义 |
| --- | --- | --- |
| `qkReady` | AIC `PIPE_FIX`，L874 / L907 | QK 结果（S）已写入 GM 槽位，AIV softmax 可读 |
| `softmaxReady` | AIV `PIPE_MTE3`，L881 / L921 | P 已写出，AIC 的 MM2 可读 |
| `pvReady` | AIC `PIPE_FIX`，L913 | OTmp 已写出，AIV rescale 可读 |

AIV 侧对应 `Arch::CrossCoreWaitFlag(pvReady)`（L491、L514）。

---

## 3. HardEvent 同步体系

### 3.1 命名规则

`HardEvent::X_Y` 读作「**X 干完 → Y 才能干**」：

- `X` = 生产者 pipe，在 X 的指令队列里 `SetFlag`；
- `Y` = 消费者 pipe，在 Y 的指令队列里 `WaitFlag`。

用到的 pipe：`MTE1`(L1→L0)、`MTE2`(GM→L1/UB，搬入)、`MTE3`(UB→GM，搬出)、`V`(向量计算)、`M`(Cube)、`FIX`(FixPipe)。

于是：

- `MTE3_MTE2` / `V_MTE2` / `MTE1_MTE2` = **WAR**（读完了才准覆写）；
- `MTE2_V` / `MTE2_MTE1` = **RAW**（搬进来了才准读）。

### 3.2 EVENT_ID

`EVENT_ID0..7` 是**同一类 HardEvent 下的独立信道编号**，只有 8 个，**不能"创建一百个"**；不同 HardEvent 类型下的同号 ID 完全独立、互不干扰。同类同 ID 的 `SetFlag` / `WaitFlag` 必须成对。

一个常见混淆点：`SetFlag<MTE3_V>(pingpongFlag)`（[online_softmax.hpp L587](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/online_softmax.hpp#L587)）里的 `pingpongFlag` 是**运行期变量**（`= softmaxPingPongFlag % 2`，取 0/1），而 `EVENT_ID0` 是**常量名**；`SetFlag` 第二形参接受"号"，常量或变量皆可。softmax 实际用 0/1 两号，恰好落在 prelaunch 集合 `{0,1,2,4,6,7}` 内。

### 3.3 prelaunch / drain 配平

kernel 上电先把 body 会 `Wait` 的号全部 `Set` 一遍，否则首段迭代会等一个不存在的生产者而**死锁**；尾部再用**完全相同**的集合 `WaitFlag` 收回，否则配平错误挂死。

- AIC prelaunch [L659-675](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L659-L675)：
  `SetFlag<M_MTE1>{0..3}`、`SetFlag<FIX_M>{0,1}`、`SetFlag<MTE1_MTE2>{0,1,3,4,5,6}`；
  AIC drain [L934-945](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L934-L945)。
- AIV prelaunch [L699-714](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L699-L714)：
  `SetFlag<MTE3_V>{0,1,2,4,6,7}`、`SetFlag<MTE3_MTE2>{0,2,3,4,5,6}`、`SetFlag<V_MTE2>{0,1,2,4}`；
  AIV drain [L947-964](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L947-L964)。

**裁剪判据**（L659-666 注释所记）：body 里"有 `Wait` 无 `Set`"的号 → 必须保留；"无人消费"的号 → prelaunch 与尾部 drain **成对删除**，账本保持平衡。因此 `MTE1_MTE2` 只铺 `{0,1,3,4,5,6}`，没有 2 和 7。

**硬约束**：prelaunch 阶段每个 `(事件类型, ID)` 组合**只允许 `SetFlag` 恰好一次**；多铺一个 → 尾部死等，少铺一个 → 线内首次死等。

### 3.4 相邻同 ID Set/Wait 构成的"刷干屏障"

[qk_matmul.hpp L187-188](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L187-L188) 是「`SetFlag<MTE2_MTE1>(EVENT_ID3)` 紧接着 `WaitFlag<MTE2_MTE1>(EVENT_ID3)`」。两个 ID **必须相同**（配对前提），关键不在 ID，而在两条指令**分属不同 pipe 队列**：

- `SetFlag` 排在 **MTE2** 队列末尾；
- `WaitFlag` 排在 **MTE1** 队列。

效果是强制 MTE1 等待 MTE2 队列中**所有前置搬运指令**执行完毕，即一个跨队列的**刷干屏障**，保证 Q 已完整落入 L1 之后 MTE1 才开始读，避免读到未就绪的旧数据（`loadQGM` 的 A 通道 [L373-374](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L373-L374) 不再单独 `WaitFlag<MTE2_MTE1>`，正是靠这条屏障兜底）。

反证：若 Wait 写成别的 ID（如 `EVENT_ID4`），则永远等不到，死锁。

### 3.5 三个具体通道实例

| 通道 | 场景 | 代码位置 |
| --- | --- | --- |
| `MTE3_MTE2` | 同一 UB 地址「先搬出、后搬入」的次序锁（WAR）。典型：rescale 把 `gUpdate` 写出去后，下段 MTE2 又要读回同一 UB 地址 | [rescale_o.hpp L334/345/389/392](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/rescale_o.hpp)、本 kernel `EVENT_ID6`：`WaitFlag` [L304](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L304)/[L316](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L316)、`SetFlag` [L351](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L351)/[L454](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L454) |
| `MTE3_V` | 搬出完成后 Vector 才能覆写这块 UB | online_softmax L569/L587；本 kernel 末段写 `gO`/`gLse` 用 `EVENT_ID0/4` |
| `MTE1_MTE2` | L1 上的 WAR：L1 槽被 MTE1 读完，MTE2 才能搬下一份进来。本 kernel 用它给 K/P 的 2 槽 ring 与 Q 区做节流 | 消费点 [qk_matmul.hpp L510](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L510)（`l1KPingPongFlag`）、归还点 [L543](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L543) |

---

## 4. 专项：L840 的 `SetFlag<MTE1_MTE2>(EVENT_ID3)`

代码：

```cpp
blockMmadQK.loadQGM(gQv[gmOffsetQv], layoutQ, rowNum, qNBlockSize,
                    qvHeadStride, strideQv);                    // L836
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);      // L840
```

（[L836-841](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L836-L841)，注释在 L838-839）

### 4.1 它是什么

`loadQGM` 内部 [L172](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L172) 先 `WaitFlag<MTE1_MTE2>(EVENT_ID3)` 借走一枚令牌，但内部**没有**归属于 `MTE1_MTE2=(X=MTE1)` 的 `SetFlag`（L187 的 Set 属于另一条通道 `MTE2_MTE1`）。因此 L840 是**外层归还**：把 `loadQGM` 消耗掉的那枚令牌补回，使 prelaunch 铺的旗标与尾部 drain 保持配平。

它**不是** L1→L0 搬运指令，只是记账。

### 4.2 提前归还有没有风险

结论：**在当前 gemv 调度下没有真实风险，但已退化为纯记账**。

- 每核只跑**一个** task（[L761](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L761) `const coreNode& coreTask = tilingData->coreInfo[coreIdx];`，函数内唯一循环是段循环 L884），**不存在"下一个 task 要覆写 Q 区"的前提**；
- Q 区与 K/P 池在 L1 中**不重叠**（[qk_matmul.hpp L119-126](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L119-L126)：Q 在 `[0, qRows*qDyn*sizeof)`，K 池从 `qRows*qDyn*sizeof` 起），Q 只写一次。

**若后续改成单核多 task（grid-stride）调度，L840 必须移到该 task 内 Q 全部消费完之后**，否则触发 L1 Q 区覆写的真实 WAR hazard。

### 4.3 通用路径对照

通用 FAInfer 的"正确归还位置"在 [mha_fwd_kvcache.cpp L1015-1016](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp#L1015-L1016)：`if (kvSIdx == kvEnd-1) SetFlag<MTE1_MTE2>(EVENT_ID3);` —— 晚于 Q 的所有 MTE1 消费。该路径有 grid-stride 任务循环（[L520](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp#L520)），所以必须这么做。设计文档 [mla_decode_gemv_design.md:254](mla_decode_gemv_design.md) 也自述 gemv 侧为"记账"。

---

## 5. 容量与规模

### 5.1 L1 为什么装得下（H = 32 时）

L1 每 AIC 512 KB，布局（详见 [mla_l1_layout.md](mla_l1_layout.md) §3.1、design §4.1）：

| 区间 | 内容 | 大小 |
| --- | --- | --- |
| `[0, 576H)` | Q（单缓冲、整块装） | `H × 576 × 2 B`，H=32 → 36 KB |
| `[576H, 576H+128K)` | K ring 2 槽 / P staging 2 槽（**K 与 P 共址**，省 128 KB） | 2 × 64 KB |
| `[576H+128K, 576H+256K)` | V ring 2 槽 | 2 × 64 KB |

- Q 侧：`seqlenQ == 1`，Q 只有 `H × 576 × 2 B`，H=32 才 36 KB，**一次性全部装进 L1**。
- K/V 侧：走 2 槽 ping-pong 滚动，与序列长度无关。
- 合计恒定 `576H + 256 KB`，H=32 → **292 KB**，余量 ~220 KB，**不随 batch size B 增大而变化**（每核只装当前 task 的工作集）。

### 5.2 head 上界

`1152H + 256 KB ≤ 512 KB ⇒ H ≤ 227`（余量随 H 线性收缩）。标准 MLA 配置 H=128 → 400 KB，安全。

- `H > 128`：M 维按 128 走 L0 tile 循环（`mL0Loop = CeilDiv(mActual, L0TileShape::M)`，[qk_matmul.hpp L357-359](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L357-L359)），**算法层面无需改代码**；
- `128 < H ≤ 227` 可跑；`H ≥ 228` 需要新方案（head 维切核，或 Q 分块滚动）。

### 5.3 B 上界与 task 截断（零保护）

真实阈值不是固定常数，而是运行时核数 `blockDim = GetCoreNumAic()`（[flash_api.cpp L223](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/flash_api.cpp#L223)）。B 一大，会**先退化、再截断**。

**第一步退化：每 batch 分到的核数被压缩。**

[fa_split.h L578-579](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L578-L579)：

```cpp
const uint32_t maxSplitsPerBatch =
    (ctx.blockDim / batchWithWork) ? (ctx.blockDim / batchWithWork) : 1U;
```

- `batchWithWork ≤ blockDim`：每 batch 可占 `blockDim / batchWithWork` 个 split，KV 维继续按 `SEG_TARGET = 4` 切，总 task ≈ `batchWithWork × nsplitPerBatch ≤ blockDim`，核基本填满；
- `batchWithWork > blockDim`：整数除法归零 → `maxSplitsPerBatch = 1` → 每 batch 只剩 **1 个 task**，KV 维**完全不切**（[L602-604](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L602-L604) 把 `nsplitPerBatch` clamp 到 1）。

这不是"主动放弃分核"，而是每 batch 已经占满一个核、没有余量再切 KV。

**第二步截断：超出 `MAX_NODE_NUM = 25` 的 task 被静默丢弃。**

`coreInfo` / `splitInfo` 数组硬编码 25（[tilingdata.h L69-70](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/tilingdata.h#L69-L70)），写入侧有静默保护：

```cpp
if (coreIdx < MAX_NODE_NUM) { tiling->coreInfo[coreIdx] = ...; }   // fa_split.h L640
coreIdx++;
...
const uint32_t needCoreNum = FaMin(coreIdx, MAX_NODE_NUM);          // fa_split.h L663
```

而 kernel 侧是**一核一 task、没有 grid-stride 循环**（[mla_decode_gemv.hpp L761](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L761) 只取 `coreInfo[coreIdx]`，函数内唯一循环是段循环 L884）。因此 `batchWithWork > 25`（更准确说是 > 实际发射核数）时：

- 第 26 个之后的 batch **不写进 `coreInfo`**，也不会被任何核认领；
- `needCoreNum = 25` → `launchBlockDim = 25`，只发射 25 个 block；
- 这些 batch 的输出**静默保持未初始化**，无任何报错。

安全区只有 `batchWithWork ≤ GetCoreNumAic()`（且该值 ≤ 25）。

**其余规模结论仍然成立**：L1 占用不随 B 变化（每核只装当前 task 工作集）；workspace 按 `blockDim` 计（[flash_api.cpp L772-786](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/flash_api.cpp#L772-L786)），**与 qHeads 无关、不随 B 膨胀**；总时间 ≈ `task 数 / 核数`。

**修法**：本仓业务 B **不受限**（可达上千），因此**不能**用 host 加界回落（限制 B 等于让大 B 场景退回慢路径），必须让 kernel 支持"一核多 task"。又因为 task 描述无法用定长数组承载，需要由 kernel 就地重算 —— 完整方案见 **§9**。

**注意**：一旦单核跑多 task，§4 里 L840 那笔"提前归还"就会从记账退化为**真实 WAR bug**，必须同时把 `SetFlag<MTE1_MTE2>(EVENT_ID3)` 移到 Q 全部消费完之后。

未实测项：同核上多个 block 的（workspace / Q 区）是否严格不交叠。

---

## 6. 分核机制

入口 `fa_split::splitKvForGemvDecode`（[fa_split.h L528](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L528)）：

```
task = (bIdx, [segStart, segEnd))      # 一个 batch 上的一段连续 128-token KV 段
segCount = ceil(kvLen / 128)           # 一核一任务
```

1. `batchWithWork`：统计 `kvLen > 0` 的 batch；为 0 → `needCoreNum = 0` 返回。
2. 每 batch split 预算 `nsplitPerBatch`：`num_splits > 1` 用显式值；否则 `ceil(maxSegCount / SEG_TARGET)`，`SEG_TARGET = 4`（[L577](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L577)）；上限 `blockDim / batchWithWork`。
3. `blocksPerSplit = ceil(segCount / nsplitPerBatch)`，`actualSplits = ceil(segCount / blocksPerSplit)`。
4. `coreInfo[i] = {b, b, segStart, segEnd}` + partial 基址（按 batch 递增，group-major `[batch][split][head]`）。
5. `needCoreNum = min(coreIdx, MAX_NODE_NUM)`，`MAX_NODE_NUM = 25`（[L531](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L531)/[L663](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L663)）。

**只切 KV 维**：head 维 `0 → qHeads`（[L629](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L629)）与 Q 维 `0 → 1` **完全不切**。

收尾：任一 batch 被切成多片（`anySplit`）→ `fdMode`，各核写 partial，末尾 `SyncAll`（[L971](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L971)）+ `CombineScale` 归约；否则 direct 直写最终 O/LSE。

---

## 7. host 保护排查（head 维度已设防，batch 维度零保护）

对 `IsMlaDecodeGemvShape` 以及 gemv 全链路做了穷尽排查。**head 维度现已落地上界 `numHeads <= 32`**（实现见本节末），**batch 维度仍是零设防**：

| 位置 | 事实 |
| --- | --- |
| 闸门 [flash_api.cpp L101-107](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp#L101-L107) | 已设上界 `numHeads <= 32`，超出即回落通用 FAInfer |
| 调用点 L518 / L725 | 无额外 head 校验 |
| `get_scheduler_metadata`（[L901-902](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp#L901-L902)） | 曾内联第三份同义 MLA 判定以放行 `num_splits > 1`，**已回退删除**；现仍只允许 paged + varlen-q 开 `num_splits > 1` |
| gemv host 分支 L724-820 | 唯一 `TORCH_CHECK` 在 L728，只约束 `num_splits` |
| `l1QKSize` / `576H` | 全仓 grep 只命中 kernel 与文档，**host 完全未出现** |
| workspace [L772-786](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/flash_api.cpp#L772-L786) | 只按 `blockDim` 算，与 qHeads 无关 |
| `splitKvForGemvDecode` | 只有 `MAX_NODE_NUM = 25` 与 `headEndIdx = qHeads`，**无 head / L1 断言** |
| kernel `static_assert` [L179-181](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L179-L181) | 只保 **UB 192 KB**，与 L1 无关 |
| batch 维度（B 上界） | 闸门与 gemv 分支**完全未提及 batch 数**；`splitKvForGemvDecode` 只把 `batchWithWork` 当除数用（[fa_split.h L578-579](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L578-L579)），无任何上界断言 |
| `coreInfo` / `splitInfo` 容量 | 硬编码 `[25]`（[tilingdata.h L69-70](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/tilingdata.h#L69-L70)，配 `MAX_NODE_NUM = 25`）；超额 task 由 `if (coreIdx < MAX_NODE_NUM)`（[fa_split.h L640](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L640)）静默跳过，不报错、不回退 |
| FA4 对照 | [flash_attn_npu_4/flash_api.cpp L128](../../csrc/ascend910/flash_attn_npu_4/flash_api.cpp#L128)（`IsMlaDecodeGemvShape`）与 `fa_split.h` 逻辑一致，已同步设上界 `num_heads <= 32` |

改前此处唯一的"兜底"只是设计文档里的一句话：「闸门只设下限 `numHeads >= 16`，上限由 L1 预算隐式约束」——即只写在文档、**未落地为任何 host 校验**。**该缺口已闭合**：闸门内现为显式上界 `numHeads <= 32`。

改前 `H ≥ 228` 走 gemv 时：host 不拦、kernel 无 `static_assert`，`blockMmadQK.init(..., l1BufAddrStart = 0)`（[L694](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L694)）与 `ndCopyL1Offset` 会**直接越界 L1**，属于静默越界风险。**上界落地后 `H > 32` 的 shape 不再进入 gemv 分支，该风险已消除**（32 比 L1 预算允许的 227 更保守）。

### 已落地的设防（head 维度）

在 `IsMlaDecodeGemvShape` 返回条件中加入 head 数上界，超预算 shape **回落到通用 FAInfer 路径**：

```cpp
// v3: csrc/ascend910/flash_attn_npu_3/flash_api.cpp
return hasQv && (numHeads <= 32) && (numHeadsK == 1) && (headSizeV == 512) && (headSizeOg + headSizeV == 576) &&
       !pagedKv && !isLocal && !isCausal && (seqlenQ == 1);
```

- 取 **32** 而非 L1 预算允许的 227：专用 kernel 的分核与持久区几何按 `qHeads = 32` 验证（每子核 16 行），32 是已实测覆盖的边界。
- FA3 / FA4 两份 `flash_api.cpp` 已同步（v4 为 `num_heads <= 32`）。
- **上界只存在于上面的闸门里**：`get_scheduler_metadata` 曾内联一份同义 MLA 判定（`num_heads_q >= 16`）以便放行 `num_splits > 1`，该判定**已回退删除**。因此单独调用这个 API 时，MLA decode 显式传 `num_splits > 1` 会被拒（仍要求 paged + varlen-q）；`mha_fwd` 内 `|| mlaDecodeGemv` 的放行未变，两条路径在这一点上口径不一致。

**batch 维度则不采用 host 加界**：B 由业务决定、不受限（可达上千），限制 B 等于让大 B 场景全部退回慢路径，业务上不可接受。正确修法是让 kernel 支持"一核多 task"，且 task 描述必须**就地重算**而不是落表（定长数组装不下）——完整方案见 **§9**。

---

## 8. 关键代码位置索引

| 主题 | 位置 |
| --- | --- |
| 选路闸门 | [flash_api.cpp L102-113](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/flash_api.cpp#L102-L113) |
| gemv 分发 | [fwd_dispatch_impl.hpp L100-105](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fwd_dispatch_impl.hpp#L100-L105) |
| kernel 入口 | [mla_decode_gemv.hpp L1037](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L1037) / [L1101](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L1101) |
| AIC/AIV prelaunch | [L659-675](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L659-L675) / [L699-714](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L699-L714) |
| 尾部 drain | [L934-945](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L934-L945) / [L947-964](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L947-L964) |
| L840 记账归还 | [L836-841](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L836-L841) |
| `loadQGM` 内部 | [qk_matmul.hpp L164-189](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L164-L189) |
| L1 A/B/C 初始化 | [qk_matmul.hpp L119-126](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L119-L126) |
| 分核 / B 退化与截断 | [fa_split.h L528-677](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/fa_split.h#L528-L677)（关键：L578-579 每 batch 预算、L602-604 clamp、L640 写入保护、L663 needCoreNum） |
| task 数组容量 | [tilingdata.h L69-70](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/tilingdata.h#L69-L70) `coreInfo[25]` / `splitInfo[25]` |
| kernel 一核一 task | [mla_decode_gemv.hpp L761](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L761) |
| 运行时核数 | [flash_api.cpp L223](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/flash_api.cpp#L223) `GetCoreNumAic()` |
| 通用路径对照（正确归还点） | [mha_fwd_kvcache.cpp L1015-1016](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp#L1015-L1016) |

---

## 9. 修复方向：B 不受限下的 kernel 重算方案

> 本节为**方案说明，尚未实现**。适用前提：业务 B 完全不受限，可能到上千。

### 9.1 硬约束与结论

- B 由业务决定，**不能限制**：限制 B 等于让大 B 场景全部退回通用 FAInfer（性能回归），或直接拒绝服务；
- 现状 B 超过可用核数时是**功能错误**，不是性能问题：超出 `MAX_NODE_NUM = 25` 的 task 被 `if (coreIdx < MAX_NODE_NUM)` 静默跳过，对应 batch 输出保持未初始化；
- `coreInfo[25]` / `splitInfo[25]` 是**定长数组**（[tilingdata.h L69-70](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/tilingdata.h#L69-L70)），而 task 数 = `Σ_b actualSplits_b ≥ B`。B 上千时**定长数组从根本上装不下**（按 `coreNode` 48 B / `splitNode` 40 B 估算，1000 个 batch 会让 tiling 结构体膨胀近 90 KB），"扩容数组"不可行。

**结论：放弃"host 生成完整 task 表"的做法，改为 host 只下沉少量标量参数，由 kernel 用与 host 完全相同的规则就地重算 task 描述。**

### 9.2 总体思路

| 项 | 现状 | 改造后 |
| --- | --- | --- |
| task 描述 | host 逐条写 `coreInfo[coreIdx]` | kernel 由 `taskIdx` 重算 `(b, s, segStart, segEnd)` |
| 每 batch split 数 | host 写 `splitInfo[splitIdx]` | kernel 用同一规则重算 |
| 核内分配 | `coreIdx` 即 `taskIdx`（一核一 task） | 核 k 认领**连续** task 区间 `[start_k, end_k)` |
| tiling 输入 | 25×48 B + 25×40 B 两张表 | 3 个标量：`totalTaskNum` / `nsplitPerBatch` / `batchWithWork` |

kernel 已具备重算所需的全部原始数据：per-batch KV 长度可直接读 `gActualKvseqlen`（BSND/BNSD）或 `gSeqUsedKv` / prefix delta（TND），见 [mla_decode_gemv.hpp L768-777](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L768-L777)。

### 9.3 host 侧改动（`fa_split.h`）

`splitKvForGemvDecode` 只保留**规则统计**，不再落表：

1. 与现在一致地算出 `batchWithWork`、`maxSplitsPerBatch = blockDim / batchWithWork`、`nsplitPerBatch`（`fa_split.h` L556-604 逻辑不动）；
2. 逐 batch 累加 `totalTaskNum` 与 `totalSplitNodeNum(= batchWithWork)`、`lseTaskOffset`、`oTaskOffset`（L611-661 的累加部分保留，删掉写 `coreInfo` / `splitInfo` 的部分）；
3. `needCoreNum = min(totalTaskNum, blockDim)` —— **不再 `min(..., 25)`**；
4. 只写标量：`totalTaskNum` / `nsplitPerBatch` / `batchWithWork` / partial 总大小 / `anySplit`。

两个注意点：

- **`nsplitPerBatch` 必须 host 算好下沉，不能留给 kernel 重算**。它依赖 `maxSplitsPerBatch = blockDim / batchWithWork`，而 host 的 `blockDim` 是 `GetCoreNumAic()`（[flash_api.cpp L223](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/flash_api.cpp#L223)），kernel 侧的 `GetBlockNum()` 返回的却是发射核数 `min(taskNum, blockDim)`，两者不相等，重算必然漂移。
- `FAInferTilingData` 需新增 `nsplitPerBatch`（uint32）；`coreInfo` / `splitInfo` 在 gemv 路径上不再被写入——但通用 FAInfer 路径仍在用这两个数组，**不能直接删除**，只能停止在 gemv 路径写入（或按路径条件编译）。

### 9.4 kernel 侧改动（`mla_decode_gemv.hpp`）

**分配（每核一段连续 task）**

```
perCore = ceil(totalTaskNum / GetBlockNum())
start   = coreIdx * perCore
end     = min(start + perCore, totalTaskNum)
```

**定位起点（每核一次，O(batchSize)）**

```
uint32_t b = 0, t = 0;
int64_t lseBase = 0, oBase = 0;              // partial 基址的 running 值
for (; b < batchSize; b++) {
    segCount_b = CeilDiv(kvLenOf(b), 128);   // kvLenOf 口径见 L768-777
    if (segCount_b == 0) continue;
    blocksPerSplit = CeilDiv(segCount_b, nsplitPerBatch);
    if (blocksPerSplit == 0) blocksPerSplit = 1;      // 与 host L618-620 兜底一致
    actualSplits   = CeilDiv(segCount_b, blocksPerSplit);
    if (t + actualSplits > start) break;              // start 落在本 batch
    t += actualSplits;
    lseBase += actualSplits * qHeads;
    oBase   += actualSplits * qHeads * headSizeV;
}
s = start - t;
```

**核内 task 循环（顺序推进）**：每轮用 `(b, s, segStart = s*blocksPerSplit, segEnd = min(segStart+blocksPerSplit, segCount_b))` 组装出原来 `coreTask` 提供的那几个字段，再走现有段循环；`s++`，`s == actualSplits` 时进入下一个 `segCount > 0` 的 batch 并同步累加 `lseBase / oBase`。

**一致性要求**：这套规则必须与 host `splitKvForGemvDecode` **逐字一致**，包括 `CeilDiv` 语义、`blocksPerSplit == 0 → 1` 兜底、`kvLen = seqlens_k + kvNewSeqlen` 口径、以及 b 递增的遍历顺序。

### 9.5 必须同步修的三处

1. **L840 的归还点（最高优先）**：把 `SetFlag<MTE1_MTE2>(EVENT_ID3)` 从 `loadQGM` 之后移到**该 task 的段循环结束之后**。原因见 §4：多 task 后 task N+1 的 `loadQGM`（MTE2 写 Q 区）会异步抢在 task N 最后一次 `copyL1ToL0A`（MTE1 读 Q 区）之前完成——标量只是把任务投进 MTE1 队列、并不等它跑完。把 Set 放在 MTE1 队列末尾（段循环后）才能挡住这次覆写；借/还各一枚，与 prelaunch 铺的那一枚配平。参照通用路径 [mha_fwd_kvcache.cpp L1015-1016](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp#L1015-L1016)。
2. **CombineScale 的输入**（[L967-989](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp#L967-L989)）：它现在依赖 host 填的 `splitInfo[]`（per-batch `splitNum` / `lseTaskOffset` / `oTaskOffset`）。B 上千时这张表同样装不下，必须用 9.4 的 running 累加在 kernel 内派生。
3. **prelaunch / 尾部 drain / `SyncAll`**：保持"每核一次"，留在 task 循环之外；不要把段循环的同步搬进 task 循环，否则令牌配平崩掉。

### 9.6 标量开销

- **连续块分配**：每核一次 O(B) 扫描 → 总开销 O(B × 核数)。B = 1000、25 核 → 约 2.5 万次整数迭代，可忽略；
- 若沿用 stride 分配（`t += numCores`）：每个 task 都要 O(B) 重定位 → 总 O(B²)，B = 1000 时约 1e6 次，仍可用但明显浪费。**推荐连续块**。

### 9.7 一致性风险（最大隐患）

规则将被实现两遍（host CPU 一份、kernel AIC 一份），任何一处漂移都会导致"host 的核数与 kernel 重算不一致" → 部分 task 无人认领或多算，且**没有报错**。

- 缓解 1：把 `blocksPerSplit / actualSplits / offset 累加` 抽成一个共享的纯函数头文件，host 与 kernel 同时 include（host 侧普通 inline、kernel 侧 `__aicore__ inline`，用宏区分）；
- 缓解 2：至少在 kernel 里加校验——重算出的 task 总数与 tiling 的 `totalTaskNum` 不一致时直接跳过并打点，便于定位。

### 9.8 回归验证要点

- 边界 B：20（恰好等于核数）/ 21 / 26 / 1000；
- KV 长度混合：短 KV（`segCount = 1`，无 split）与长 KV（多 split）落在同一 batch 数组里；
- `fdMode` on / off 两条收尾路径各验一次；
- 布局：BSND 与 TND（TND 的 per-batch kvLen 走 prefix delta / `seqUsedKv`）；
- 口径：与通用 FAInfer 路径同 shape 对拍。

### 9.9 落地顺序

1. host 去掉截断并改为下沉标量（`fa_split.h` + `tilingdata.h`）——此步单独提交会让 kernel 读到空 `coreInfo`，**必须与第 2 步同批提交**；
2. kernel 加重算 + task 循环 + L840 归位 + CombineScale 输入派生；
3. FA4（`flash_attn_npu_4/`）同步；
4. 按 9.8 验证。
