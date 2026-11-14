# MLA decode GEMV kernel：L1 流水 + 大 B 调度 合一改造方案

- 日期：2026-11-11
- 参考：[MLA 场景 L1 流水设计说明.md](../MLA%20%E5%9C%BA%E6%99%AF%20L1%20%E6%B5%81%E6%B0%B4%E8%AE%BE%E8%AE%A1%E8%AF%B4%E6%98%8E.md)
- 对象：[mla_decode_gemv.hpp](../../csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp)、[qk_matmul.hpp](../../csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp)、[fa_split.h](../../csrc/ascend910/flash_attn_npu_3/fa_split.h)
- 配套：[mla_decode_gemv_sync_and_guard.md](mla_decode_gemv_sync_and_guard.md)（§4 L840 记账、§5 容量、§9 kernel 重算）
- 状态：**纯设计，未实现**

---

## 1. 目标

| # | 目标 | 现状 | 本方案 |
| --- | --- | --- | --- |
| G1 | **head（M = qHeads）不受 L1 限制** | Q 整块常驻 `576H`，H > 227 时越界 | Q 按 m 块 + 288 维段滚动进 4 槽，L1 占用**恒定 504 KB，与 H 无关** |
| G2 | **batch 不受核数限制** | 一核一 task，task 数被 `MAX_NODE_NUM = 25` 截断，超出 batch 静默不计算 | 一核多 task + **kernel 就地重算 task 描述**（不落表） |
| G3 | 保持 mm1/mm2 段间软流水与既有数值口径 | — | 事件账本按新槽位重新配平 |

G1 来自参考文档的布局（m 分块 + Q 288 分段 + QP 4 槽分时复用 + KV 3 槽环形），G2 来自 [mla_decode_gemv_sync_and_guard.md §9](mla_decode_gemv_sync_and_guard.md)。**两者正交**：G2 是功能错误修复，G1 是容量上限解除。

---

## 2. 计算与 tile

沿用参考文档的几何（与当前 kernel 的 `MLA_DECODE_*` 常量一致）：

```
mm1: P = Q_nope·K_nopeᵀ + Q_rope·K_ropeᵀ      Q(M,576) × K(128,576)ᵀ → P(M,128)
mm2: O = P·V                                  P(M,512) × V(512,128)  → O(M,128)

M = qHeads（运行时）     576 = 512 NoPE + 64 RoPE
KV tile = 128            （MLA_DECODE_KV_TILE）
mL1Loops = ceil(M / 128)  ← M = 256 时 = 2，与参考文档示例一致
```

mm1 的 k 方向 576 分 2 段（每段 288），mm2 的 k 方向 512 分 4 块（每块 128）。

---

## 3. L1 / L0 空间划分

```
L1 总占用 = 288 K(QP) + 216 K(KV) = 504 K  ≤  L1_MAX_SIZE(512 K)
```

| 区域 | 大小 | 槽数 | 槽大小 | 用途 |
| --- | --- | --- | --- | --- |
| QP | 288 K | 4 | 72 K | mm1 装 Q 段；mm2 装 P 段（**分时复用**） |
| KV | 216 K | 3 | 72 K | mm1 装 K 块；mm2 装 V 块（**交替混装**） |
| L0C | 128 K | 2 | 64 K | mmad 输出双缓冲 |

槽容量校验：`128 × 288 × 2 B = 72 K`（Q 段 / K 块）；`128 × 128 × 2 B = 32 K ≤ 72 K`（P 块 / V 块，有余量）。

### 3.1 QP 4 槽排布

Q 的 576 维按 288 切两段，`kL1 = 0/1`，槽内保持 NoPE/RoPE 连续：

```
kL1=0 槽: [NoPE 256][RoPE 32]
kL1=1 槽: [RoPE 32][NoPE 256]
```

- 同一 m 块的两个 k 段落在**相邻**槽（0/1 或 2/3），拼成完整 576 维；
- 槽映射 `ka = idxMap[mIdx % 2] + kL1`，`idxMap = {0, 2}`：偶 m 块 → 槽 0/1，奇 m 块 → 槽 2/3；
- 两组槽构成乒乓：槽 0/1 算 m0 时，槽 2/3 搬 m1（参考文档 §4.3）。

### 3.2 KV 3 槽环形

K 与 V 共用 `L1KV[3]` 与同一个 `iKV` 计数器，**跨 mm1/mm2 连续递增**，落槽 `iKV % 3`：

```
iKV:  0    1    2    3    4    5    6
L1KV[0] ← K0 ────────→ V1 ────────→ K4 ...
L1KV[1] ←──── K1 ────────→ V2 ────────→ ...
L1KV[2] ←──────── K2 ────────→ V3 ────────→ ...
```

1 个在算 + 1 个在搬 + 1 个缓冲，搬运抖动不影响计算。

### 3.3 与当前布局的差异

| | 当前 gemv | 本方案 |
| --- | --- | --- |
| Q | 整块常驻 `[0, 576H)`，装一次全段复用 | 按 (m 块, kL1 段) 滚动，**每个 KV 段重装** |
| P | GM → L1 staging，与 K 共用 2 槽 | L0C → fixp → **复用 QP 4 槽** |
| K/V | K/P 共 2 槽 ping-pong、V 独立 2 槽 | K/V 共 3 槽环形 |
| L1 占用 | `576H + 256 K`（随 H 线性膨胀） | 恒定 504 K |

---

## 4. 同步事件分配

参考文档要求"每个槽配一对独立事件"，但必须落到硬件约束上：**同一 HardEvent 类型下 EVENT_ID 只有 0..7 共 8 个**。12 对槽事件无法放在同一类型里，需按类型分散：

| 用途 | 事件类型 | ID | 个数 |
| --- | --- | --- | --- |
| QP 槽（4） | `MTE1_MTE2` / `MTE2_MTE1` | 0..3 | 4 + 4 |
| KV 槽（3） | `MTE1_MTE2` / `MTE2_MTE1` | 4..6 | 3 + 3 |
| L0C 槽（2） | `FIX_M` / `M_FIX` | 0..1 | 2 + 2 |

每类 ID 用量 ≤ 7 < 8，合法。方向语义与既有约定一致（`X_Y` = 「X 干完 → Y 才能干」）：

- `MTE1_MTE2`：槽被计算读空 → 允许 MTE2 覆写（WAR）；
- `MTE2_MTE1`：槽搬完 → 允许 MTE1 读取（RAW）。

**与当前 kernel 的两处关键差别**：

1. 当前 A 通道（Q）**不单独等 `MTE2_MTE1`**，靠 [qk_matmul.hpp L187-188](file:///home/w00811046/code/fan/1111mla/csrc/ascend910/flash_attn_npu_3/qk_matmul.hpp#L187-L188) 的跨队列刷干屏障兜底。新方案 Q 分块滚动后每块 Q 都要被 mmad 立刻消费，**必须真正使用每槽独立的 `MTE2_MTE1` 等待**，屏障技巧不再够用。
2. prelaunch / 尾部 drain 的集合必须**按新分配重算并重新配平**：变成 `MTE1_MTE2{0..6}`、`MTE2_MTE1{0..6}`、`FIX_M{0,1}`、`M_FIX{0,1}`（外加 G2 引入的额外号）。规则不变：body 里"有 Wait 无 Set"的号必须铺，无人消费的首尾成对删除，每 `(类型, ID)` 只铺一次。

---

## 5. mm1 流水

### 5.1 循环结构

```
for nL1 in 0..3:                 # n 方向 512/128 = 4 块
  for kL1 in 0..1:               # k 方向 576/288 = 2 段
    Get KV 槽, 搬 K 块
    for mL1 in 0..mL1Loops:
      Get QP 槽 ka = GetQPL1RealIdx(mIdx, kL1)
      if nL1 == 0: 搬 Q（仅首次）
      mmad(Q[ka], K[kb]) → L0C[kc]
```

### 5.2 时序

```
时间 →        t0    t1      t2        t3        t4        t5
──────────────────────────────────────────────────────────────
QP槽0/1   ▶Q(m0)▶ ║ █mmad n0█ █mmad n1█ █mmad n2█ █mmad n3█ →释放
QP槽2/3          ▶Q(m1)▶（与 m0 的 n0~n3 重叠）
──────────────────────────────────────────────────────────────
KV槽 S0   ▶K0▶   ║ █mmad(K0)█  ·   ·   ·   ▶K3▶
KV槽 S1       ·  ▶K1▶ ║ █mmad(K1)█  ·   ·   ▶K4▶
KV槽 S2       ·   ·  ▶K2▶ ║ █mmad(K2)█
──────────────────────────────────────────────────────────────
L0C       █P(m0,n0)█ █P(m0,n1)█ █P(m0,n2)█ █P(m0,n3)█
```

### 5.3 要点

- **Q 仅 `nL1 == 0` 搬一次**，4 个 n 块复用；`nL1 == last` 才 `Set(MTE1_MTE2)` 释放；
- **Q 两组乒乓**：搬运被 4 次 mmad 掩盖；
- **K 三槽滚动**；
- **P 累加**：`kL1 = 0/1` 两段在 L0C 累加得到完整 P 的一个 n 块。

---

## 6. mm2 流水

```
for mL1:
  for kL1 in 0..3:               # k 方向 512/128 = 4 块
    Get KV 槽, 搬 V 块
    for nL1:
      Get QP 槽 kb（P 复用 Q 已释放的 4 槽）
      mmad(V[ka], P[kb]) → L0C[kc]
```

要点：

- **P 复用 QP 4 槽**，按 `iQP % 4` 落槽；
- **V 进 KV 3 槽**，与 K 共用 `iKV`，跨 mm1/mm2 连续递增；
- **O 累加**：`O = Σ P_i·V_i`（i = 0..3），每块累加进 L0C。

### 6.1 mm1 → mm2 衔接

```
mm1: Q(4槽) × K(3槽) → P 累加在 L0C
  ↓ fixp 搬出
P 写 L1QP[4]（Q 已释放，P 复用 4 槽）
  ↓
mm2: P(4槽) × V(3槽) → O 累加在 L0C
```

三条硬约束（继承参考文档 §6）：

1. **QP 4 槽分时复用**：Q 与 P 不同时活跃，靠 `nL1 == last` 的 `Set(MTE1_MTE2)` 释放；
2. **KV 3 槽交替混装**：K/V 时间交错，靠 `iKV % 3` 落槽；
3. **L0C 双缓冲**：fixp 搬出与下一次 mmad 重叠。

---

## 7. 叠加"一核多 task"（G2）

L1 流水改造与调度改造正交，调度侧完全按 [mla_decode_gemv_sync_and_guard.md §9](mla_decode_gemv_sync_and_guard.md) 执行：

1. **host 只下沉标量**：`totalTaskNum` / `nsplitPerBatch` / `batchWithWork`，`needCoreNum = min(totalTaskNum, blockDim)`，删除 `MAX_NODE_NUM = 25` 截断；
2. **kernel 就地重算 task 描述**：连续块分配 + 每核一次 `O(batchSize)` 扫描定位起点 + `lseBase/oBase` running 累加 + 核内顺序推进；
3. **CombineScale 输入同样重算**（`splitInfo[]` 装不下几百个 batch）；
4. **prelaunch / 尾部 drain / `SyncAll` 保持"每核一次"**，留在 task 循环之外。

与 L1 改造叠加后的两条额外要求：

- **每个 KV 段的 Q 重装发生在 task 内**，令牌账本要按"每 task 每段借还"重新配平（不能沿用 `§4` 里那种"每核铺一次"的简化陈述）；
- **L840 那笔归还必须归位**：从 `loadQGM` 之后移到该 task 的 Q 全部消费完之后（多 task 后它从记账变成真实 WAR 屏障），事件类型/ID 按 §4 的新分配选取。

---

## 8. 代价与取舍

### 8.1 Q 重装是硬代价

QP 槽被 P 复用 ⇒ **Q 无法跨 KV 段常驻**，每个 KV 段都要重装 `mL1Loops × 2` 个 72 KB 段：

| | 每 KV 段的 MTE2 流量（H = 256） |
| --- | --- |
| 当前布局（Q 常驻） | Q 一次性 288 KB + 每段 K/V ≈ 272 KB |
| 本方案 | **每段 Q ≈ 288 KB（随 H/128 线性增长）** + K/V ≈ 272 KB |

段数越多，本方案的 MTE2 压力越大。

### 8.2 mm1 / mm2 的 L1 级重叠消失

Q 必须全部释放（`nL1 == last`）后 P 才能落槽，因此 mm1 与 mm2 在 QP 区上**硬串行**；当前布局靠 K/P 共享 2 槽滚动实现的 mm1/mm2 交叠在 L1 层面不再成立（只剩段间软流水）。

### 8.3 建议：双路径按 shape 选择（推荐）

两个目标的最优解并不重合，建议**按 H 选布局**而不是一刀切：

| 条件 | 布局 | 理由 |
| --- | --- | --- |
| `H ≤ 227`（含标准 128） | 保留当前"Q 常驻"布局 | 无 Q 重装、mm1/mm2 可重叠，性能更优 |
| `H > 227` | 本方案"Q 分块 + QP 复用 + KV 3 槽" | 唯一能在 512 KB 内支持大 H 的布局 |

落地方式：同一 kernel 模板按 tilingKey / 模板参数实例化两套 L1 布局；闸门不需要新增上界（G1 的目标正是"不再需要上界"）。

---

## 9. 关键风险

| # | 风险 | 说明 | 缓解 |
| --- | --- | --- | --- |
| R1 | **P 的双消费路径** | mm2 需要 P 在 L1；online-softmax 需要 P 在 UB，而 Vector 读不到 L1（只有 L1→L0 通路） | 保留 GM staging 作为 Vector 输入，同时把 P 写回 QP 槽；明确两条路径的顺序与地址，作为第一阶段之外的独立步骤 |
| R2 | **事件账本全面重算** | prelaunch 集合、每槽借还次数、task 循环后的配平全部变化 | 先列出"每个 (类型, ID) 的 Set/Wait 计数表"再动代码；多铺/少铺都会死等 |
| R3 | **Q 的 `MTE2_MTE1` 必须真正生效** | 不能再用刷干屏障替代 | 每块 Q 独立等 `MTE2_MTE1[ka]` |
| R4 | **host/kernel 规则实现两遍的漂移** | 见 §9.7 (guard 文档) | 抽共享纯函数头文件，或在 kernel 校验 task 总数 |
| R5 | 槽内 `(256+32)` / `(32+256)` 排布与 `loadA` 分段对齐 | 参考文档 §7.1 列为"必须做对" | 单测覆盖 kL1=0/1 的 96 维分段不跨界 |
| R6 | 参考文档 §7.2 已列出的历史问题 | `Set(MTE2_MTE1)` 后立刻 Wait 导致串行、`qpL1BufIter` 未递增等 | 实现时逐条对照，勿照抄伪代码 |

---

## 10. 分阶段落地

**顺序建议：G2 先、G1 后**。G2 是功能错误（静默丢 batch，输出错），G1 是容量上限（超界可回落通用路径），且两者正交，G2 改动面更小、收益确定。

| 阶段 | 内容 | 交付 |
| --- | --- | --- |
| P0 | G2：host 下沉标量 + kernel 重算 + task 循环 + L840 归位 + CombineScale 派生（guard 文档 §9） | 大 B 功能正确 |
| P1 | G1-a：QP 4 槽 + KV 3 槽布局，Q 分块/分段滚动，事件表重算（mm1 部分） | H > 227 可跑 |
| P2 | G1-b：mm2 侧 P 复用 QP 槽 + `iKV` 跨 mm1/mm2 连续递增 | 完整流水 |
| P3 | R1：P 的双消费路径收敛（可选，性能向） | 减少 GM 往返 |
| P4 | FA4（`flash_attn_npu_4/`）同步 | — |

P0 与 P1 无依赖，可并行开发；P1 建议先只改 mm1 以缩小调试面。

---

## 11. 验证要点

- **G2 边界**：B = 20 / 21 / 26 / 1000；短 KV（无 split）与长 KV（多 split）混在同一 batch 数组；fdMode on/off 两条收尾；BSND 与 TND。
- **G1 边界**：H = 128（走旧布局）/ 228 / 256（走新布局），与通用 FAInfer 同 shape 对拍。
- **流水正确性**：每个槽的 `MTE1_MTE2` / `MTE2_MTE1` 计数配平（可先加断言）；`iKV` 跨 mm1/mm2 单调递增。
- **一致性**：host 与 kernel 重算出的 task 总数一致。

> 以上均为设计，未改动任何源码。
