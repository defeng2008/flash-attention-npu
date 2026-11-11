# MLA 计算流程与 L1 内存布局

> 设备：Ascend 910（Atlas A2）。代码路径：`csrc/ascend910/flash_attn_npu_3/`（v4 对称）。
> 相关文档：[mla_design.md](./mla_design.md)（MLA 主体设计）、[mla_bnsd_design.md](./mla_bnsd_design.md)（BNSD 扩展）。

本文说明 MLA 两种形态（absorbed / prefill）在 kernel 内的计算流程、L1 缓存布局，以及与普通（非 MLA）FA 的差异。

## 1. 高层数据流（两形态通用）

```
                    ┌─────────────┐
 q_rope (64) ───────┤             │
 qv/latent (512) ──▶│  QK Stage   │  scores (B,H,Sq,Sk)
                    │  q·kᵀ×scale │──────▶ softmax ──▶ probs
                    └─────────────┘                         │
                    ┌─────────────┘                         ▼
 k_rope (64) ───────┘            ┌─────────────┐   ┌─────────────┐
 v_latent (512) ────────────────▶│  PV Stage   │◀──│   (probs)   │
                                 │  p·v        │──────▶  O (B,S,H,512)
                                 └─────────────┘
```

- **absorbed / decode**：`q` = rope(64)，`qv` = latent(512)，`k` = k_rope(64)，`v` = v_cache(512)，KV 头恒为 1，输出 `dv = 512`。
- **prefill / unabsorbed**：`q`/`k` = nope|rope 已拼好(192)，`v` = 128，KV 头可 >1（GQA），输出 `dv = 128`。

## 2. Kernel 内部流程（absorbed，QK 段拆两源）

对应 `qk_matmul.hpp` 的 `loadQSegmentToL1` / `loadQGM` 与 `headDimSplitQK`：

```
  Q 侧（GM → L1A）               K 侧（GM → L1B）
┌─────────────────────┐        ┌──────────────────────┐
│ qv(latent,512)      │  [0,512)   v_cache(latent,512)│
│ q_rope(64)          │  [512,576)  k_rope(64)        │
│   拼成 576 宽 zN 块  │        │   拼成 576 宽 zN 块    │
└─────────────────────┘        └──────────────────────┘
         │                              │
         ▼                              ▼
   headDimSplitQK（576>256 → 按列切片，每片 ≤256 → L0A/L0B）
         │
         ▼
   矩阵乘 Σ d片 (Q片 · K片ᵀ) × scale
         │
         ▼
      scores → softmax（causal mask / FD 归约）
         │
         ▼
   PV: probs × v_cache(latent) ──▶ 累加 O（embedV=512）
         │
         ▼
   O 写出，layout 强 BSND → 连续 (B,S,H,512)
```

要点：

1. **Q/K 双源拼装**：latent 走主操作数（`gQv` / v_cache），rope 走副源（`mlaRopeQ` / `mlaRopeK`）；两段各用自己的 stride 搬运，在 L1 里按列首尾相接成一块 576。两段宽度（512/64）均为 C0（16 元素）整数倍，分段 ND2NZ 与整块搬运结果逐位相同。
2. **576 宽必须走 `headDimSplitQK`**：CUBE 单次 mmad K 上限 256，按列切片（每片 16 对齐），分片结果累加回 L0C，对上层透明。
3. **数学等价**：`score = scale·(qv·vᵀ + q_rope·k_ropeᵀ)`，即 `[qv | q_rope]·[v | k_rope]ᵀ` 展开。
4. **PV 只乘 latent**：`O = softmax(scores)·v_latent`，rope 的 k 只进 QK 不进 PV。

## 3. L1 内存布局

**QK 与 PV 同时共存、按地址前后分区**（不是顺序复用）：`blockMmadQK.init(..., 0)`，`blockMmadPV.init(..., L1_QK_SIZE)`。预算计算见 `mha_fwd_kvcache.cpp`：

```
uint32_t L1_QK_SIZE  = qL1Rows * qL1DynNum * sizeof(ElementQ);                        // Q 区
uint32_t qkKPoolSize = nDynNum * kDynNum * sizeof(ElementK) * DOUBLE_BUFFER;          // K 双缓冲
uint32_t kPVPoolSize = pL1Rows * kPVDynNum * sizeof(ElementP) * pL1Stages;            // P 区
uint32_t vPoolSize   = vL1Stages * vSegLen * vSliceWidth * sizeof(ElementV);          // V 区
```

### 3.1 MLA absorbed（embed=576 > 256 → split 分支）

```
L1 (L1_MAX_SIZE=512KB)，地址从低到高：
┌────────────────┬────────────┬────────────┬───────────┬──────────────┬─────────┐
│ Q (128×576)    │ K0(128×256)│ K1(128×256)│ P 区 ×2   │ V 区 ×2      │ ndCopy  │
│ A 侧·全宽常驻   │ B 侧·K 按列│ B 侧 ping- │ (pL1Rows× │ (256×128)    │ 预留    │
│ 576 = 512+64   │ 片 0       │ 片 1 pong  │ 256)×2   │ ×2          │         │
└────────────────┴────────────┴────────────┴───────────┴──────────────┴─────────┘
 L1_QK_SIZE=147456   qkKPool=131072    kPVPool   vPool=131072
```

split 分支参数（`headDimSplit = embedQK > 256`）：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `qL1Rows` | `Q_TILE_CEIL = 128` | Q 行数 |
| `qL1DynNum` | `RoundUp(576, 16) = 576` | Q 全宽常驻 L1 |
| `nDynNum` | `L1_D512_KV_TILE = 128` | KV tile 长度 |
| `kDynNum` | `NUM_256 = 256`（固定） | K 按列切 256 宽，576 = 2×256+64 三片 |
| `kPVDynNum` | `NUM_256` | P 区宽度 |
| `vL1Stages` / `vSegLen` / `vSliceWidth` | `2 / 256 / 128` | V = 512 再切 4 片（128 宽）进 L1B |
| `pL1Stages` | `DOUBLE_BUFFER = 2` | P 区双缓冲 |

- **A 侧 Q 全宽（576）常驻** L1，QK 循环每次取一列片（256/256/64）经 `CopyL1ToL0A` 进 L0A；
- **B 侧 K 不存全宽**：`(nDyn=128)×256` 双缓冲轮换，对应 3 个列片；
- **PV 的 V**：`256×128` 双缓冲，embedV=512 切 4 片；P 区放 softmax 结果，256 宽 ×2。

### 3.2 普通 FA（embed ≤ 256，非 split 分支）

预算走 `maxQKPL1Size` 反推路径：

| 参数 | 值 |
| --- | --- |
| `kDynNum` | `RoundUp(embed, 128)`，钳位 ≥256（全宽单块） |
| `nDynNum` | 由 `(maxQKPL1Size - maxQL1Size) / kDynNum / 2` 反推，`≤ L1_MAX_N_NUM=128`，32 对齐 |
| `qL1DynNum` | `= kDynNum` |
| `kPVDynNum` | `nDynNum * kDynNum / pL1Rows` |
| `vL1Stages` / `vSegLen` / `vSliceWidth` | `1 / kvStackLen / embedV`（V 全宽单块） |

### 3.3 compact decode 特例（非 MLA）

`embed == 512 && embedV == 512 && !FLASH_DECODE_FLAG && kvNewSeqlen == 0 && maxQSeqlen*groupSize ≤ 64` 时：

- `qL1Rows = pL1Rows = 64`，`pL1Stages = 1`，`kPVDynNum = kvStackLen`；
- **K 与 P 共享同一 L1 区**（`pBufOffset = L1_QK_SIZE + qkKPoolSize`），QK/PV 两阶段复用，不计二次。

MLA absorbed 因 `embed=576` 不命中该分支。

## 4. 与普通 FA 的对比

**分区结构完全一致**（`| Q | K×2 | P×2 | V×2 |`），差异只在尺寸：

| 项 | 普通 FA（embed ≤ 256） | MLA absorbed（embed=576） |
| --- | --- | --- |
| `kDynNum` | `RoundUp(embed,128)`，全宽单块 | **固定 256**，K 按列切片 ×3（split 分支） |
| A 侧 Q | `128×embed` | `128×576`（两源拼接后全宽） |
| B 侧 K | `nDyn×embed` | `nDyn×256` ×2（片轮换，非全宽） |
| B 侧 V | `embedV` 宽单块 | `128` 宽 ×4 片 |
| Q 源 | 单张量 | qv(512) + q_rope(64) 拼装 |
| K 源 | 单张量 | v_cache(512) + k_rope(64) 拼装 |
| 预算路径 | `maxQKPL1Size` 反推 nDyn | 固定 256 片 |

一句话：**MLA 没有引入新的 L1 结构，只是激活了「K/V 按列切片的 split 路径」**（embed=576 才触发）；Q 全宽常驻、K 切 256 片轮换、V 切 128 片轮换；非 MLA 的 ≤256 路径仍走全宽单块。

## 5. 关键代码位置

| 内容 | v3 位置 |
| --- | --- |
| L1 预算 / QK/PV init | `mha_fwd_kvcache.cpp` `SetBufferRange` 附近（`L1_QK_SIZE` 起） |
| Q 双源搬运 | `qk_matmul.hpp` `loadQSegmentToL1`（L145-162）、`loadQGM`（L173-186） |
| K 双源 / page stride | `mha_fwd_kvcache.cpp` `SetMlaRopeK` + QK 加载处 |
| headDimSplitQK 列切片 | `qk_matmul.hpp` 对应 d-slice 循环 |
| QK L1 分配 | `qk_matmul.hpp` `init`（A 前 B 后，B `STAGES` 深） |
| PV L1 分配 | `pv_matmul.hpp` `init`（A=P、B=V，B `vStages` 深） |
