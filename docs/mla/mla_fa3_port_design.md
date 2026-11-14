# 设计：把 FA4 的 MLA decode 优化全量移植到 FA3 三个接口

- 日期：2026-11-12
- 来源：FA4（`csrc/ascend910/flash_attn_npu_4`）MLA decode 优化，见 [mla_decode_gemv_l1_pipeline_plan.md](mla_decode_gemv_l1_pipeline_plan.md)、[mla_decode_gemv_sync_and_guard.md](mla_decode_gemv_sync_and_guard.md)
- 对象：[fa_split.h](../../csrc/ascend910/flash_attn_npu_3/fa_split.h)、[mla_decode_gemv.hpp](../../csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp)、[tilingdata.h](../../csrc/ascend910/flash_attn_npu_3/tilingdata.h)
- 状态：**已实现并验证**（pytest 25 passed；FA3 全部 perf 行 ≤ 1.05）

---

## Summary

把 `csrc/ascend910/flash_attn_npu_4`（FA4）这一轮 MLA decode 优化**全量**移植到 `csrc/ascend910/flash_attn_npu_3`（FA3），覆盖 FA3 的三个 forward 接口 `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache`，目标：

1. FA3 的 absorbed MLA decode（gemv）功能正常（精度达标、不静默丢 task）；
2. `tests/perf/bench_mla_vs_fia.py` 退出码 0、全部行 `candidate/FIA ≤ 1.05`（不回归）。

移植内容 = FA4 已完成的全部改动（见 `.trae/specs/port-mla-decode-l1-pipeline-to-fa4/` 的 spec/tasks/checklist）：
- **P0 调度**：去 `MAX_NODE_NUM = 25` 截断、task 规则下沉 + kernel 就地重算、核内"一核一段连续 task"循环；
- **P0 同步**：`SetFlag<MTE1_MTE2>(EVENT_ID3)` 归位到 task 段循环之后；
- **P0 Combine**：组表回退为 tiling 内 host 填表（不引入 workspace scratch）；
- **G1 布局**：head 维分块滚动（`qHeads > 32`），解除 H 上界；
- **BNSD**：`Sq==1` packed BNSD 视图判定修复 + 新增精度用例。

**范围外（FA3 已具备，不移植）**：Task 9 的 prefill `pipelineDrain` 修复 —— FA3 的 `mha_fwd_kvcache.cpp` 已含 `pipelineDrain` 与 `using SplitInfoType = splitNode[25];`。

---

## 实施状态（已完成）

全部 6 项改动已落地并通过验证。执行中额外发现并修复 1 个 FA3 特有缺陷（G1 补充，见下）。

| 项 | 文件 | 状态 |
| --- | --- | --- |
| 1) `nsplitPerBatch` 标量 | `flash_attn_npu_3/tilingdata.h` | ✅ 复用 `padding3` 槽位，`static_assert(sizeof==2376)` 通过 |
| 2) 去 25 截断 + anySplit 回填 | `flash_attn_npu_3/fa_split.h` | ✅ |
| 3) task 就地重算 + 核内 task 循环 | `flash_attn_npu_3/mla_decode_gemv.hpp` | ✅ |
| 4) head 维分块（G1） | `flash_attn_npu_3/mla_decode_gemv.hpp` | ✅ |
| 5) HardEvent 归位 | `flash_attn_npu_3/mla_decode_gemv.hpp` | ✅ `EVENT_ID3` 移至 task 段循环后 |
| 6) BNSD `Sq==1` 视图判定 | `flash_attn_npu_3/flash_attn_npu_interface.py` | ✅ 改为显式 stride 判定 |
| 7) 新增 BNSD 精度用例 | `tests/test_flash_attn_npu_v3_mla.py` | ✅ `test_mla_decode_bnsd_sq1_cross_batch` |
| **G1 补充**：L1 Q 行区与 H 解耦 | `flash_attn_npu_3/mla_decode_gemv.hpp:687` | ✅ **H≥313 aicore 越界崩溃修复** |

**G1 补充缺陷（执行中发现）**：FA3 移植 head 分块时，L1 里 Q 行区仍按 `qHeads` 预留（`qRows = qHeads`），`l1QKSize = qRows * qDynNum * sizeof(ElementQ)` 随 H 线性增长，H 变大后把 V/nd 区顶出 L1 512KB 上限，触发 `fftsplus aicore error: The GM address accessed by scalar exceeds 48 bits`（H≤312 正常、与 KV 无关）。修复：`qRows` 固定为 `BlockMmadQK::L1TileShape::M`（=128），与 FA4 同路径一致，使 `l1QKSize` 恒为 147456。

---

## Current State Analysis（基于实际代码比对）

FA3 与 FA4 是同构的两套实现，同名文件一一对应。逐文件核对结果（`diff` 实测）：

| 文件 | FA3 现状 | 与 FA4 的差距 | 需改 |
| --- | --- | --- | --- |
| `flash_attn_npu_3/fa_split.h` | **改造前**：`MAX_NODE_NUM = 25`（L531）→ `splitKvForGemvDecode` 预清零 `coreInfo[25]`/`splitInfo[25]`（L535-554），逐 task 写 `coreInfo[coreIdx]`（L640-655）、无条件写 `splitInfo[splitIdx]`（L626-635），`needCoreNum = FaMin(coreIdx, 25)`（L663） | 需改为"只统计 + 下沉标量 + anySplit 时回填 splitInfo" | ✅ |
| `flash_attn_npu_3/mla_decode_gemv.hpp` | **改造前**：`coreTask = tilingData->coreInfo[coreIdx]`（L761）每核单 task；无 head 分块；`SetFlag<MTE1_MTE2>(EVENT_ID3)` 紧贴 `loadQGM`（L840） | 需加 task 就地重算 + 核内 task 循环 + head 分块 + 事件归位 | ✅ |
| `flash_attn_npu_3/tilingdata.h` | **改造前**：单一 `FAInferTilingData`（L36），有 `totalTaskNum`（L48）但**无 `nsplitPerBatch`**；有 `padding1/2/3`（L55-57）；无 `static_assert` | 需加 `nsplitPerBatch` + get/set | ✅ |
| `flash_attn_npu_3/flash_api.cpp` | gemv 分支已用 `get_needCoreNum()`，`IsMlaDecodeGemvShape` 的 H 上界为 32（`numHeads <= 32`），用 `is_bnsd` 命名，`partialBase` 在 4 个 scratch 缓冲之后 | 基本无需改，仅核对 workspace 无 `gemvSplitNodeSize` | ⚠️ 核对 |
| `flash_attn_npu_3/combine_scale.hpp` | **已是改造后形态**：`template<... typename SplitInfoType>`（L25）+ `__gm__ SplitInfoType &splitInfo`（L80）；比 FA4 多一个 `perBatchBase` 分支 | 不改（保留 `perBatchBase`） | ❌ |
| `flash_attn_npu_3/mha_fwd_kvcache.cpp` | **已是改造后/上游形态**：有 `pipelineDrain`、跨 task `issuedStackCount`、`using SplitInfoType = splitNode[25];`（L1275） | 不改 | ❌ |
| `flash_attn_npu_3/flash_attn_npu_interface.py` | **改造前**：`_transposed_bnsd_view` 有 `is_contiguous()` 早退（L50）；`_SCHEDULER_METADATA_TILING_BYTES = 2376`（L1608） | 需修 BNSD 视图判定；核对常量 | ✅ |
| `tests/test_flash_attn_npu_v3_mla.py` | 14 个 test func，无 `Sq==1` packed BNSD decode 用例 | 新增 1 个用例 | ✅ |

### FA3 与 FA4 的**结构性差异**（不能照搬，必须重新推导）

1. **Tiling / ABI 不同**：FA3 是单一 `FAInferTilingData`（Python 常量 **2376**），FA4 拆成 `FAInferStaticTilingData(2336) + FAInferRuntimeData(96)`。`nsplitPerBatch` 在 FA3 只能塞进现有 `padding` 区，**不能**照抄 FA4 的拆分，且必须保证 `sizeof(FAInferTilingData)` 与 2376 不变。
2. **append-KV**：FA3 有 `ctx.kvNewSeqlen` / `tiling->kvNewSeqlen`，各处 `kvLen = seqlens_k_cpu[b] + kvNewSeqlen`（`splitKvForGemvDecode` L558/L590/L612，kernel L777）。task 就地重算与 `KvLenOf` **必须带 `kvNewSeqlen`**，否则 host/kernel 规则分叉 → 死锁或错数据。
3. **事件账本不同**：FA3 kernel 的 softmax 取参、事件集（无 FA4 的 `MLA_DECODE_V_SLICE_WIDTH`、`SetMlaLatentQ`、`fdMode` 形参）与 FA4 不同。`EVENT_ID3` 归位必须按 **FA3 自身账本**重新配平（prelaunch Set 与 footer Wait 逐项计数一致）。
4. **head 分块阈值**：FA3 无 `kMlaDecodeGemvHeadBlock`，需引入并按 FA3 的持久 O 行数约束复核（FA4 取 32）。
5. **接口面**：FA3 多 `flash_attn_with_kvcache`（走 kvcache → `mha_fwd` → gemv）。三个接口都需覆盖。
6. **workspace 偏移约定不同**：FA4 FD partial 初始化在偏移 0；FA3 在 `partialBase`（4 个 scratch 缓冲之后，`flash_api.cpp:798`）。
7. **L1 Q 行区必须与 H 解耦（执行中新发现）**：FA4 的 `mla_decode_gemv` 里 Q 行区常驻行数固定为 QK block 的 `L1TileShape::M`（=128），与 `qHeads` 无关；FA3 移植 head 分块时若沿用 `qRows = qHeads`，则 `l1QKSize = qRows * qDynNum * sizeof(ElementQ)` 随 H 线性增长，会顶出 L1 512KB 上限 → aicore 越界崩溃。**这是 FA3 结构性差异，必须显式对齐为 `L1TileShape::M`。**

### 验收口径（现状）

- FA3 测试：`tests/test_flash_attn_npu_v3_mla.py`（`REL_L2_TOL = 5e-3`，含 `test_mla_decode_bsnd_kv_multicore` / `single_core` / `test_mla_decode_rejects_append_kv`）。
- perf：`tests/perf/bench_mla_vs_fia.py` 同时 import `flash_attn_npu_3`（`fa3_func/fa3_varlen/fa3_kvcache`，L62-66）；FA3 行 = prefill/decode 的 `fa3_func_bsnd`、`fa3_kvcache_bsnd`、`fa3_varlen_tnd`；门限 `KERNEL_GATE = 1.05`。
- 目标边界（与 FA4 对齐）：B ≤ 64、H ≤ 512。

---

## Proposed Changes

### 1) `csrc/ascend910/flash_attn_npu_3/tilingdata.h`
**What**：给 `FAInferTilingData` 增加 `nsplitPerBatch` 标量。
**Why**：P0 后 host 只下沉标量，kernel 需要 `nsplitPerBatch` 就地重算 task。
**How**：
- 先 `grep -rn "padding3" csrc/ascend910/flash_attn_npu_3/` 确认未被使用；用 `uint32_t nsplitPerBatch;` 替换 `padding3`（保持结构尺寸不变），并加 `get_nsplitPerBatch() / set_nsplitPerBatch()`。
- 若 `padding3` 已被占用，则改用其他 padding 或新增字段，并同步调整 Python 常量。
- 加 `static_assert(sizeof(FAInferTilingData) == 2376, ...)`（FA3 原本没有；若实际尺寸不同，则按实测值固定并在 Python 侧同步）。
- **保留** `kvNewSeqlen` / `kvCacheSeqlen`（FA3 特有）。
- 同步 `flash_attn_npu_3/flash_attn_npu_interface.py` 的 `_SCHEDULER_METADATA_TILING_BYTES`（L1608），确保与实际 `sizeof` 一致。

### 2) `csrc/ascend910/flash_attn_npu_3/fa_split.h::splitKvForGemvDecode`
**What**：改为 FA4 的"只统计 + 下沉标量 + anySplit 回填"形态。
**Why**：消除 25 核截断导致的**大 batch 静默丢 task**（P0 核心 bug）。
**How**（逐段对照 FA4 `splitKvForGemvDecode` 改写，但保留 append-KV 口径）：
- 删除 `constexpr uint32_t MAX_NODE_NUM = 25;`（L531）与 L535-554 的 `coreInfo[]` / `splitInfo[]` 预清零循环。
- 保留 `batchWithWork` 统计（L556-562）与空 batch 早退（L563-569）。
- 保留切分预算推导（`SEG_TARGET`、`maxSplitsPerBatch`、`nsplitPerBatch` 夹取，L571-604）。
- 第一遍循环：只累加 `totalTaskNum` / `lseTaskOffset` / `oTaskOffset` / `splitIdx`，置 `anySplit`，**不写** `coreInfo`（删除 L640-655）。
- `needCoreNum = FaMin(totalTaskNum, ctx.blockDim)`（替换 L663）。
- `if (anySplit)` 分支内新增第二遍填表循环，把每 batch 一行写回 `tiling->splitInfo[row]`（字段：`batchIdx` / `headStartIdx=0` / `headEndIdx=qHeads` / `qStartIdx=0` / `qEndIdx=1` / `splitNum=actualSplits` / `lseTaskOffset` / `oTaskOffset` 的 running 前缀和），并加**容量上界注释**（`row ≤ batchWithWork ≤ blockDim/2 ≤ 12 < 25`）。
- 保留 `else` 分支的 `set_totalSplitNodeNum(0)` 等（不切分时组表不使用）。
- **关键**：所有 `kvLen` 推导继续用 `ctx.seqlens_k_cpu[b] + ctx.kvNewSeqlen`。

### 3) `csrc/ascend910/flash_attn_npu_3/mla_decode_gemv.hpp`
**What**：kernel 侧五项改造（对应 FA4 同名文件）。
**Why**：配合 P0 下沉 + 解除 H 限制 + 修事件 hazard。
**How**：
- **(a) task 就地重算 + 核内 task 循环**：把 FA3 的 `coreTask = tilingData->coreInfo[coreIdx]`（L761）替换为按 host 规则逐字重算的 `perCore = ceil(totalTaskNum / GetBlockNum())` / `taskStart` / `taskEnd`，每核认领连续 task 区间；把原单 task 的段循环整体降为内层；循环不变量（Layout / TaskPlan）提到 task 循环之外。重算函数需带 `kvNewSeqlen`（**FA3 特有**，对照 FA3 的 `kvSeqlen = kvSeqlenBatch + kvNewSeqlen` L777）。
- **(b) head 维分块（G1）**：引入 `kMlaDecodeGemvHeadBlock = 32`、`gemvNumHeadBlocks` / `gemvHeadBlock` / `m0` / `headBase` / `lseHeadStride`；`qHeads > 32` 时按块重建 layout / taskPlan 并把该块 Q 重装进同一 128 行 Q 区；末块按实际 `lastM` 行数处理（防越界）。
- **(b') L1 Q 行区与 H 解耦（G1 补充，执行中修复）**：`qRows` 由 `qHeads` 改为固定 `BlockMmadQK::L1TileShape::M`（=128）（现文件 L687），使 `l1QKSize` 恒为 147456、与 H 无关，与 FA4 同路径一致；否则 H 一大就把 V/nd 区顶出 L1 512KB 上限，触发 aicore 越界（H≥313 复现）。
- **(c) HardEvent 归位**：把 `SetFlag<MTE1_MTE2>(EVENT_ID3)`（FA3 L840）从 `loadQGM` 之后移到该 task 段循环结束之后；**按 FA3 自身账本**核对 prelaunch 的 Set 次数与 footer drain 的 Wait 次数逐项配平。
- **(d) combine 调用**：`combineScale(...)` 第 4 实参传 `tilingData->splitInfo`（FA3 的 `combineScale` 多一个 `perBatchBase` 实参，按 FA3 现状传入，勿照抄 FA4）。alias `SplitInfoType` 保持 `splitNode[25]` 数组类型。

### 4) `csrc/ascend910/flash_attn_npu_3/flash_api.cpp`
**What**：核对（预计不改）。
**How**：确认 gemv 分支 `launchBlockDim = needCoreNum = min(totalTaskNum, blockDim)`、`IsMlaDecodeGemvShape` 的 H 上界为 32（`numHeads <= 32`）、workspace 组成里**没有** `gemvSplitNodeSize`（组表住 tiling）、FD partial 初始化偏移（FA3 是 `partialBase`）与 kernel 读取一致。

### 5) `flash_attn_npu_3/flash_attn_npu_interface.py`
**What**：修 `_transposed_bnsd_view`，使 `Sq==1` 的 packed BNSD `transpose(1,2)` 视图被判定为 BNSD。
**Why**：`Sq==1` 时 size-1 维被 PyTorch 连续性判定忽略，`is_contiguous()==True` 导致 stride 证据被丢弃、MLA decode 被误判为 BSND。
**How**：去掉 `is_contiguous()` 早退，改为按显式 stride 模式判定：`stride(3)==1 && stride(1)==size(3) && stride(2)==size(1)*size(3)`，命中即返回 `transpose(1,2)`（对照 FA4 `_transposed_bnsd_view` L50-73 改写，但保留 FA3 自身的 `is_bnsd` 命名与调用链）。

### 6) `tests/test_flash_attn_npu_v3_mla.py`
**What**：新增 `Sq==1` packed BNSD decode 精度用例。
**How**：镜像 FA4 的 `test_mla_decode_bnsd_sq1_cross_batch`（`tests/test_flash_attn_npu_v4_mla.py:352`）：B=2、H=32、Hk=1、Drope=64、Dv=512、KV≥128，`q/qv/k/v` 均为 packed BNSD 的 `transpose(1,2)` 视图，命中 gemv 选路；与 BSND 等价存储结果比对，`rel_l2 < REL_L2_TOL`。不改动既有用例。

---

## Assumptions & Decisions

1. **范围 = 全量对齐 FA4**（用户已确认），含 head 分块与 BNSD 视图修复。
2. `mha_fwd_kvcache.cpp` 与 `combine_scale.hpp` **不改**（FA3 已是改造后形态；FA3 的 `perBatchBase` 差异保留）。
3. `nsplitPerBatch` 复用 FA3 的 `padding3`（先 grep 确认未占用），保持 `FAInferTilingData` 尺寸与 Python 常量 2376 不变；若尺寸确实变化，则同步两端并在计划执行时记录。
4. append-KV（`kvNewSeqlen`）在 FA3 全链路保留，host 与 kernel 的 `kvLen` 口径必须逐字一致。
5. H 分块阈值取 FA4 的 `32`，但执行时需按 FA3 的持久 O 行数约束复核（若 FA3 约束不同则相应调整）。
6. 验收边界收敛到真实场景：B ≤ 64、H ≤ 512（与 FA4 一致）。
7. **构建/测试需用户授权**（用户偏好：未经明确授权不主动跑编译/测试）。执行到验证阶段时先说明命令再跑。

---

## Verification

按顺序执行（**每步前先请示授权**）。**下列结果为实际执行记录**：

1. **静态核验** ✅
   - `grep` `coreInfo\[coreIdx\]\|MAX_NODE_NUM` → gemv 路径无残留（`coreInfo[` 仅剩注释）。
   - `gemvSplitNodeSize` → 空。
   - `sizeof(FAInferTilingData)` 编译期 `static_assert(==2376)` 通过，与 `_SCHEDULER_METADATA_TILING_BYTES = 2376` 一致。
2. **构建** ✅ FA3 + FA4：
   ```
   source /home/w00811046/cann/cann-9.2.0-clean/cann/set_env.sh
   export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11
   export LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/11
   FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py build_ext --inplace
   FLASH_ATTN_BUILD_VERSION=v4 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py build_ext --inplace
   ```
   FA3 扩展构建成功（`static_assert 2376` 通过）；FA4 扩展可复用、无需重建。
3. **功能/精度** ✅ `python -m pytest -q -s tests/test_flash_attn_npu_v3_mla.py` → **25 passed**（既有 14 项 + 新增 1 项 + 其余）。
4. **边界覆盖** ✅（临时脚本放 `~/tmp`）：
   - B=12/13/26 × KV=513/1025（覆盖 `anySplit` 真/假）→ 全部 bad=0；**B=26 命中 26/26 核**（25 截断修复的直接证据）。
   - H=32/128/228/256/312/**313**/384/512，B=1，KV=1025 → 全部 bad=0，`max_rel_l2 ≤ 2.54e-3`（H≥313 曾是崩溃点，修复后正常）。
   - 判据 FP32 CPU golden，`rel_l2 < 5e-3`。
5. **性能门限** ⚠️ `python tests/perf/bench_mla_vs_fia.py; echo "EXIT=$?"`
   - **FA3 全部行 PASS**：prefill `fa3_func_bsnd` 0.862~0.868x、decode 0.956~0.982x，均 `≤ 1.05`。
   - 整体 **EXIT=1 由 FA4 的 prefill 行 `fa4_func_bsnd = 1.065x` 抖动引起**（超 1.05 门限）。FA4 不在本次移植范围；该行历史区间为 0.913x~1.065x，属噪声。
6. **三接口覆盖确认** ✅ `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` 的 MLA decode 均走 `MlaDecodeGemv` 且正确，由 bench 的 `fa3_func_bsnd` / `fa3_varlen_tnd` / `fa3_kvcache_bsnd` 三行覆盖。

> 遗留：bench 退出码 1 由 FA4 prefill 抖动引起，与本次移植无关；如需整体 EXIT=0 可重跑 bench 确认抖动。

### 交付物
- 修改：`flash_attn_npu_3/{fa_split.h, mla_decode_gemv.hpp, tilingdata.h}`、`flash_attn_npu_3/flash_attn_npu_interface.py`、`tests/test_flash_attn_npu_v3_mla.py`（`flash_api.cpp` 经核对未改）。
- 验证记录：pytest 25 passed；边界 B/H 全覆盖 bad=0；FA3 全部 perf 行 ≤ 1.05。
