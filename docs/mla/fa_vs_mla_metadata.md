# FA 与 MLA scheduler-metadata 统一结构

本文描述 `flash_attn_npu_3` 下普通 FlashAttention（下称 FA）与 MLA 两条 scheduler-metadata 管线的**统一结构**：入参结构、输出记录、AICPU 编译单元与载体 dtype 全部共用一份定义，差异只保留在"设备侧是否重算长度"与"调度算法"两处。

统一后的核心事实：

- **入参一份**：`FAMetadataArgs`（[fa_metadata_args.h](../../csrc/ascend910/flash_attn_npu_3/fa_metadata_args.h)），FA 用其通用段，MLA 追加一段可选字段。
- **输出一份**：`FAInferTilingData`（[tilingdata.h](../../csrc/ascend910/flash_attn_npu_3/tilingdata.h)），固定 3840 字节；MLA 记录把 header/partition 织回该结构的既有字段，再在其后追加变长尾部。
- **AICPU 一份**：[fa_metadata.aicpu](../../csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu) 单编译单元，保留两个 `__aicpu__` 入口 `ComputeFAMetadataV3` / `ComputeMlaMetadata`。
- **载体一份**：`uint8` 张量；FA 恒 `sizeof(FAInferTilingData)`，MLA 为 `MlaUnifiedBytes`。

---

## 1. 代码位置与调用链

### 1.1 共用定义

| 环节 | 位置 |
|---|---|
| 入参结构 | [fa_metadata_args.h](../../csrc/ascend910/flash_attn_npu_3/fa_metadata_args.h) `FAMetadataArgs`（通用段 + MLA 段） |
| 输出结构 | [tilingdata.h](../../csrc/ascend910/flash_attn_npu_3/tilingdata.h) `FAInferTilingData`（`sizeof == 3840`）、`MlaCorePartition`（346 字）、`kFaInferTilingBytes` |
| 统一映射 | [mla_metadata.h](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h) `MlaUnifiedSlot`（L277-289）、`MlaUnifiedTailWords` / `MlaUnifiedBytes`（L291-298）、`WriteMlaHeader`（L299-306） |
| AICPU 入口 | [fa_metadata.aicpu](../../csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu) `ComputeFAMetadataV3`、`ComputeMlaMetadata` |

### 1.2 FA 调用链

```
GetSchedulerMetadataImpl (flash_api.cpp L123-164)
  → at::empty({sizeof(FAInferTilingData)})          // uint8
  → 独立 AICPU stream + 两个 event 串行
  → ComputeFAMetadataV3<<<1, nullptr, aicpuHandle>>>(&metaArgs, sizeof(metaArgs))
  → 先 memset 全 0（连带 MLA 段清零），再逐字段赋值；FD 时调 splitBN2S1GS2
  → DataStoreBarrier, return 0
```

MLA 段在 FA 路径下由 `std::memset(tilingData, 0, sizeof(FAInferTilingData))` 一并清零，FA 无需感知 MLA 字段。

### 1.3 MLA 调用链

```
mla_forward.cpp
  → MlaRecordLayout 预算 l.words / MlaUnifiedBytes(l)
  → at::empty({MlaUnifiedBytes(l)}, kByte)          // uint8 × bytes
  → FillMlaArgs(FAMetadataArgs&, ...)               // 与 FA 共用同一入参结构
  → RunMlaLaunch（同一 stream）
  → ComputeMlaMetadata<<<1, nullptr, a.stream>>>
      → BuildNonquantMetadata
          → MlaContextFromArgs（从 FAMetadataArgs 重建几何）
          → MlaRecordLayout（校验 + 默认布局）
          → ParseUsedLengths（读 4 个设备数组）
          → PrepareMlaGeometry + 带 tileM/tileN 重算布局（absorbed）
          → BuildWindowSchedule 或 BuildGqaSchedule
          → memset(out, 0, MlaUnifiedBytes(l))；span/前缀写尾部、partition 写 mlaPartition
          → WriteMlaHeader（最后写，清除 status 进行中标记）
  → LaunchMla<dtype>
```

---

## 2. 入参结构 `FAMetadataArgs`

FA 与 MLA 共用同一个扁平结构体。MLA 需要的字段统一以 `mla` 前缀追加在尾部，`MlaContextFromArgs` 负责把它还原成 MLA 内部使用的 `MlaContext`。

| 概念 | FA 字段 | MLA 还原来源 |
|---|---|---|
| batch | `batch` | `context.batch` |
| Q 头数 | `numHeads` | `context.heads` |
| KV 头数 | `numHeadsK` | `context.kvHeads` |
| 头维 | `embeddingSize` / `embeddingSizeV` | `embeddingSizeV` → `context.valueDim`（128 / 512） |
| Q 长度 | `maxQSeqlen` + `seqlensQAddr` | `maxQSeqlen` → `context.sq` + `mlaCuQuery` / `mlaUsedQuery` |
| KV 长度 | `cacheSeqlensAddr` + `kvNewSeqlen` | `mlaMaxKv` → `context.sk` + `mlaCuKv` / `mlaUsedKv` |
| 布局 | `isVarlen`、`isVarlenKv` | `mlaLayoutCode`（0~4）+ `mlaTotalQ` / `mlaTotalK` |
| 分页 | `pagedKV`、`blockSize`、`numBlocks`、`maxNumBlocksPerBatch` | `mlaPageSize` + `mlaCache*` 展平描述符 |
| 核数 | `blockDim` | `context.physicalCores` + `mlaMargin` |
| 掩码 | `maskType` + `windowSizeLeft/Right` | `context.window.{kind,left,right}` + `mlaHasSoftcap` |
| 拆分 | `numSplits` | `context.splits` |
| dtype | 无 | `mlaDtype` → `context.dtype`（必须 fp16/bf16） |
| 输出地址 / 长度 | `metaOutAddr`（固定 `sizeof`） | `metaOutAddr` + `mlaWords` |
| workspace | 无（输出里写 `workSpaceSize`） | `mlaWorkspaceBytes`（消费侧校验） |

MLA 专属字段：`mlaWords`、`mlaLayoutCode`、`mlaTotalQ/mlaTotalK`、`mlaPageSize`、`mlaUsedFlags`、`mlaMaxKv`、`mlaPacked`、`mlaMargin`、`mlaInferTotalQ`、`mlaGenerate`、`mlaDtype`、`mlaHasSoftcap`、`mlaCuQuery/mlaCuKv`、`mlaUsedQuery/mlaUsedKv`、`mlaChecked`、`mlaWorkspaceBytes`、`mlaCache*`（页表 / K / V 的地址、count、stride、span）。

长度语义（MLA 侧）：capacity（地址步长）/ length（mask 范围，`mlaUsedQuery/mlaUsedKv`，缺省等于 capacity）/ start（物理起点）三层，由 `ParseUsedLengths` 解析并强校验（前缀首项为 0、末项等于 `totalTokens`、单调不减、capacity ≤ maxLength、used ≤ capacity）。FA 侧只有容量即长度一层。

---

## 3. 输出结构 `FAInferTilingData` + 变长尾部

### 3.1 物理布局

`FAInferTilingData` 固定 3840 字节（960 字），其中前 594 字是原有 FA 字段，末尾 366 字为 MLA 段：

| 物理字区间 | 内容 |
|---|---|
| `0 ~ 593` | 原有 FA 字段（`numHeads` … `splitInfo[25]`） |
| `594 ~ 613` | MLA 标量段：`mlaWords`、`mlaDtype`、`mlaStatus`、`mlaLayout`、`mlaPacked`、`mlaMargin`、`mlaPhysicalCores`、`mlaTotalQ`、`mlaTotalK`、`mlaPageSize`、`mlaUsedFlags`、`mlaValueDim`、`mlaHasSoftcap`、`mlaM`、`mlaN`、`mlaCores`、`mlaHeadGroups`、`mlaQueryCapacity`、`mlaKvCapacity`、`mlaAlgorithm` |
| `614 ~ 959` | `MlaCorePartition`（346 字） |
| `960 ~ 960+8B-1` | 每 batch 8 字 span（B = batch） |
| `960+8B ~ …` | 宽 M 组前缀（`2·(B·headGroups+1)` 字） |

`static_assert` 固定 ABI（[tilingdata.h](../../csrc/ascend910/flash_attn_npu_3/tilingdata.h) L336-340）：`offsetof(mlaWords) == 2376`（第 594 字）、`sizeof(MlaCorePartition) == 346·4`、`sizeof(FAInferTilingData) == 3840`。

### 3.2 逻辑记录与 `MlaUnifiedSlot`

MLA 记录在**逻辑**上仍是原来的四段（header / spans / partition / 宽前缀），但物理上被织入 `FAInferTilingData`：header 分散落到 FA 的既有字段与 MLA 标量段，spans 与前缀落进固定区之后的尾部，partition 落进 `mlaPartition`。`MlaUnifiedSlot(batch, word)`（[mla_metadata.h](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h) L277-289）是唯一的逻辑字→物理字映射：

```cpp
spanEnd = 32 + 8*batch;
word < 32                    -> kMlaUnifiedHeaderSlot[word]            // 头 32 字散落
word < spanEnd               -> 960 + (word - 32)                      // spans 追加区
word < spanEnd + 346         -> 614 + (word - spanEnd)                 // MlaCorePartition
else                         -> 960 + 8*batch + (word - spanEnd - 346) // 宽前缀追加区
```

生产者与消费者都只经此函数定位，因此不感知固定区/尾部的分界。

HEAD 32 字的物理落点（`kMlaUnifiedHeaderSlot`，[mla_metadata.h](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h) L240-273）：

| 逻辑字 | 字段 | 物理字 |
|---|---|---|
| 0/1/2 | `words` / `dtype` / `status` | 594 / 595 / 596 |
| 3 | `batch` | 8（FA `batch`） |
| 4 | `sq` | 5（FA `maxQSeqlen`） |
| 5 | `sk` | 6（FA `maxKvSeqlen`） |
| 6 | `heads` | 0（FA `numHeads`） |
| 7 | `kvHeads` | 7（FA `kvHeads`） |
| 8 | `splits` | 41（FA `numSplits`） |
| 9~18 | `physicalCores`…`hasSoftcap` | 597~606（MLA 标量段） |
| 19 | `mask` | 12（FA `maskType`） |
| 20/21 | `left` / `right` | 30 / 32（FA `windowSizeLeft/Right`） |
| 22~28 | `m`…`algorithm` | 607~613（MLA 标量段） |
| 29~31 | `reserved[3]` | 无落点，恒 0 |

每 batch span（8 字，追加区，偏移同原语义）：

```
+0/+1 queryStart 低/高    +2/+3 kvStart 低/高
+4 queryCapacity          +5 kvCapacity
+6 queryLength            +7 kvLength
```

`MlaCorePartition` 内部 word 偏移（物理字 614 起）：

| 相对 word | 字段 |
|---|---|
| 0~3 | `mask`、`left`、`right`、`usedCoreNum` |
| 4~29 | `bEnd[26]` |
| 30~55 | `mEnd[26]` |
| 56~81 | `nEnd[26]` |
| 82~85 | `fdHeads`、`maxSplits`、`vectors`、`fdRows` |
| 86~111 | `fdBatch[26]` |
| 112~137 | `fdM[26]` |
| 138~163 | `fdSplits[26]` |
| 164~189 | `coreSplitStart[26]` |
| 190~215 | `fdParts[26]` |
| 216~241 | `fdLastRows[26]` |
| 242~293 | `vectorHeadEnd[52]` |
| 294~345 | `vectorPartEnd[52]` |

组前缀语义：`count = ceil(queryLength × g / m)`，跨 batch 连续累加后写成 64 位（低字在前）。

### 3.3 尺寸

| 维度 | FA | MLA |
|---|---|---|
| 张量 dtype | `uint8`（`at::kByte`） | `uint8`（`at::kByte`） |
| 大小 | 固定 `sizeof(FAInferTilingData)` = **3840 字节** | `MlaUnifiedBytes(l)` = `3840 + 4·tailWords` |
| tailWords | — | `l.words - 32 - 346 = 8B + 2·(B·headGroups + 1)` |
| 谁决定 | 编译期常量 | 运行期 `MlaRecordLayout` |
| 写入方式 | `memset` + 逐字段赋值 | 本地拼装 span/前缀/partition 后写回固定区与尾部，最后 `WriteMlaHeader` |
| 自描述 | 无（FA 段全 0） | `mlaWords` / `mlaDtype` / `mlaStatus` |
| 防腐 | 无 | `mlaStatus` 脏标记 + 消费侧逐字重建比对 |
| 返回值 | 恒 `0` | `ok ? 0 : 1` |

### 3.4 数值例子

**FA**：`batch=2, numHeads=8, numHeadsK=1, dim=128, maxQSeqlen=1, sk=4096, blockDim=20, paged=1, varlen=1, causal, numSplits=1`

```
[0]=8  [1]=128  [2]=128  [3]=0  [4]=0  [5]=1  [6]=4096  [7]=1
[8]=2  [9]=0    [10]=1   [11]=2 [12]=1
[14,15]=15,728,640  [16,17]=7,864,320  [18,19]=15,728,640  [20,21]=15,728,640
[22,23]=55,050,240  [24,25]=[26,27]=0  [28]=0
[30,31]=[32,33]=0xFFFFFFFFFFFFFFFF
[34,35]=[36,37]=[38]=[39]=0（若未触发 FD）
[40]=1  （groupSize=8≤16 且 4096≥1024，触发 FD）
[41]=1  [42]=0  [43]=0
[44~593]：FD 触发时写 coreInfo/splitInfo，否则全 0
[594~959]：MLA 段全 0
```

**MLA**：`batch=1, sq=64, sk=64, heads=kvHeads=1, valueDim=128, layout=0, splits=0, physicalCores=20, dtype=2`

`words = 32 + 8 + 346 + 2 + 2 = 390`；`tailWords = 8 + 4 = 12`；总字节 `3840 + 48 = 3888`。

逻辑记录（经 `MlaUnifiedSlot` 落到物理字）：

```
header: [0]=390 [1]=2 [2]=0 [3]=1 [4]=64 [5]=64 [6]=1 [7]=1 [8]=0 [9]=20
        [10]=0 [11]=0 [12]=0 [13]=0 [14]=0 [15]=0 [16]=0 [17]=128 [18]=0
        [19]=0 [20]=0 [21]=0 [22]=512 [23]=64 [24]=20 [25]=1 [26]=64 [27]=64 [28]=0
spans:  [32..39] = 0,0,0,0,64,64,64,64
partition: [40]=0(mask) [41]=0(left) [42]=0(right) [43]=1(usedCoreNum)
           [44]=1(bEnd[0]) [70]=0(mEnd[0]) [82]=0(fdHeads) [85]=8(fdRows)
prefix: [388]=1 [389]=0（共 1 组）
```

物理落点示例：`words`→594、`dtype`→595、`batch`→8、`heads`→0、`splits`→41、`mask`→12、partition 起于 614、span 起于 960、前缀起于 `960+8B`。

---

## 4. 校验与错误处理

| 项 | FA | MLA |
|---|---|---|
| 入参校验 | 无 | 全量：地址对齐/重叠/溢出、`words` 与 `MlaUnifiedBytes` 一致、布局组合、容量上限 |
| host 侧校验 | 无（消费侧断言 `nbytes == MetadataBytes`） | `CheckMetadataBuffer(tensor, device, MlaUnifiedBytes(l), ...)`（要求 `kByte`、`numel() == bytes`、连续） |
| 错误返回 | 恒 `0` | `false` → AICPU 返回 1；`true` → 软跳过 |
| 半成品防护 | 无 | 先 `memset` 全 0 + `mlaStatus=1`，成功才 `WriteMlaHeader` 覆盖为 `status=0` |
| 消费侧校验 | 无 | `mlaStatus` 检查 + 重建 header 逐字比对 + `l.bytes ≤ mlaWorkspaceBytes` |
| 附加校验输出 | 无 | 可选 `mlaChecked`（`MlaCheckedAutoSplitSchedule`） |
| 异常处理 | 无 | `try/catch std::bad_alloc`，异常不外泄到 AICPU |

MLA 的三层防线：① host 预算 `MlaUnifiedBytes` 并预分配 workspace；② AICPU 生成时重算布局并核对 `words`；③ 消费侧 `ReadMlaMetadata` 重建 header 比对、校验 workspace 容量。

---

## 5. 消费侧

| 消费方 | 读法 |
|---|---|
| FA kernel | 启动参数直接传缓冲区，按 `FAInferTilingData*` 强转读取 |
| MLA kernel（AIC） | [mla_kernel.hpp](../../csrc/ascend910/flash_attn_npu_3/mla/mla_kernel.hpp) `MetadataReader`，`data.GetValue(MlaUnifiedSlot(batch, word))` |
| MLA AICPU 自校验 | [fa_metadata.aicpu](../../csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu) `RecordWords`（经 `MlaUnifiedSlot`）与 `RawWords`（page table 裸读） |
| MLA 拆分 | [mla_auto_split_dispatch_impl.hpp](../../csrc/ascend910/flash_attn_npu_3/mla/mla_auto_split_dispatch_impl.hpp) span 区实参 `metadata + 4 * kMlaUnifiedFixedWords` |

page table 是独立裸 `int32` 数组，不参与映射，故单列 `RawWords`。

---

## 6. 同步与 stream

| 项 | FA | MLA |
|---|---|---|
| metadata 执行 stream | 独立 AICPU stream（`getNPUStreamFromPool`） | 与 kernel 同一 stream（`a.stream`） |
| 同步手段 | 两个 event：`inputReady`（cur→aicpu）、`metadataDone`（aicpu→cur） | 同流顺序执行，无需 event |
| 生命周期登记 | `recordStream` 登记 meta / seqlens 到 aicpu stream | 同流天然一致 |
| 启动包裹 | `OpCommand::RunOpApiV2` | `RunMlaLaunch` |

FA 做成独立 stream 是为了让 metadata 生成与主 stream 上的其它工作重叠；MLA 同流是因为 metadata 是 kernel 的直接输入且需要保证"先生成后消费"的强顺序。

---

## 7. 分核与布局算法（仍按各自算法，不共用）

| 维度 | FA | MLA |
|---|---|---|
| 切块来源 | `qNBlockTile = min(128/qSeqlen, groupSize)`，`Q_TILE_CEIL=128` | `MlaRecordLayout` 推导 `m/n`；absorbed 由 `PrepareMlaGeometry` 校准 |
| FD 判定 | 启发式布尔 | 算法副产物：`BuildGqaSchedule` 均衡时自然沿 S2 拆；`splits==1` 的 absorbed 走 `BuildWindowSchedule` |
| 切分输出 | FD 时 `splitBN2S1GS2` 写 `coreInfo/splitInfo` | 写 `MlaCorePartition` 的 `bEnd/mEnd/nEnd` 三级端点 + FD 表 |
| 负载模型 | 任务数均分 | 三级坐标（BN2 / S1G / S2）贪心切分 + 成本模型 |
| GQA | 由 `groupSize` 推导 | `packed` 时 `g = heads/kvHeads`，`headGroups = heads/g` |

---

## 8. 与 ops-transformer 的对齐

ops-transformer 的 FA 与 MLA metadata 也是同构的（共享 `optiling::detail::FaMetadata` 与 `FlashMlaCommon::{HEAD,FA,FD}_METADATA_STRIDE`，head 自描述 sectionNum/isFd/mBaseSize/s2BaseSize/aicNum/aivNum/outputLayout）。本仓库采纳其三条原则：

1. **一份结构定义**：FA 与 MLA 共用 `FAInferTilingData` / `FAMetadataArgs`。
2. **自描述 head**：`mlaWords` / `mlaDtype` / `mlaStatus` 及几何字段。
3. **定长 stride、按核寻址**：`MlaCorePartition` 定长 346 字，`MlaUnifiedSlot` 定长映射。

保留的差异：ops-transformer 走 aclnn 框架并允许设备侧重算长度；本仓库是直调（`<<<>>>`），FA 侧 host 提供 `cu_seqlens`，MLA 侧设备侧重算长度并自校验，均保持现状。

---

## 9. 附录：一句话对照

| 关注点 | FA | MLA |
|---|---|---|
| 入参结构 | `FAMetadataArgs` 通用段 | `FAMetadataArgs` + `mla*` 段 |
| 输出结构 | `FAInferTilingData`（3840 B） | `FAInferTilingData`（3840 B）+ 变长尾部 |
| 载体 dtype | `uint8` | `uint8` |
| 读几个数组 | 2 | 4（+ page table 裸读） |
| 谁保证正确 | host | host + 设备 + 消费端三层 |
| 失败如何表达 | 不表达 | `false` / `mlaStatus=1` / `mlaChecked` |
| FD 决策 | 启发式阈值 | 调度算法副产物 |
| 数据读取方式 | `FAInferTilingData*` 强转 | `MlaUnifiedSlot` 逻辑字映射 |
| AICPU 编译单元 | `fa_metadata.aicpu` | 同文件（`ComputeMlaMetadata` 入口） |
