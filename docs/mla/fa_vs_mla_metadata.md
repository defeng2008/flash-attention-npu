# FA 与 MLA metadata 管线对比分析

本文对比 `flash_attn_npu_3` 下普通 FlashAttention（下称 FA）与 MLA 两条 scheduler-metadata 生成管线：入口、输入参数、输出二进制布局、分核算法、校验策略与同步方式。

两条管线的核心差异可以一句话概括：

- **FA 的 metadata 是“纯 tiling 参数包”**——固定 2376 字节、无自描述、无校验、无错误通道。
- **MLA 的 metadata 是“自包含调度快照”**——变长、带长度/dtype/状态字、含每 batch 跨度与每核端点，生成端与消费端双重校验。

---

## 1. 代码位置与调用链

### 1.1 FA

| 环节 | 位置 |
|---|---|
| 参数结构 | [fa_metadata_args.h](../../csrc/ascend910/flash_attn_npu_3/fa_metadata_args.h) `FAMetadataArgs`（L51-75） |
| 尺寸/偏移 | [fa_metadata_args.h](../../csrc/ascend910/flash_attn_npu_3/fa_metadata_args.h) `MetadataBytes`（L22-25）、`TilingOffset`（L18-21） |
| 输出结构 | [tilingdata.h](../../csrc/ascend910/flash_attn_npu_3/tilingdata.h) `FAInferTilingData`（L36-71）、`coreNode`（L12-23）、`splitNode`（L25-34） |
| AICPU 入口 | [fa_metadata.aicpu](../../csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu) `ComputeFAMetadataV3`（L37-141） |
| FD 拆分 | [fa_split.h](../../csrc/ascend910/flash_attn_npu_3/fa_split.h) `splitBN2S1GS2`（L524-551） |
| host 编排 | [flash_api.cpp](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp) `GetSchedulerMetadataImpl`（L123-164） |
| 消费 | kernel 启动参数直接传该缓冲区，按 `FAInferTilingData*` 强转读取 |

调用链：

```
GetSchedulerMetadataImpl
  → at::empty({sizeof(FAInferTilingData)})          // kByte
  → 独立 AICPU stream + 两个 event 串行
  → ComputeFAMetadataV3<<<1, nullptr, aicpuHandle>>>(&metaArgs, sizeof(metaArgs))
  → 先 memset 全 0，再逐字段赋值；FD 时调 splitBN2S1GS2
  → DataStoreBarrier, return 0
```

### 1.2 MLA

| 环节 | 位置 |
|---|---|
| 参数结构 | [mla_metadata.h](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h) `MlaRecordArgs`（L235-242）、`MlaContext`（L44-49） |
| 布局规划 | 同文件 `MlaRecordLayout`（L114-200）、`MlaHeader`（L201-234） |
| 输出结构 | 同文件 `MlaRecordHeader`（L56-62）、`MlaNativePartition`（L63-70）、`MlaNativeWord`（L72-89） |
| AICPU 入口 | [mla_metadata.aicpu](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.aicpu) `ComputeMlaMetadata`（L51-70） |
| 生成主体 | [mla_metadata.cpp](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.cpp) `BuildNonquantMetadata`（L16-155） |
| 长度解析 | [mla_tiling.cpp](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tiling.cpp) `ParseUsedLengths`（L571-583）→ `Parse` |
| 分核 | 同文件 `PrepareMlaGeometry`、`BuildWindowSchedule`、`BuildGqaSchedule` |
| host 编排 | [mla_forward.cpp](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)（L240-330） |
| 消费 | [mla_metadata_consumer.h](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata_consumer.h) `ReadMlaMetadata`，按 word 偏移直读 |

调用链：

```
mla_forward.cpp
  → MlaRecordLayout 预算 l.words / l.bytes
  → at::empty({l.words}, kInt)                      // int32 × words
  → RunMlaLaunch（同一 stream）
  → ComputeMlaMetadata<<<1, nullptr, a.stream>>>
      → BuildNonquantMetadata
          → MlaRecordLayout（校验 + 默认布局）
          → ParseUsedLengths（读 4 个设备数组）
          → PrepareMlaGeometry + 带 tileM/tileN 重算布局（absorbed）
          → BuildWindowSchedule 或 BuildGqaSchedule
          → 组装 result 后整体 std::copy 到 out
  → LaunchMla<dtype>
```

---

## 2. 输入参数对比

### 2.1 结构体形态

| 维度 | `FAMetadataArgs` | `MlaRecordArgs` |
|---|---|---|
| 形态 | 扁平 22 字段 | 嵌套：`MlaContext` + 8 个字段 |
| 地址型入参 | 3 个（`seqlensQAddr`、`cacheSeqlensAddr`、`metaOutAddr`） | 7~10 个（`cuQuery`、`cuKv`、`usedQuery`、`usedKv`、`output`、`checked`、`cache.{table,k,v}`） |
| 长度数组 | 2 个 | 4 个 |
| 期望输出长度 | 无 | `words`（host 预算，设备必须核对一致） |
| 附加控制位 | 无 | `inferTotalQ`、`checked`、`generate` |

### 2.2 字段语义映射

| 概念 | FA | MLA |
|---|---|---|
| batch | `batch` | `context.batch` |
| Q 头数 | `numHeads` | `context.heads` |
| KV 头数 | `numHeadsK` | `context.kvHeads` |
| 头维 | `embeddingSize` / `embeddingSizeV` | `context.valueDim`（128 / 512） |
| Q 长度 | `maxQSeqlen` + `seqlensQAddr` | `context.sq` + `cuQuery` / `usedQuery` |
| KV 长度 | `cacheSeqlensAddr` + `kvNewSeqlen` | `context.sk` + `cuKv` / `usedKv` |
| 布局 | `isVarlen`、`isVarlenKv` | `context.layout`（0~4）+ `totalQ` / `totalK` |
| 分页 | `pagedKV`、`blockSize`、`numBlocks`、`maxNumBlocksPerBatch` | `context.pageSize` + `cache` 描述符 |
| 核数 | `blockDim` | `context.physicalCores` + `context.margin` |
| 掩码 | `maskType`（0/1/2）+ `windowSizeLeft/Right` | `context.window.{kind,left,right}` + `hasSoftcap` |
| 拆分 | `numSplits` | `context.splits` |
| GQA | 由 `numHeads/numHeadsK` 推导 `groupSize` | `context.packed` 显式给出 |
| dtype | 无 | `context.dtype`（必须 fp16/bf16） |
| 输出地址 | `metaOutAddr` | `output` |
| 输出长度 | 无（固定 `sizeof`） | `words` |
| workspace | 无（反过来在输出里写 `workSpaceSize`） | `context.workspaceBytes`（消费侧校验） |
| Append-KV | `kvNewSeqlen`、`kvCacheSeqlen` | 无 |

### 2.3 只在一侧存在的入参

- **仅 FA**：`numBlocks`、`blockSize`、`maxNumBlocksPerBatch`、`isVarlenKv`、`kvNewSeqlen`、`kvCacheSeqlen`。
- **仅 MLA**：`words`、`inferTotalQ`、`checked`、`generate`、`cache`、`margin`、`totalQ/totalK`、`hasSoftcap`、`dtype`、`workspaceBytes`。

### 2.4 长度语义差异（关键）

FA 只有“序列长度”一个概念，容量即长度；长度要么取标量 `maxQSeqlen`，要么逐 batch 读 `seqlensQ/cacheSeqlens`。

MLA 区分三层：

| 概念 | 来源 | 用途 |
|---|---|---|
| 分配容量 capacity | TND 取前缀差、BSND 取 `maxLength` | 地址步长 |
| 实际长度 length | `usedQuery/usedKv`，缺省等于 capacity | mask 范围与计算量 |
| 物理起点 start | TND 取前缀、BSND 取 `b × maxLength` | 数据定位 |

这套区分由 `ParseUsedLengths` 完成，并附带强校验：前缀首项为 0、末项等于 `totalTokens`、单调不减、capacity ≤ maxLength、used ≤ capacity。

---

## 3. 输出参数对比

### 3.1 总览

| 维度 | FA | MLA |
|---|---|---|
| 目标地址 | `metaOutAddr + TilingOffset`（`TilingOffset ≡ 0`） | `output` |
| 大小 | 固定 `sizeof(FAInferTilingData)` = **2376 字节** | 变长 `words × 4` 字节 |
| 谁决定 | 编译期常量 | `words = 32 + 8B + 346 + 2 + 2·B·H` |
| 写入方式 | `memset` + 逐字段赋值 | 本地 `result` 拼装后整块 `std::copy` |
| 自描述 | 无 | `words` / `dtype` / `status` |
| 防腐 | 无 | `status` 脏标记 + 消费侧逐字重建比对 |
| 返回值 | 恒 `0` | `ok ? 0 : 1` |

### 3.2 FA 的 2376 字节布局

`memset(tilingData, 0, 2376)` 后逐字段赋值，未赋值字节必为 0。

标量区（0 ~ 175）：

| 偏移 | 类型 | 字段 | 写入 |
|---|---|---|---|
| 0 | u32 | `numHeads` | 入参直传 |
| 4 | u32 | `embeddingSize` | 直传 |
| 8 | u32 | `embeddingSizeV` | 直传 |
| 12 | u32 | `numBlocks` | 直传 |
| 16 | u32 | `blockSize` | 直传 |
| 20 | u32 | `maxQSeqlen` | 直传 |
| 24 | u32 | `maxKvSeqlen` | `max(cacheSeqlens[b] + kvNewSeqlen)` |
| 28 | u32 | `kvHeads` | `numHeadsK` |
| 32 | u32 | `batch` | 直传 |
| 36 | u32 | `maxNumBlocksPerBatch` | 直传 |
| 40 | u32 | `firstBatchTaskNum` | 第 0 个 batch 任务数 |
| 44 | u32 | `totalTaskNum` | 任务数累加 |
| 48 | u32 | `maskType` | 直传 |
| 52 | 4B | 对齐填充 | 0 |
| 56 | u64 | `mm1OutSize` | `blockDim×65536×4×3` |
| 64 | u64 | `smOnlineOutSize` | `blockDim×65536×2×3` |
| 72 | u64 | `mm2OutSize` | `blockDim×65536×4×3` |
| 80 | u64 | `UpdateSize` | `blockDim×65536×4×3` |
| 88 | u64 | `workSpaceSize` | 四项之和；FD 再加 `splitLse + splitO` |
| 96 | u64 | `padding1` | 0 |
| 104 | u64 | `padding2` | 0 |
| 112 | u32 | `padding3` | 0 |
| 116 | 4B | 对齐填充 | 0 |
| 120 | i64 | `windowSizeLeft` | 无窗口为 -1 → `0xFFFF...FF` |
| 128 | i64 | `windowSizeRight` | 同上 |
| 136 | u64 | `splitLseTotalSize` | 仅 FD |
| 144 | u64 | `splitOTotalSize` | 仅 FD |
| 152 | u32 | `totalSplitNodeNum` | 仅 FD |
| 156 | u32 | `needCoreNum` | 仅 FD |
| 160 | u32 | `flashDecodeFlag` | 0 / 1 |
| 164 | u32 | `numSplits` | `>0 ? : 1`，不会为 0 |
| 168 | u32 | `kvNewSeqlen` | 直传 |
| 172 | u32 | `kvCacheSeqlen` | 直传 |

`coreInfo[25]`（偏移 176，每项 48 字节）：

```
+0  startBIdx   +4  startN1Idx   +8  startS1Idx   +12 startS2Idx
+16 endBIdx     +20 endN1Idx     +24 endS1Idx     +28 endS2Idx
+32 firstSplitKVTaskLseOffset(i64)
+40 firstSplitKVTaskOOffset(i64)
```

`splitInfo[25]`（偏移 1376，每项 40 字节）：

```
+0  batchIdx  +4  headStartIdx  +8  headEndIdx  +12 qStartIdx  +16 qEndIdx  +20 splitNum
+24 lseTaskOffset(i64)   +32 oTaskOffset(i64)
```

合计 176 + 1200 + 1000 = 2376 字节。

### 3.3 MLA 的 words × 4 布局

```
word 0 ~ 31                      MlaRecordHeader（32 字 = 128 B）
word 32+8b ~ 39+8b               每 batch 8 字 span
word tail ~ tail+345             MlaNativePartition（346 字 = 1384 B），tail = 32 + 8·batch
word tail+346 / +347             2 个保留字，恒 0
word tail+348 ~ words-1          batch·headGroups 个 64 位组前缀（高低 32 位分两字）
```

Header（word 序号即顺序）：

| word | 字段 | 说明 |
|---|---|---|
| 0 | `words` | 总字数，自描述 |
| 1 | `dtype` | 1=fp16，2=bf16 |
| 2 | `status` | 生成中=1，成功=0 |
| 3 | `batch` | |
| 4 | `sq` | |
| 5 | `sk` | |
| 6 | `heads` | |
| 7 | `kvHeads` | |
| 8 | `splits` | |
| 9 | `physicalCores` | |
| 10 | `margin` | |
| 11 | `packed` | |
| 12 | `layout` | 0~4 |
| 13 | `totalQ` | |
| 14 | `totalK` | |
| 15 | `pageSize` | |
| 16 | `usedFlags` | bit0=Q、bit1=KV |
| 17 | `valueDim` | 128 / 512 |
| 18 | `hasSoftcap` | |
| 19 | `mask` | `window.kind` |
| 20 | `left` | `(uint32)window.left` |
| 21 | `right` | 同上 |
| 22 | `m` | 校准后 M 块 |
| 23 | `n` | 校准后 N 块 |
| 24 | `cores` | 可用核数 |
| 25 | `headGroups` | |
| 26 | `queryCapacity` | 各 batch Q 容量最大值 |
| 27 | `kvCapacity` | 各 batch KV 容量最大值 |
| 28 | `algorithm` | 恒 0 |
| 29~31 | `reserved[3]` | 恒 0 |

每 batch span（8 字）：

```
+0/+1 queryStart 低/高    +2/+3 kvStart 低/高
+4 queryCapacity          +5 kvCapacity
+6 queryLength            +7 kvLength
```

`MlaNativePartition` 内部 word 偏移：

| 相对 word | 字节 | 字段 |
|---|---|---|
| 0~3 | 0~15 | `mask`、`left`、`right`、`usedCoreNum` |
| 4~29 | 16~119 | `bEnd[26]` |
| 30~55 | 120~223 | `mEnd[26]` |
| 56~81 | 224~327 | `nEnd[26]` |
| 82~85 | 328~343 | `fdHeads`、`maxSplits`、`vectors`、`fdRows`（恒 8） |
| 86~111 | 344~447 | `fdBatch[26]` |
| 112~137 | 448~551 | `fdM[26]` |
| 138~163 | 552~655 | `fdSplits[26]` |
| 164~189 | 656~759 | `coreSplitStart[26]` |
| 190~215 | 760~863 | `fdParts[26]` |
| 216~241 | 864~967 | `fdLastRows[26]` |
| 242~293 | 968~1175 | `vectorHeadEnd[52]` |
| 294~345 | 1176~1383 | `vectorPartEnd[52]` |

组前缀语义：`count = ceil(queryLength × g / m)`，跨 batch 连续累加后写成 64 位（低字在前）。

### 3.4 数值例子

**FA**：`batch=2, numHeads=8, numHeadsK=1, dim=128, maxQSeqlen=1, sk=4096, blockDim=20, paged=1, varlen=1, causal, numSplits=1`

```
[0]=8  [4]=128  [8]=128  [12]=0  [16]=0  [20]=1  [24]=4096  [28]=1
[32]=2 [36]=0   [40]=1   [44]=2  [48]=1
[56]=15,728,640  [64]=7,864,320  [72]=15,728,640  [80]=15,728,640
[88]=55,050,240  [96]=[104]=0  [112]=0
[120]=[128]=0xFFFFFFFFFFFFFFFF
[136]=[144]=[152]=[156]=0（若未触发 FD）
[160]=1  （groupSize=8≤16 且 4096≥1024，触发 FD）
[164]=1  [168]=0  [172]=0
176~2375：FD 触发时写 coreInfo/splitInfo，否则全 0
```

**MLA**：`batch=1, sq=64, sk=64, heads=kvHeads=1, valueDim=128, layout=0, splits=0, physicalCores=20, dtype=2`

`words = 32 + 8 + 346 + 2 + 2 = 390`，总 1560 字节。

```
[0]=390  [1]=2  [2]=0  [3]=1  [4]=64  [5]=64  [6]=1  [7]=1  [8]=0  [9]=20
[10]=0  [11]=0  [12]=0  [13]=0  [14]=0  [15]=0  [16]=0  [17]=128  [18]=0
[19]=0  [20]=0  [21]=0  [22]=512  [23]=64  [24]=20  [25]=1  [26]=64  [27]=64
[28]=0  [29..31]=0
[32..39]=0,0,0,0,64,64,64,64
[40]=0(mask)  [41]=0(left)  [42]=0(right)  [43]=1(usedCoreNum)
[44]=1(bEnd[0])  [70]=0(mEnd[0])  [82]=0(fdHeads)  [85]=8(fdRows)
[86..385]=0
[386][387]=0（保留）   [388]=1  [389]=0（组前缀，共 1 组）
```

---

## 4. 分核与布局算法

| 维度 | FA | MLA |
|---|---|---|
| 切块来源 | `qNBlockTile = min(128/qSeqlen, groupSize)`，`Q_TILE_CEIL=128` | `MlaRecordLayout` 推导 `m/n`；absorbed 由 `PrepareMlaGeometry` 校准 |
| FD 判定 | 启发式布尔：`paged && varlen && !BAND && maxQ·groupSize≤128 && maxQ≤16 && maxKv≥1024 && minQ>0 && (长/短序列)` | 算法副产物：`BuildGqaSchedule` 均衡时自然沿 S2 拆；`splits==1` 的 absorbed 走无 FD 的 `BuildWindowSchedule` |
| 切分输出 | FD 时 `splitBN2S1GS2` 写 `coreInfo/splitInfo` 数组 | 恒定输出 `bEnd/mEnd/nEnd` 三级端点 + FD 表 |
| 负载模型 | 任务数均分（`perCoreTaskNum`） | 三级坐标贪心切分 + 成本模型（`maxCost`） |
| GQA | 由 `groupSize` 推导 | `packed` 时 `g = heads/kvHeads`，`headGroups = heads/g`，组前缀按组累加 |
| 对齐约束 | 无 pageSize 参与 | `n` 需按 `pageSize` 向下对齐；`layout≥3` 必须带 `pageSize` |

MLA 三级坐标：`BN2`（batch × kv head 组）/ `S1G`（Q 行块）/ `S2`（KV 块），端点三元组构成每个核的工作区间，末端点固定 `(B,0,0)`。

---

## 5. 校验与错误处理

| 项 | FA | MLA |
|---|---|---|
| 入参校验 | 无 | 全量：地址对齐/重叠/溢出、`words` 一致、布局组合、容量上限、32 位字数不溢出 |
| 错误返回 | 恒 `0` | `false` → AICPU 返回 1；`true` → 软跳过 |
| 半成品防护 | 无 | 先写 `status=1` + 全 0，成功才整体覆盖为 `status=0` |
| 消费侧校验 | 仅长度断言（`nbytes == MetadataBytes`） | `status` 检查 + 重建 header 逐字比对 + `l.bytes ≤ workspaceBytes` |
| 附加校验输出 | 无 | 可选 `checked`（`MlaCheckedAutoSplitSchedule`） |
| 异常处理 | 无 | `try/catch std::bad_alloc`，异常不外泄到 AICPU |

MLA 的三层防线：

1. host 预算 `words` 并预分配 workspace；
2. AICPU 生成时重算布局并核对 `words`；
3. 消费侧 `ReadMlaMetadata` 重建 header 比对、校验 workspace 容量。

FA 只有第 1 层（host 负责正确），且断言发生在回传时（`flash_api.cpp` L513）。

---

## 6. 同步与 stream

| 项 | FA | MLA |
|---|---|---|
| metadata 执行 stream | **独立 AICPU stream**（`getNPUStreamFromPool`） | 与 kernel 同一 stream（`a.stream`） |
| 同步手段 | 两个 event：`inputReady`（cur→aicpu）、`metadataDone`（aicpu→cur） | 同流顺序执行，无需 event |
| 生命周期登记 | `recordStream` 登记 meta / seqlens 到 aicpu stream | 同流天然一致 |
| 启动包裹 | `OpCommand::RunOpApiV2` | `RunMlaLaunch` |

FA 做成独立 stream 是为了让 metadata 生成与主 stream 上的其它工作重叠；MLA 同流是因为 metadata 是 kernel 的直接输入且需要保证“先生成后消费”的强顺序。

---

## 7. 为什么 MLA 不照搬 FA

1. **kernel 侧不重算长度。** FA kernel 拿启动参数里的 `cu_seqlens` 自己算偏移；MLA kernel 直接读 metadata 中的 per-batch span 与组前缀，因此记录必须自包含。
2. **吸收形态的切块依赖运行时。** `mBaseSize` 会随 `batch` 与核数关系变化（`batch % aicNum`、等长 KV 均分），无法在 host 静态确定，必须在设备端“先调度再定布局”。
3. **变长 + used 长度带来口径风险。** 有 allocation/used 两层语义后，host 与 kernel 一旦理解不一致就是静默错误，所以需要 `status` 字与消费侧比对。
4. **可选张量多。** `usedQuery/usedKv/cuQuery/cuKv/cache` 都可能缺省，必须有一套“该有则有、该缺则缺”的判断，并把失败表达为“记录无效”而不是崩溃。

---

## 8. 若合并到 `fa_meta` 的注意事项

- **必须全量重编 AICPU 编译单元**，否则 stale 对象会导致结构体 ABI 不兼容并静默踩内存。
- **禁止修改 `tilingdata.h` 中 `FAInferTilingData` 的固定布局**，普通 FA3/FA4 消费路径依赖它。
- **两个管线的输出尺寸契约不同**：FA 是编译期常量，MLA 是 `words` 运行期值；合并时不能共用同一个“固定大小”假设。
- **`status` 字不可省**：FA 侧没有等价机制，合并后若丢状态位，MLA 的软失败将无法被识别。
- **stream 语义不同**：FA 用 event 跨流，MLA 依赖同流顺序；合并入口时需保留各自的执行语义，不能简单统一。

---

## 9. 附录：一句话对照

| 关注点 | FA | MLA |
|---|---|---|
| metadata 是什么 | tiling 参数包 | 自包含调度快照 |
| 大小 | 2376 B 固定 | `words×4` 变长 |
| 读几个数组 | 2 | 4（+ page table） |
| 谁保证正确 | host | host + 设备 + 消费端三层 |
| 失败如何表达 | 不表达 | `false` / `status=1` / `checked` |
| FD 决策 | 启发式阈值 | 调度算法副产物 |
| 数据读取方式 | 结构体强转 | word 偏移直读 |
