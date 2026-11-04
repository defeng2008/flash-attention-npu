# MLA metadata Host 侧调用链路

本文梳理 MLA metadata 在 host 侧的两条调用路径：

1. **前向主链**：`flash_attn_npu_3.fwd` → `fa_fwd` →（命中 MLA 形状签名）`mla_fwd` → `mla::Forward`，生成 metadata 后同流接着启动 AI Core kernel。
2. **离线 metadata 链**：`flash_attn_npu_3.get_scheduler_metadata` → `mla::SchedulerMetadata`，只生成 metadata 张量并返回给 Python。

相关代码位置：

- [flash_api.cpp](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)
- [mla_forward.cpp](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)
- [mla_metadata.cpp](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.cpp)
- [fa_metadata.aicpu](../../csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu)（FA/MLA 共用的唯一 AICPU 编译单元）
- [mla_dispatch.h](../../csrc/ascend910/flash_attn_npu_3/mla/mla_dispatch.h)
- [mla_dispatch_impl.hpp](../../csrc/ascend910/flash_attn_npu_3/mla/mla_dispatch_impl.hpp)
- [flash_attn_npu_interface.py](../../flash_attn_npu_3/flash_attn_npu_interface.py)

---

## 一、前向主链（带 metadata 生成）

```
Python: flash_attn_npu_interface.py
  flash_attn_func / flash_attn_varlen_func / flash_attn_with_kvcache
        │
        ▼  pybind  flash_api.cpp L1460  m.def("fwd", &fa_fwd)
fa_fwd(...)                                   flash_api.cpp L1427
        │   IsMlaForward 形状签名门控           flash_api.cpp L169
        │
        ├─► flash_attn::mla::mla_fwd(...)   flash_api.cpp L1441
        │        │  参数适配 + MLA 形态判定（compressed / prefill）
        │        │  非 MLA 形态返回 nullopt → fa_fwd 调 mha_fwd（L177）回落普通 FA3 路径
        │        ▼
        │   flash_attn::mla::Forward(...)      mla_forward.cpp L209
        │        │
        │        ├─① 输入校验
        │        │    CheckTndInputs / CheckTndBsndForwardInputs / CheckMetadataQueryPrefix  L222-229
        │        │    CheckUsedInputs                                                       L231
        │        │    BuildPagedCacheLayout                                                L237
        │        │
        │        ├─② MakeContext(...) → MlaContext c + MlaLayout l   L247 / 定义在 L147
        │        │    └─ MlaRecordLayout(c, l)   ← 算出 words / bytes / outOffset            L184
        │        │    └─ valueDim==512 时二次 MlaRecordLayout(c, small, 128, l.n) 反算最坏容量  L193-203
        │        │
        │        ├─③ 若用户传了 scheduler_metadata：
        │        │    CheckMetadataBuffer(*supplied, q.device(), MlaUnifiedBytes(l))        L255-256
        ├─④ 分配输出 out / lse / accum / accumLse                                 L258-267
        ├─⑤ 分配 workspace / metadata
        │        │    workspace = at::empty(checkedOffset + sizeof(MlaCheckedAutoSplitSchedule)
        │        │                            + platform.libApiWorkspaceBytes)             L271-273
        │        │    metadata  = supplied ? *supplied : at::empty({MlaUnifiedBytes(l)}, kByte) L274
        ├─⑥ 组装 FAMetadataArgs args（FillMlaArgs：cuQ / cuK / usedQ / usedK / output / 缓存描述） L275-281
        │        ├─⑦ RecordInput(...) 逐张量登记到当前 stream                              L284-290
        │        ├─⑧ 组装 MlaLaunchArgs a（各 GM 指针、ffts、checked）                       L291-322
        │        │    a.workspace = workspace + platform.libApiWorkspaceBytes              L303
        │        │    a.checked   = a.workspace + checkedOffset                            L309
        │        │    args.checked = a.checked；args.generate = !supplied                   L310-311
        │        │
        │        └─⑨ RunMlaLaunch("ascendc_mla", λ)                                        L323
        │             ├─ 普通路径：RECORD_FUNCTION + launch() 检查 ACL_SUCCESS              L47-48
        │             └─ ASCEND_LAUNCH_BLOCKING / TORCH_NPU_SANITIZER：
        │                  OpCommand::RunOpApiV2(name, λ)                                  L40
        │             └─ λ：
        │                  ComputeMlaMetadata<<<1, nullptr, a.stream>>>(&args, sizeof(args));  L325
        │                  LaunchMla<half | bfloat16_t>(a)                                  L326
        │
        ▼  设备侧（同一 stream 顺序执行）
   ┌──────────────────────────────────────────────────────────────────────┐
   │ AICPU  ComputeMlaMetadata            fa_metadata.aicpu             │
   │   BuildNonquantMetadata(a)  ← args.mlaGenerate 为假时跳过生成         │
   │   CheckRecord(a, checked.schedule) → 写 a.mlaChecked                 │
   │   （返回值 0/1 无人消费）                                             │
   └──────────────────────────────────────────────────────────────────────┘
   ┌──────────────────────────────────────────────────────────────────────┐
   │ AICore LaunchMla<DType>(a)           mla_dispatch.h L52-65           │
   │   margin || (!packed && heads != kvHeads) || splits > 1              │
   │        → LaunchGrouped    (mla_gqa_dispatch_impl.hpp L238-243)       │
   │   否则 valueDim == 512 → LaunchCompressed (mla_auto_split_…  L124+)   │
   │        否则            → LaunchPrefill    (mla_prefill_…      L227+)  │
   │   三个入口内部都先 LoadMla(a, …)      (mla_dispatch_impl.hpp L14-37)  │
   │     └─ LoadCheckedAutoSplitSchedule(a.checked, …) 作为逐核门禁        │
   │     └─ read.data.SetGlobalBuffer(a.metadata)                        │
   │   之后 <<<l.cores, blockDim, stream>>> 启动具体 kernel                │
   └──────────────────────────────────────────────────────────────────────┘
```

---

## 二、离线 metadata 链（get_scheduler_metadata）

```
Python: flash_attn_npu_interface.py
  _get_scheduler_metadata_op  L1842  →  flash_attn_npu_3.get_scheduler_metadata(...)  L1867
        │
        ▼  pybind  flash_api.cpp L1462
get_scheduler_metadata(...)                   flash_api.cpp L812
        │
        └─► flash_attn::mla::SchedulerMetadata(...)   flash_api.cpp L825 / 定义 L333
                │
                ├─① 入参校验：shape 上下界、dtype∈{FP16, BF16}、page_size、
                │    CheckCacheLengths、dim/valueDim 必须是 64/512 或 192/128      L339-350
                ├─② MakeContext(batch, sq, sk, ..., valueDim, layout,
                │    totalQ=0, totalK=0, page, hasSoftcap, ...)                   L353-355
                │    layout = page ? (cuQ ? 4 : 3) : (cuQ ? 2 : 0)
                ├─③ metadata = at::empty({MlaUnifiedBytes(l)}, lengths.options().dtype(kByte))  L356
                ├─④ FAMetadataArgs args；FillMlaArgs(..., inferTotalQ = uint32_t(bool(cuQ)),
                │    generate = true, cache = {})                                    L359-366
                ├─⑤ RecordInput(lengths / cuQ, stream)                            L368-370
                └─⑥ OpCommand::RunOpApiV2("ascendc_mla_metadata", λ)               L371
                     └─ ComputeMlaMetadata<<<1, nullptr, stream>>>(&args, sizeof(args));  L372
                │
                ▼ 返回 metadata 张量（无 checked、无 kernel 启动）
        Python 侧把该张量原样回传（不要 clone / copy）
                │
                ▼
        前向链 ⑦/⑧/⑨ 中 supplied != nullptr
                → CheckMetadataBuffer 校验 MlaUnifiedBytes(l) 一致     L256
                → metadata 直接复用 supplied                           L274
                → args.mlaGenerate = false                             L311
                → AICPU 只跑 CheckRecord，不再 BuildNonquantMetadata
```

---

## 三、两条链的差异要点

| 维度 | 前向链 `Forward` | 离线链 `SchedulerMetadata` |
|---|---|---|
| 入口 | `fa_fwd`（`IsMlaForward` 门控）→ `mla_fwd` | `get_scheduler_metadata` |
| 启动方式 | `RunMlaLaunch`（直调 ACL，带 blocking 兜底） | 恒 `OpCommand::RunOpApiV2` |
| `generate` | `!supplied`（有则跳过生成） | 恒 true（走默认值） |
| `inferTotalQ` | 0（未设置） | `bool(cuQ) \| (cuKv ? 2 : 0)`（bit0 由 `cuQuery` 推断 totalQ、bit1 由 `cuKv` 推断 totalK） |
| `cuKv / usedQuery / usedKv` | 真实地址 | `cuKv`/`usedQ`/`usedK` 传真实地址；`usedK` 为空且 `cuKv` 也为空时回落 `cache_seqlens` |
| `checked` | 有（workspace 内偏移，写 `mlaChecked`） | 无（`mlaChecked` 为空 → 跳过自校验） |
| 产出 | `out / lse / accum / accumLse` + metadata | 仅 metadata 张量 |
| 后续 | 同流继续 `LaunchMla` | 结束，等待用户回传 |

两条链共用：`MakeContext` → `MlaRecordLayout` → `FillMlaArgs`（产出 `FAMetadataArgs`）→ `ComputeMlaMetadata`（AICPU）→ `BuildNonquantMetadata`，区别只在「是否生成」与「是否接着跑 kernel」。

---

## 四、Python 入口是否走离线链

`flash_attn_func`（`FlashAttnFunc.forward`）在 host 侧内部生成一份记录并随前向下发，MLA 与非
MLA 一致（`flash_attn_npu_interface.py:788-818`），此时前向链第 ③/⑦/⑧ 步走 `supplied` 分支，
AICPU 只做 `CheckRecord` 自校验。`_get_scheduler_metadata_op` 对该记录同样挂
`_fa_scheduler_params` fingerprint（`:1792-1803`），但 MLA 记录的校验在
`flash_attn_with_kvcache` 中被显式关闭。

以下入口的 MLA 分支也走离线链，记录由 host 生成、AICPU 只做 `CheckRecord` 自校验：

- v3 `flash_attn_varlen_func`：MLA 传 `headdim_v=v.shape[-1]`、`cu_seqlens_k`、`seqlens_q=cu_seqlens_q`
  与 `seqused_q/seqused_k`，离线链据此表达布局 1（TND 打包 KV）或布局 2（TND Q + BSND KV）
  （`flash_attn_npu_interface.py:996-1023`）。
- v4 `flash_attn_func` / `flash_attn_varlen_func`：v4 的 `get_scheduler_metadata` 已扩展
  `qkv_dtype`/`cu_seqlens_k`/`seqused_k`，非分页 MLA 记录由 host 生成
  （`flash_attn_npu_4/flash_api.cpp:782-790`）。

唯一例外：v4 分页 MLA KV（布局 3/4）仍由 AICPU 现场生成 —— v4 离线链没有 page size 入参，
无法绑定页对齐容量，varlen 分支以 `is_mla and page_table is not None` 为守卫跳过 host 生成。
