# MLA 前向支持设计文档（flash_attn_npu_3 / flash_attn_npu_4）

> 任务 5.1。本文档描述 MLA（Multi-head Latent Attention）前向在 `flash_attn_npu_3`（FA3）
> 与 `flash_attn_npu_4`（FA4）中的实现设计。
> 所有结论均来自工作区当前源码，引用形式为 `path:line`。无法从源码确认处标注「待确认」。

---

## 1. 背景与目标

MLA 是 DeepSeek 系列使用的注意力变体：query/key 的 RoPE 部分与 latent（compressed）部分
维度不同。标准 FA3/FA4 前向要求 `q` 与 `v` 最后一维相等，无法直接承载 MLA。

本次新增 MLA 前向支持，目标：

- 复用现有 `mha_fwd` / `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache`
  公共入口，在 host 侧做**形状签名自动判定**，命中 MLA 时早退到独立实现，不接管普通 FA3/FA4；
- 支持两种 MLA 形态：**prefill（未吸收）** 与 **decode（compressed / 吸收）**；
- 复用 CANN ops-transformer MLA（FIA MLA）的分核/调度口径，使结果与参考基线一致；
- 仅前向、仅 FP16/BF16、不做量化。

## 2. 范围与不支持项

| 项 | 状态 | 依据 |
| --- | --- | --- |
| 前向计算 | 支持 | `mla_forward.cpp::mla_fwd` / `Forward` |
| 反向（backward） | **不支持** | MLA 仅实现前向，未提供反向；前向不显式拒绝 `requires_grad` 输入 |
| 量化（INT8 / descale） | **不支持** | `mla_validation.h:104-106`；`mla_forward.cpp:413-416`（descale 必须为空） |
| dtype | 仅 FP16 / BF16 | `mla_validation.h:107-108`；`MlaMetadataDtype`（`mla_metadata.h:19-22`） |
| 设备 | 仅 Ascend 910B / 910C（含 910_93） | `mla_forward.cpp:56` |
| layout | 仅 TND(0) / BSND(1) | 见 §3、§8 |
| append-KV / rotary / leftpad | 不支持 | `mla_forward.cpp:405-409` |
| `gather_kv_indices` | 不支持（v4 参数存在但必须为空） | `flash_api.cpp:1415,258`（v4）；`flash_attn_npu_interface.py:1290,1426` |

> 说明：本次改动**已剔除**参考实现 pr-338 中的 INT8 相关资产，因此量化路径完全不存在。

## 3. 形态判定规则（形状签名 → prefill / decode）

判定分两层：**Python 接口层**决定是否走 MLA 与使用哪种 layout；**C++ host 层** `mla_fwd`
再做一次 MLA 边界判定与形态校验。

### 3.1 Python 层 `_is_mla`

`flash_attn_npu_3/flash_attn_npu_interface.py:39-54`（v4 对应位置
`flash_attn_npu_4/flash_attn_npu_interface.py:41`，本轮未改动）：

```python
def _is_mla(q, v, qv, cu_seqlens_q=None):
    if qv is not None:
        return True
    if q.ndim == 0 or v.ndim == 0:
        return False
    return (cu_seqlens_q is None or q.ndim == 3) and q.shape[-1] == 192 and v.shape[-1] == 128
```

即：**显式传 `qv` 一律判定为 MLA**，形状不匹配的 `qv` 交给 native MLA 校验报错，避免被普通
FA3 kernel 静默忽略；**不传 `qv`** 时只认 prefill 签名 `Dq==192 && Dv==128`（对应 §3.2 的
`IsMlaForward`），其余头维组合交回普通 FA3 路径。`(cu_seqlens_q is None or q.ndim == 3)`
保留 `CheckForwardBoundary` 的「普通布局 / 畸形 query 交回上游校验」语义。

> v4 的 Python 层仍是旧口径（`qv is not None or q.shape[-1] != v.shape[-1]`）：`qv` 分支等价，
> 不带 `qv` 时更宽松（任何 `Dq != Dv` 都判 MLA）。

### 3.2 C++ host 层判定与形态签名

入口 `mla_fwd`（`csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp:379`，v4 同源）
先调用 `CheckForwardBoundary`（`mla_validation.h:91`）判定是否 MLA；非 MLA 直接
`return std::nullopt`（`mla_forward.cpp:398-399`），公共入口随后执行原有 FA3/FA4 逻辑。

v3/v4 的 Python 入口 `.fwd` 都绑定到 **`fa_fwd`**（v3 `flash_attn_npu_3/flash_api.cpp:1427-1455`、
v4 `flash_attn_npu_4/flash_api.cpp:1404-1431`），它在调用 `mla_fwd` 前先按同一套形状签名做
**入口门控**，封装为 `IsMlaForward(q, v, qv)`（v3 `:166-175` 调用点 `:1440`；v4 `:147-153`
调用点 `:1416`）：传 `qv` 时要求 `Dq==64 && Dv==512`，不传 `qv` 时要求 `Dq!=Dv &&
Dq==192 && Dv==128`；未命中的输入直接交给 `mha_fwd`（v3 `:1450`、v4 `:1427`），保持普通
FA3/FA4 路径不变。门控仅判定形状签名，layout/dtype/KV 头数等约束仍由 `mla_fwd` 内部的
`CheckForwardBoundary` 抛出明确错误。两侧 `mha_fwd` 本身都已不含 MLA 逻辑。

`CheckForwardBoundary` 内部按两个形状签名分支（`mla_validation.h:125-158`）：

| 形态 | 触发条件 | 形状签名 | 约束 |
| --- | --- | --- | --- |
| **decode（compressed / 吸收）** | 传 `qv` | `qv->size(-1) == v.size(-1)`，`q.size(-1)==64`，`v.size(-1)==512` | KV 头数**恒为 1**（`mla_validation.h:134`）；`q/qv` 形状一致（除最后一维） |
| **prefill（未吸收）** | 不传 `qv` 且 `q.size(-1) != v.size(-1)` | `q.size(-1)==192`，`v.size(-1)==128` | KV 头数可 >1（GQA），也允许 `== heads_q`（`mla_validation.h:123`） |

`q/k` 头维必须相等（`mla_validation.h:101`），`heads_q % heads_kv == 0`（`mla_validation.h:123`）。
`SchedulerMetadata` 对形态做同样的双签名校验（`mla_forward.cpp:349-350`）：
`(dim==64 && valueDim==512 && kvHeads==1) || (dim==192 && valueDim==128)`。

### 3.3 layout 判定

- **v3 `flash_attn_func`（`FlashAttnFunc.forward`）**：仅当**非** MLA 时才调用
  `_detect_layout`；MLA 分支不改变 `layout`，其初值为 `_LAYOUT_BSND`，因此 v3 训练入口下
  MLA 固定走 BSND（`flash_attn_npu_interface.py:862-873`）。
- **v3 `flash_attn_varlen_func`**：`cu_seqlens_q` 存在 → 固定 TND（`:1026-1027`）。
- **v3 `flash_attn_with_kvcache`**：`cu_seqlens_q` 存在 → TND；否则 MLA → BSND（`:1604-1607`）。
- **C++ host**：`mla_fwd` 仅接受 `layout==0 || layout==1`，否则报
  `"MLA: layout must be TND or BSND"`（`mla_forward.cpp:400`）。
- **v4**：interface 层的 MLA 分支同样**不**调用 `_detect_layout`，`layout` 固定为
  `_LAYOUT_BSND`（`flash_attn_npu_4/flash_attn_npu_interface.py:914-921`、`:1111-1118`），
  与 v3 一致。
- **BNSD 内存上的 BSND 视图**：调用方把连续 BNSD `(B,H,S,D)` 经 `.transpose(1,2)` 得到的
  BSND 形状视图，v3/v4 的 MLA 路径均按 BSND 接纳（strides 透传、零拷贝），
  `out.shape == q.shape`。详见
  [mla_shape_spec.md §3](./mla_shape_spec.md#bnsd-内存上的-bsnd-视图transpose1-2)。
  TND / 分页 KV 不适用。

`MlaContext.layout` 的完整编码（`mla_metadata.h:43`）：`BSND=0, TND Q/KV=1, mixed=2, BSND PA=3, TND PA=4`。

### 3.4 host 侧 `scheduler_metadata` 的生成与传入

训练入口（`flash_attn_func`）不对外暴露 `scheduler_metadata`，但会在 host 侧生成一份记录、
随前向一同下发，使设备端 AICPU 只做自校验（`args.generate=false`）而不必现场构建：

- **v3 `flash_attn_func`（`FlashAttnFunc.forward`）**：MLA 与非 MLA 都生成。以 `q.dim() == 4`
  为前置条件（rank≠4 的非法输入不生成，交 native 抛 `RuntimeError`）；MLA 分支不调用
  `_detect_layout`，`layout` 保持 `_LAYOUT_BSND`；`headdim_v` 取 `v.shape[-1]`
  （非 MLA 时与 `head_size` 相等，MLA 时即 Dv），`qkv_dtype=q.dtype`
  （`flash_attn_npu_interface.py:788-818`）。
- **v3 `_get_scheduler_metadata_op`**：不再对 MLA 提前返回，MLA 记录同样挂上
  `_fa_scheduler_params` fingerprint（`:1792-1803`）。该 fingerprint 只在
  `flash_attn_with_kvcache` 中经 `_validate_scheduler_metadata` 校验，且该校验对 MLA 关闭
  （见 §8.3），因此不影响 MLA 记录 `clone()` / 存储切片后的形态。
- **v3 `flash_attn_varlen_func`（`FlashAttnVarlenFunc.forward`）**：MLA **不**生成 host 侧记录
  （`:959-960`）。原因是 TND 打包 KV（`cu_seqlens_k`）下 forward 走布局 1（TND 稠密），而
  `get_scheduler_metadata` 只能表达布局 2（TND Q + BSND KV），host 记录与 forward 选择的
  布局不一致；由 forward 内部生成才与自身布局一致。
- **v4**：`flash_attn_func` / `flash_attn_varlen_func` 的 MLA 分支均**不**生成 host 侧记录。
  v4 的 `get_scheduler_metadata` 签名（FA4 风格）没有 `qkv_dtype`，其 MLA 路由固定按
  `at::kBFloat16` 落记录（`flash_attn_npu_4/flash_api.cpp:779-786`）；对 FP16 调用方该记录的
  dtype 与实际张量不符，会导致设备侧异常。故 v4 保持 MLA 记录仅由 AICPU 内部生成。

## 4. 数据流与模块划分

```
Python: flash_attn_func / flash_attn_varlen_func / flash_attn_with_kvcache
   │  _is_mla 分流；MLA 分支计算 softmax_scale、layout，调用底层 op
   ▼
C++ host: fa_fwd(MLA 签名) ──► flash_attn::mla::mla_fwd(...)   (早退钩子)
   │                          │  CheckForwardBoundary 命中?
   │                          ▼  是
   │                    flash_attn::mla::Forward(...)
   │                          │  MakeContext → MlaRecordLayout（shape/workspace 校验）
   │                          │  Check* 输入/长度/layout 校验（mla_validation.h）
   │                          │  RunMlaLaunch:
   │                          │    1) ComputeMlaMetadata<<<1>>> (AICPU) 生成记录
   │                          │    2) LaunchMla<dtype>() → 数值 kernel
   ▼  否（普通 FA3/FA4）       ▼
 原有 mha_fwd 实现        {out, lse[, accum, accumLse]}
```

模块职责：

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| host 入口 / 早退钩子 | `mla/mla_forward.cpp`、`mla/mla_forward.h` | `mla_fwd`（判定+参数适配）、`Forward`（分配/launch）、`SchedulerMetadata`（独立 metadata 入口） |
| 输入校验 | `mla/mla_validation.h` | 形状签名、dtype/device/stride、量化、TND/BSND/paged 组合校验 |
| layout 描述 | `mla/mla_layout.cpp/.h` | BSND/TND tensor view 的 stride/storage 与 paged cache layout 解析 |
| Tiling / 分核 | `mla/mla_tiling.cpp/.h`、`mla/mla_split.cpp/.h`、`mla/cann_split_core.cpp/.h` | 形状→块大小/核数/端点；workspace 预算；GQA/pack 资源解析 |
| metadata 生产 | `mla/mla_metadata.cpp/.h`、`mla/mla_metadata.aicpu` | 记录格式定义（`MlaRecordHeader`/`MlaNativePartition`）、AICPU `ComputeMlaMetadata` |
| metadata 消费 | `mla/mla_metadata_consumer.h` | device 侧读取记录 |
| 调度模式注入 | `mla/mla_dispatch.cpp/.h`、`setup.py` | `--wrap` 强制 `ScheduleMode::BATCH_MODE` |
| dispatch（数值 kernel 选择） | `mla/mla_dispatch.h` + `autogen/generate_kernels.py` 生成的 `mla_dispatch_{dtype}_{family}.cpp` | `LaunchMla` → `LaunchCompressed`/`LaunchPrefill`/`LaunchGrouped` |
| kernel 实现 | `mla/mla_kernel.hpp`、`mla/mla_prefill_kernel.hpp`、`mla/mla_auto_split_dispatch_impl.hpp`、`mla/mla_prefill_dispatch_impl.hpp`、`mla/mla_gqa_dispatch_impl.hpp`、`mla/*_block.hpp`、`mla/*_tile.hpp` | `__mix__(1,2)` 核函数（cube+vector） |

**AICPU metadata 通道**：`Forward` 在同一次 host 提交中先 `ComputeMlaMetadata<<<1, nullptr, stream>>>`
（`mla_forward.cpp:325`），再 launch 数值 kernel（`:326`）。记录写入调用方传入的
`scheduler_metadata`（`supplied` 非空）或内部 workspace（`mla_forward.cpp:274`）。
`SchedulerMetadata` 独立入口同样通过 `ComputeMlaMetadata`（`:371-373`）。

## 5. Tiling 与分核策略

### 5.1 三级坐标系（BN2 / S1G / S2）

分核结果用三个拍平的端点表达每一段任务的归属（`mla_tiling.h:104-108`）：

- `bN2End`：`batch * n2`（latent KV 头，MLA 下 n2=1）方向的端点；
- `gS1End`：S1G（query 行按 `mBaseSize` 分块后）方向的端点；
- `s2End`：S2（KV 按 `s2BaseSize` 分块后）方向的端点。

`MlaSchedule`（`mla_tiling.h:88-113`）还包括 `mBaseSize`、`s2BaseSize`、`usedCoreNum`、
`blockDim`、`scheduleMode`（固定 1 = `BATCH_MODE`）。

### 5.2 块大小

- **dense 调度（`mla_tiling.cpp`）**：`mBaseSize` 默认 256；当 `batch % aicNum != 0` 且各
  batch KV 长度一致时，可按 `rows / splitOfS1G`（需落在 [128,256]）缩小
  （`mla_tiling.cpp:31-48`）。`s2BaseSize = min(s2Size, 512)`（`mla_tiling.cpp:239`）。
  KV 最大长度受 `kMlaMaxKvLength = 16384` 约束（`mla_tiling.h:27`）。
- **native record（`mla_metadata.h::MlaRecordLayout`）**：compressed（`valueDim==512`）时
  `m=256`、`n=min(sk,512)`；prefill（`valueDim==128`）时先算 `prefillN`（短序列 S≤16 时按
  GQA 组数取 8192/4096/2048，否则 512；受 window 限 1024、受 page_size 对齐、上限 `sk`），
  再由 `aligned=ceil32(prefillN)` 映射 `prefillM`（512→512 / 1024→256 / 2048→128 /
  4096→64 / 其它→32），`r.n = prefillN`（`mla_metadata.h:140-163`）。

### 5.3 分核（贪心 + 容差 + 试多核）

`mla_tiling.cpp` 通过 `SplitWithoutFlashDecode`（`:102`）复用 CANN `detail::SplitCore`，并在
FA3 `num_splits=1` 语义下把每个端点向上取整到完整 M 块、合并空区间（`:109-159`）。

`detail::SplitCore`（`mla/cann_split_core.cpp:668`）的核心：

```cpp
uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
uint32_t minCore = (uint32_t)(std::sqrt((float)totalBlockNum + 0.25f) + 0.5f);   // ≈ round(sqrt(总块数))
minCore = std::min(minCore, maxCore);
result.maxCost = INT64_MAX;
for (uint32_t i = minCore; i <= maxCore; ++i) {        // 试多核：从 sqrt 到可用核数
    CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
    if (tmpResult.maxCost < result.maxCost) CopyTmpResult(tmpResult, result);  // 取最慢核成本最小
}
```

- **试多核**：候选核数从 `≈sqrt(总块数)` 遍历到 `min(可用核数, 总块数)`，逐个调用 `CalcSplitPlan`
  并保留 `maxCost`（最慢核成本）最小者（`cann_split_core.cpp:680-695`）。
- **贪心**：内核数固定后按 batch/行/块三级贪心填充——`AssignByBatch`（`:395`）、`AssignByRow`（`:434`）、
  `AssignByBlock`（`:462`）、`ForceAssign`（`:488`，至少分配 1 块）。
- **容差**：`FA_TOLERANCE_RATIO = 2`（`cann_split_core.h:25`），`IsWithinTolerance(limit,tol,value)`
  判据为 `limit + tolerance >= value`（`:51-54`）；每个贪心循环用「当前块成本 / 2」作为容差阈值。

核数上限 `kMlaMaxAic = 26`（`mla_tiling.h:18`）。`usedCoreNum` 与 `blockDim` 取自平台
（`mla_tiling.cpp:286-287`，须 `blockDim >= usedCoreNum`，`:289`）。

### 5.4 Workspace

`MakeContext`（`mla_forward.cpp:147`）调用 `MlaRecordLayout` 计算记录字数与 workspace 字节，
并额外为 `valueDim==512` 预留 256 行（动态 M 上界 128..256）的累加空间
（`mla_forward.cpp:193-203`）。实际分配带 64B 对齐与 libApi workspace 追加（`:270-273`）。

## 6. 派发与 kernel 选择

### 6.1 host 侧 `LaunchMla`

`mla_dispatch.h:59-65`：

```cpp
template <typename DType>
aclError LaunchMla(const MlaLaunchArgs& a) {
    const auto& c = a.context;
    if (c.margin || (!c.packed && c.heads != c.kvHeads) || c.splits > 1)
        return LaunchGrouped<DType>(a);
    return c.valueDim == 512 ? LaunchCompressed<DType>(a) : LaunchPrefill<DType>(a);
}
```

即：**GQA（heads≠kvHeads）/ sm_margin / 手写 splits>1** 走 `LaunchGrouped`（`GqaAttention`，
`mla_gqa_dispatch_impl.hpp:208,238`）；否则 compressed（`valueDim==512`）走 `LaunchCompressed`
（`CompressedAttention`，`mla_auto_split_dispatch_impl.hpp:98,124`）；prefill（`valueDim==128`）
走 `LaunchPrefill`（`PrefillAttention`，`mla_prefill_dispatch_impl.hpp:184,227`）。

kernel 模板进一步按编译期常量特化：

- `LaunchPrefill`：`Brc`（`l.m<=128`）、`Partial`（`splits!=1`）、`Paged`（`pageSize!=0`）、
  `IsTnd`（`mla_prefill_dispatch_impl.hpp:233-237`）。
- `LaunchCompressed`：`IsTnd` / `Paged`，核数取 `physicalCores - margin`
  （`mla_auto_split_dispatch_impl.hpp:124-137`）。
- kernel 属性均为 `__mix__(1, 2)`（cube:vector = 1:2）。

### 6.2 dispatch TU 生成

`autogen/generate_kernels.py` 为每个 (dtype, family) 生成一份显式模板实例化 TU
（`csrc/ascend910/flash_attn_npu_3/autogen/generate_kernels.py:111-128`）：

```python
MLA_FAMILIES = {
    "auto_split": ("mla_auto_split_dispatch_impl.hpp", [("LaunchCompressed", "MlaLaunchArgs", "")]),
    "prefill":    ("mla_prefill_dispatch_impl.hpp",    [("LaunchPrefill",    "MlaLaunchArgs", "")]),
    "gqa":        ("mla_gqa_dispatch_impl.hpp",        [("LaunchGrouped",    "MlaLaunchArgs", "")]),
}
```

生成文件：`mla_dispatch_{fp16,bf16}_{auto_split,prefill,gqa}.cpp`（每版本 6 个，v4 同源，
`..._4/autogen/generate_kernels.py:100-117`）。dtype 组合仅 FP16/BF16。

### 6.3 `ScheduleMode::BATCH_MODE` 注入

`mla_dispatch.cpp` 用 `--wrap=aclrtLaunchKernelWithHostArgs` /
`aclrtLaunchKernelWithArgsArray` 在**线程局部**作用域内改写 launch 属性，追加
`ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE = 1`（`mla_dispatch.cpp:24-40,43-83`）；
链路参数在 `setup.py:327-328`。作用域由 `MlaBatchLaunchScope` 管理，普通 FA3/FA4 launch
不受影响（`mla_dispatch.cpp:47-50`）。

## 7. v3 与 v4 差异对照表

| 维度 | v3（`flash_attn_npu_3`） | v4（`flash_attn_npu_4`） |
| --- | --- | --- |
| `mha_fwd` 参数 | 含 `k_new/v_new/cu_seqlens_k_new/kv_batch_idx/leftpad_k/rotary_cos/rotary_sin/seqlens_rotary/q_descale/k_descale/v_descale`；`softmax_scale_` 为 `optional<double>`（`flash_api.cpp:177-209`） | 无 append-KV/rotary/descale/leftpad；新增 `gather_kv_indices_`、`learnable_sink_`；`softmax_scale_` 为 `optional<float>`（`flash_api.cpp:155-175`） |
| 前向入口 | 独立入口 **`fa_fwd`**（`flash_api.cpp:1427-1455`）：先 `IsMlaForward` 门控（`:166-175`，调用点 `:1440`），命中则调 `mla_fwd`（`:1441`），否则回落 `mha_fwd`（`:1450`）；`fwd` 绑定到 `fa_fwd`（`:1460`），`mha_fwd`（`:177`）已无 MLA 逻辑 | 同构：独立入口 **`fa_fwd`**（`flash_api.cpp:1404-1431`）：先 `IsMlaForward` 门控（`:147-153`，调用点 `:1416`）并拒绝 `gather_kv_indices`（`:1415`），命中则调 `mla_fwd`（`:1417`），否则回落 `mha_fwd`（`:1427`）；`mha_fwd` 已无 MLA 逻辑 |
| 返回值 | 4 张量 `{out, lse, out_accum, lse_accum}`；未定义槽补空（`mla_forward.cpp:424-428`） | **2 张量** `{out, softmax_lse}`；`result.resize(2)`（v4 `mla_forward.cpp:424-428`） |
| metadata 路由 | `get_scheduler_metadata`：`headdim != headdim_v` → `flash_attn::mla::SchedulerMetadata`（`flash_api.cpp:824-829`） | 同口径（`flash_api.cpp:779-786`） |
| 公共入口 | `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` | `flash_attn_func` / `flash_attn_varlen_func`（**无 `flash_attn_with_kvcache`**）；decode 分页走 varlen 的 `page_table`/`seqused_k` |
| BSND / BNSD | `FlashAttnFunc.forward` 对 MLA 固定 BSND；native 仅接受 layout 0/1，否则 `"MLA: layout must be TND or BSND"` | 同样对 MLA 固定 BSND；native 契约与 v3 相同。两侧都接纳「BNSD 内存上的 BSND 视图」（见 §3.3） |
| `gather_kv_indices` | 无该参数 | 参数存在但必须为 `None`（interface `:1290` assert + native `flash_api.cpp:1415,258` TORCH_CHECK） |
| gradient | 无反向实现；不显式拒绝 `requires_grad`，反向结果不受支持 | 同 v3 |

> 数值 kernel 源文件 v3/v4 逐字节同源（`mla_forward.cpp` 目录 diff 仅 limit 返回值的收尾处理，
> 其余 `mla_dispatch.cpp` / `mla_tiling.cpp` 等 diff 为空），差异集中在 host 契约与 Python 接口。

## 8. 接口契约（MLA 下）

### 8.1 `flash_attn_func`

- `q/k/v` BSND：`q (B,S,Hq,Dq)`、`k (B,Sk,Hkv,Dk)`、`v (B,Sk,Hkv,Dv)`；
  decode 需额外 `qv (B,S,Hq,512)`。
- MLA prefill：`Dq=Dk=192`、`Dv=128`、`qv=None`；KV 头数可 >1（GQA）。
- MLA decode：`q=64`、`qv=512`、`k=64`、`v=512`、KV 头数恒为 1。
- 输出最后一维为 `v.shape[-1]`（`flash_attn_npu_interface.py:1316`）。
- `softmax_scale=None` 时按 `(q.size(-1) + (qv.size(-1) if qv else 0)) ** -0.5`（`:860-861`）。
- host 侧内部生成 `scheduler_metadata` 并随前向下发（MLA 同，见 §3.4），但不对外暴露该参数。

### 8.2 `flash_attn_varlen_func`

- 需要 `cu_seqlens_q/cu_seqlens_k`（int32，`[B+1]`）与 `max_seqlen_q/k`（`mla_forward.cpp:417-418`）。
- 固定 TND layout（`:1027`）。
- MLA 下 metadata 每次内部生成；无显式 metadata / page_table 参数（文档字符串 `:1378-1380`）。
- 支持 `seqused_q/seqused_k`（used 前缀限制）。

### 8.3 `flash_attn_with_kvcache`（仅 v3）

- cache 只读 BSND 或 ND paged；MLA 下 append（`k/v`）与 rotary 不支持（文档字符串 `:1548-1550`）。
- 本入口按 BSND 形状定义 `q`/`qv`（打包 TND query 走 `flash_attn_varlen_func`）；
  `cache_seqlens` 为 int 时按 `q.shape[0]` 广播为 int32（`:1582-1586`）。
- 分页容量 = `page_table.shape[1] * k_cache.shape[1]`，上限 16384；page_block_size 必须为
  16 的倍数且落在 [16,1024]（设备侧校验：`mla_metadata.h:135`、`kMlaMaxKvLength`）。
- `return_softmax_lse=True` 返回 4 张量 `(out, softmax_lse, out_accum, lse_accum)`。

### 8.4 `qv` 语义

`qv` 是 MLA 的「吸收后 value 侧 query」（`(B,S,Hq,512)`）。带 `qv` 时分数 =
`[q_rope | qv] · [k_rope | v]ᵀ`（576 宽），输出取 `v` 的最后一维（512）。
`qv` 与 `q` 形状一致（除最后一维），且与 `q` 同 device/dtype（`mla_validation.h:126-131`）。

### 8.5 layout 约束小结

| 入口 | 允许 layout |
| --- | --- |
| v3 `flash_attn_func` | BSND（含 BNSD 内存上的 BSND 视图） |
| v3 `flash_attn_varlen_func` | TND |
| v3 `flash_attn_with_kvcache` | BSND（含 BNSD 内存上的 BSND 视图；paged 时 3 号编码）；打包 TND query 走 `flash_attn_varlen_func` |
| v4 `flash_attn_func` / `flash_attn_varlen_func` | BSND（含 BNSD 内存上的 BSND 视图）/ TND |
| 所有路径 C++ host | 仅 0(TND) / 1(BSND)，其余报错 |

## 9. 约束与注意事项

1. **不支持量化**：`q/k/v/qv` 全为 FP16/BF16；任一 `q_descale/k_descale/v_descale` 出现即拒绝
   （`mla_validation.h:104-106`、`mla_forward.cpp:413-416`）。
2. **不支持反向**：MLA 仅实现前向，未提供反向；前向不显式拒绝 `requires_grad`
   输入，反向结果不受支持。
3. **`softmax_scale` 口径（重要）**：
   - 调用方**未显式传** `softmax_scale` 时，MLA 路径按
     `(q.size(-1) + qv.size(-1)) ** -0.5` 自动推导（接口层
     `flash_attn_npu_interface.py:860-861,1028-1029,1574-1575`；native 兜底
     `mla_forward.cpp:419`：`softmax_scale_.value_or(1.0/sqrt(q_v_ ? 576.0 : 192.0))`）。
   - 因此 **compressed 形态缺省得 `576^-0.5`（≈0.041667）**、**prefill 形态缺省得
     `192^-0.5`（≈0.072169）**，两者相差 `sqrt(3) ≈ 1.732` 倍。
   - MLA 的**训推一致性要求调用方显式传入统一的 softmax_scale**：DeepSeek 口径为
     `192^-0.5`（即 128+64，`tests/test_flash_attn_npu_mla_consistency.py:37`）；用户场景
     实际传入的是 `0.14467962580268923`。
   - 实测：缺省 `576^-0.5` 与显式 `192^-0.5` 的相对 L2 差异可达 **0.32**（该数值由规格给出，
     非本仓库测试直接断言）。**这是沿用参考实现 pr-338 的行为，本次未做更改。**
4. **576 需拆 512+64**：compressed/decode 的分数维度是 `qv(512)+q_rope(64)=576`、
   `v(512)+k_rope(64)=576`，实现按该拆分处理（`tests/test_flash_attn_npu_mla_consistency.py:118-125`）。
5. **BNSD**：MLA 不设 BNSD 分支。v3/v4 的 MLA 入口都固定 `layout=_LAYOUT_BSND`，不调用
   `_detect_layout`；native 走到 layout 校验时只接受 0/1。调用方持有的 BNSD 数据可
   `.transpose(1,2)` 成 BSND 视图后直接传入（strides 透传、`out.shape == q.shape`），见 §3.3。
6. **`gather_kv_indices`**：v4 专用参数，MLA 与普通 FA4 均必须为 `None`（`flash_api.cpp:1415,258`）。
7. **page_table 与 cu_seqlens_k 互斥**（`mla_validation.h:103`）。
8. **`attention_chunk` 必须为 0**；`num_splits` 允许 0..256，但 `num_splits != 1` 时压缩路径
   走 GQA/auto-split 分支（`mla_forward.cpp:410-411`、`mla_dispatch.h:62`）。
9. **`num_splits=0`** 表示自动拆分，compressed 路径会走 auto-split + `AutoMerge`
   （`mla_auto_split_dispatch_impl.hpp:119-120`）。

## 10. 关键文件清单

### host / 调度（v3；v4 同路径同源）

| 文件 | 关键位置 |
| --- | --- |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp` | `Forward:209`、`SchedulerMetadata:333`、`mla_fwd:379`、default scale `:419` |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_forward.h` | `mla_fwd:9`、`SchedulerMetadata:25` |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h` | `CheckForwardBoundary:91`、qv 分支 `:125`、prefill 分支 `:145` |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_dispatch.h` | `LaunchMla:59` |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_dispatch.cpp` | `MlaAppendBatchMode:24`、wrap `:43,:66` |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_tiling.cpp` | `CalcMBaseSize:31`、`SplitWithoutFlashDecode:102`、`s2BaseSize:239` |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_tiling.h` | `kMlaMaxAic:18`、`MlaSchedule:88`、三级端点 `:104` |
| `csrc/ascend910/flash_attn_npu_3/mla/cann_split_core.cpp` | `SplitCore:668`、`minCore/maxCore:680-683`、候选遍历 `:689` |
| `csrc/ascend910/flash_attn_npu_3/mla/cann_split_core.h` | `FA_TOLERANCE_RATIO:25`、`IsWithinTolerance:51` |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h` | `MlaContext:44`、`MlaRecordHeader:56`、`MlaNativePartition:63`、`MlaRecordLayout:114` |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.aicpu` | AICPU 编译单元 |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_metadata_consumer.h` | device 读取记录 |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_layout.cpp/.h` | BSND/TND/paged 视图解析 |
| `csrc/ascend910/flash_attn_npu_3/mla/mla_split.cpp/.h` | 累加器布局、GQA 资源、GQA 调度 |

### kernel

| 文件 | 关键位置 |
| --- | --- |
| `.../mla/mla_auto_split_dispatch_impl.hpp` | `CompressedAttention:98`、`LaunchCompressed:124` |
| `.../mla/mla_prefill_dispatch_impl.hpp` | `PrefillAttention:184`、`LaunchPrefill:227` |
| `.../mla/mla_gqa_dispatch_impl.hpp` | `GqaAttention:208`、`LaunchGrouped:238` |
| `.../mla/mla_kernel.hpp` / `mla_prefill_kernel.hpp` | 主体 kernel |
| `.../mla/*_block.hpp` / `*_tile.hpp` | cube/vector 分块实现 |

### 接入点

| 文件 | 关键位置 |
| --- | --- |
| `csrc/ascend910/flash_attn_npu_3/flash_api.cpp` | 前向入口 `fa_fwd:1427`、MLA 规格门控 `IsMlaForward:166`、mla_fwd 调用 `:1441`、回落 `mha_fwd:1450`、`fwd` 绑定 `:1460`、metadata 路由 `:824` |
| `csrc/ascend910/flash_attn_npu_4/flash_api.cpp` | 前向入口 `fa_fwd:1404`、MLA 规格门控 `IsMlaForward:147`、gather 拒绝 `:1415,258`、mla_fwd 调用 `:1417`、回落 `mha_fwd:1427`、`fwd` 绑定 `:1436`、metadata 路由 `:779` |
| `csrc/ascend910/flash_attn_npu_{3,4}/autogen/generate_kernels.py` | `MLA_FAMILIES:111`(v3)/`:100`(v4) |
| `flash_attn_npu_3/flash_attn_npu_interface.py` | `_is_mla:39`、default scale `:861`、BSND `:863` |
| `flash_attn_npu_4/flash_attn_npu_interface.py` | `_is_mla:43`、MLA 固定 BSND `:914-921,:1111-1118` |
| `setup.py` | wrap 链接参数 `:327-328`、MLA host 对象 `:695,:735`、AICPU 输入 glob `:422`、`mla_metadata.aicpu:578` |

## 11. 如何运行

```bash
source /home/w00811046/miniconda3/etc/profile.d/conda.sh && \
conda activate cann && \
source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh && \
export ASCEND_RT_VISIBLE_DEVICES=0
python -m pytest tests/test_flash_attn_npu_v3_mla.py -q
```

（另见测试用例文档 §运行方式。）
