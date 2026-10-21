# Ascend NPU 后端 QKV 支持 BNSD 布局（仅前向）Spec

> 覆盖两代平台、共 5 个后端：
> - Ascend 950：`flash_attn_npu_3`（v3）、`flash_attn_npu_4`（v4），目录 `csrc/ascend950/`
> - Ascend 910：`flash_attn_npu`（v2）、`flash_attn_npu_3`（v3）、`flash_attn_npu_4`（v4），目录 `csrc/ascend910/`
>
> 950 侧先落地（提交 `a514357`，分支 `950bnsd`）；910 侧三个后端随后按同一设计各自独立移植。
> 本文档为两代合并后的一套规格。

## Why

Ascend 950 后端的 `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` 与 Ascend 910 后端
（v2 / v3 / v4）的 `flash_attn_func`（910 v3/v4 另含 `flash_attn_varlen_func`、v2/v3 含 `flash_attn_with_kvcache`）
目前只接受 BSND（`(B, S, H, D)`）与 TND（varlen packed）布局。推理侧 Q 与 KV cache 常以
BNSD（`(B, H, S, D)`，per-head plane）形式存放，调用方需要 `q.transpose(1, 2)`（910 还须再 `.contiguous()`）
转成 BSND 形状再喂入，这不仅在长序列 / 长 cache 下产生明显额外拷贝与显存开销，也让
「输入到底是 BSND 还是 BNSD 转置视图」这一语义在接口上不可表达。需要在两代的 Python 接口层与 CANN
内核层原生支持 BNSD 前向，且**不改变对外 Python 接口参数**。

两代在布局能力上彼此对齐：使用同一套布局码（`0 = TND / 1 = BSND / 2 = BNSD`）、同一套 stride 严格匹配
探测判据、同一套「输出形状等于传入 q 形状」约定。平台差异仅体现在枚举的**名字与取值次序**、kernel 布局
**模板参数个数**、host 侧是否单独定义 `FwdLayout`，以及文档 / 测试文件的**命名**上（详见「支持矩阵」与
「What Changes」）。

实现节奏：950 先落地（提交 `a514357`，分支 `950bnsd`），v3 与 v4 两个 950 后端各自按本设计实现，且为
**独立重写**——不复用、不合入 `bsnd` 分支的任何代码或提交；910 随后把同一设计移植到 v2 / v3 / v4 三个后端
（三份后端目录彼此独立、无共享头）。

## 支持矩阵

后端 × 平台 × 公开入口是否支持 BNSD（`✓` = 支持 BNSD 探测与零拷贝路径；`—` = 该后端无此公开入口；
`✗` = 入口存在但不做 BNSD 解读，仍按既有 BSND / TND 语义）：

| 平台 | 后端 | `flash_attn_func` | `flash_attn_varlen_func` | `flash_attn_with_kvcache` | packed 变体 |
|---|---|---|---|---|---|
| Ascend 950 | v3 | ✓ | ✓ | ✓ | ✗（BSND / TND） |
| Ascend 950 | v4 | ✓ | ✓ | —（v4 无此入口） | ✗（BSND / TND） |
| Ascend 910 | v2 | ✓ | ✗（varlen / packed 按 BSND / TND 解读） | ✓ | ✗（按 BSND / TND 解读） |
| Ascend 910 | v3 | ✓ | ✓ | ✓ | ✗（BSND / TND） |
| Ascend 910 | v4 | ✓ | ✓ | —（v4 无此入口） | ✗（BSND / TND） |

补充：

- 950 侧接口文件为 `flash_attn_npu_3/flash_attn_npu_interface_950.py`、`flash_attn_npu_4/flash_attn_npu_interface_950.py`；
  910 侧三后端均为 `flash_attn_npu{,_3,_4}/flash_attn_npu_interface.py`。
- 950 侧测试为 `tests/test_flash_attn_npu_v3_bnsd.py` / `tests/test_flash_attn_npu_v4_bnsd.py`；
  910 侧测试带 `_910` 后缀以与 950 同名文件区分：`tests/test_flash_attn_npu_v2_bnsd_910.py` /
  `tests/test_flash_attn_npu_v3_bnsd_910.py` / `tests/test_flash_attn_npu_v4_bnsd_910.py`。
- 910 v2 的 `mha_varlen_fwd`（rank-3 / TND）不参与 BNSD；v2 的 packed 变体同样只按 BSND / TND 解读。

## What Changes

### 共性设计（两代一致）

- **单一编译期布局参数承载 Q/K/V 布局语义**：Q、K、V 理论上是同一种输入格式，因此布局只用一个枚举值
  承载，并且它是 kernel 的**编译期模板参数**，由 kernel 内 `if constexpr` 推导 token / head-plane / batch
  步长。业务分支不再依赖任何运行期 `bnsd` 布尔量。（950 形如 `FAInfer<..., Format qFormat, Format kvFormat, ...>`，
  即两个布局形参但喂同一个 `kLayout`；910 的 `FAInfer` 本就只有一个 `INPUT_LAYOUT`，故直接一个 `inputLayout kLayout`。）
- **对外 Python 公开签名不变**：`flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache`
  及 packed 变体均不新增 `bnsd`（或 `q_bnsd` / `kv_bnsd`）参数。布局完全由自动探测决定，探测信号为：

  **stride 严格匹配**：rank-4 且非连续，且
  `stride(3) == 1 && stride(1) == shape[3] && stride(2) == shape[1] * shape[3]`
  ——正是「连续 BNSD 张量经 `.transpose(1, 2)` 得到的 BSND 形状视图」的步长特征。
  要求 Q、K、V **三者全部**命中才判定为 BNSD，然后零拷贝地把它们转回连续 BNSD 交给 kernel
  （`t.transpose(1, 2)` 免费取回原 storage）。

  先做 `is_contiguous()` 短路：连续张量的 stride 是 shape 的纯函数，连续 BNSD 与连续 BSND 在 stride 上
  不可区分；短路同时排除 `S == H == 1` 之类的退化巧合。**不命中 stride 的 rank-4 输入一律按 BSND 解读**：
  shape 无法区分 BSND 与 BNSD，用整除规则去猜会在两种读法都成立时静默选错，故不作为判据。
  混合输入（如 q 命中 stride 而 v 未命中）整体按 BSND 处理，数值仍正确，仅多一次 `contiguous()` 拷贝。

- **输出形状与调用方传入的 q 形状一致**：stride 路径下 q 以 BSND 形状视图传入 →
  `out` 回写为 `out.transpose(1, 2)`（零拷贝视图），`out.shape == q.shape`；
  BSND 路径下 `out` 原样以传入 shape 返回。
  LSE 沿用各入口既有约定（v3 `flash_attn_func` 为 `(B, S, H)`，v3 `flash_attn_with_kvcache` 与 v4 为 `(B, H, S)`），
  不因布局路径改变。
- **不修改** `FAInferTilingData`（v3/v4）与 `FAInferStaticTilingData` / `FAInferRuntimeData`（910 v4）的字段集合，
  也不触碰 `fa_metadata.aicpu`（预编译 AICPU 产物的 ABI 不动）；`setup.py` 用 glob 收集 `autogen/*.cpp`，无需改动。
  布局信息全部走 kernel 入口参数与 host 侧布局码。
- **host 底层入口新增 `int64_t layout`**（0 / 1 / 2，Python 公开层不透出）：host 只做取值范围校验与不变式校验
  `layout == TND ⟺ cu_seqlens_q is not None`，布局探测结果由 Python 侧一次算出后透传，host 不再重复探测。
- kernel 寻址改造：Q/O 的 token 步长与 head-plane 步长、K/V 的 per-head plane 与 batch 步长、
  `gmOffset{Q,K,V,O}`、LSE offset；输出写回与 Nd2Nz 源步长参数化。
- BNSD **仅前向**：各后端 autograd backward 遇 BNSD 抛明确 `NotImplementedError`，不实现 BNSD 反向。

### 950 实现要点

- kernel 侧枚举：`csrc/ascend950/flash_attn_npu_{3,4}/kernel_common.hpp` 的 `Format { TND = 0, BSND = 1, BNSD = 2 }`；
  host 侧另有 `FwdLayout { TND = 0, BSND = 1, BNSD = 2 }`（`csrc/ascend950/flash_attn_npu_{3,4}/tilingdata.h`），
  替换原 `bool is_tnd`，取值与 `Format` 逐项一致，可用 `static_cast<Format>(layout_enum)` 直接得到编译期模板实参。
- kernel 布局模板参数**两个**：`FAInfer<..., Format qFormat, Format kvFormat, ...>`，二者同喂 `kLayout`。
- dispatch 模板参数：`launch_fwd_impl<DType, Format kLayout>`；`FWD_KERNEL_LAUNCH` 把 `kLayout` 同时喂给
  kernel 的 `qFormat` 与 `kvFormat`。
- fwd dispatch/autogen TU：v3 / v4 均 4 → 6（`(fp16, bf16) × (TND, BSND, BNSD)`），在
  `autogen/generate_kernels.py` 的 `LAYOUTS` 增加 `("bnsd", "BNSD", "Format::BNSD")` 行即可。
- `FwdLaunchArgs` 新增 `layout`（`Format`）与 `kv_seqlen_allocated`；`FAIKernelParams` 新增
  `int64_t kvSeqlenAllocated`（其余布局填 0），无任何布局布尔量。
- DN 快路径专项（950 特有）：DN 版 `loadQGM` 的 `srcNdMatrixStride` 由硬编码 `embed` 改为按布局推导的
  `qHeadStride`（BNSD 为 `maxQSeqlen * embed_`），并放开两处 BNSD×DN 排除（`enableDN` 与 `kUseDnFastPath`）。
- v4 后端同样改造其 `flash_attn_func` / `flash_attn_varlen_func`（v4 无 `flash_attn_with_kvcache`）；
  v4 varlen query 仅 rank-3，因此 packed（`cu_seqlens_q` 存在）路径恒为 `FwdLayout::TND`，否则为 `FwdLayout::BSND`。
- C++ 侧 `mha_fwd` 的底层入口参数由 `std::optional<bool> bnsd_` 换成 `int64_t layout`。

### 910 实现要点

- kernel 侧枚举：三份 `csrc/ascend910/flash_attn_npu{,_3,_4}/kernel_common.hpp` 各自独立定义
  `enum class inputLayout : uint32_t { BSND = 0, TND = 1, BNSD = 2 }`（注意：取值次序与 950 的 `Format` / `FwdLayout`
  不同，BSND 与 TND 对调，但 BNSD 同为 2）；**910 没有 host 侧 `FwdLayout` 枚举**，host 持 `int64_t layout`
  （`0=TND` / `1=BSND` / `2=BNSD`），并显式映射到 kernel 的 `inputLayout`（两者编号不同）。
- kernel 布局模板参数**一个**：910 的 `FAInfer` 只有一个 `INPUT_LAYOUT` 形参（不像 950 需要 `qFormat` / `kvFormat` 两个）。
- dispatch 模板参数：由 `bool IS_TND` 改为 `inputLayout kLayout`，即 `launch_fwd<kLayout>` /
  `launch_fwd_dtype<DType, kLayout>` / `launch_fwd_impl<DType, kLayout>`；`FWD_LAUNCH` 宏把 `kLayout` 与
  `kvSeqlenAllocated` 传给 kernel 例化。
- fwd dispatch/autogen TU 数量（三轮各自独立）：v3 fwd 4 → 6、bwd 4 不变；v4 fwd 4 → 6；
  v2 `fwd_dispatch` 4 → 6、`fwd_combo` 32 → 48，而 v2 的 `fag_general_dispatch` / `varlen_bwd_dispatch`
  仍为 2 项。反向（bwd / fag）布局列表保持 2 项不变——BNSD 不进入反向。
- `FAIKernelParams` 新增 `int64_t kvSeqlenAllocated`（BNSD 下 KV 的 `S_alloc`，其余布局填 0），
  `FwdLaunchArgs` 同步新增 `inputLayout layout` 与 `int64_t kv_seqlen_allocated`。
- host 底层入口新增 `int64_t layout`：v2 `mha_fwd` / `mha_fwd_kvcache`，v3、v4 的 `mha_fwd`；
  探测在 Python 侧一次算完，host 只做范围与不变式校验。
- **明确不改**：`fag_layout.hpp` 与 `fag_common/common_header.h` 的 FAG 布局常量（反向仍只有 BSND / TND 两值）；
  910 v2 的 `mha_varlen_fwd`（rank-3 / TND）不参与 BNSD。
- append-KV 拒绝（910 特有）：v3 的 `k_new` / `v_new`（append-KV）或 `q_v` 非空且探测为 BNSD 时，
  host 报错拒绝（BNSD 仅覆盖非 append 的标准 KV 路径）。
- host 侧 `v2` 接口中 `layout=q.layout`（tensor layout，即 `torch.strided`）的既有用法未被破坏。

### 两代差异速览

| 差异项 | Ascend 950 | Ascend 910 |
|---|---|---|
| kernel 侧枚举名与取值 | `Format { TND = 0, BSND = 1, BNSD = 2 }`（`kernel_common.hpp`） | `inputLayout { BSND = 0, TND = 1, BNSD = 2 }`（三份各自独立） |
| host 侧布局枚举 | 有 `FwdLayout`（`tilingdata.h`），与 `Format` 逐项一致 | 无独立 `FwdLayout`；host 持 `int64_t layout`（`0=TND` / `1=BSND` / `2=BNSD`），并显式映射到 `inputLayout`（两者编号不同） |
| kernel 布局模板参数个数 | 两个（`Format qFormat, Format kvFormat`） | 一个（`inputLayout kLayout`，即 `INPUT_LAYOUT`） |
| dispatch 模板参数 | `<DType, Format kLayout>` | 由 `bool IS_TND` 改为 `<DType, inputLayout kLayout>` |
| fwd TU 变化 | v3 / v4 均 4 → 6（bwd 不变） | v3 fwd 4 → 6（bwd 4 不变）、v4 fwd 4 → 6；v2 `fwd_dispatch` 4 → 6、`fwd_combo` 32 → 48，`fag_general_dispatch` / `varlen_bwd_dispatch` 仍 2 项 |
| 接口文件 | `flash_attn_npu_interface_950.py`（v3 / v4） | `flash_attn_npu_interface.py`（v2 / v3 / v4） |
| 测试文件 | `tests/test_flash_attn_npu_v3_bnsd.py` / `_v4_bnsd.py` | `tests/test_flash_attn_npu_v2_bnsd_910.py` / `_v3_bnsd_910.py` / `_v4_bnsd_910.py` |
| 平台专项 | DN 快路径专项（`qHeadStride` 修复） | append-KV / `q_v` 拒绝；v2 FAG 常量与 `mha_varlen_fwd` 不改 |

## Impact

- Affected specs: 950 v3/v4 与 910 v2/v3/v4 的前向 kernel 寻址、kernel 入口参数、host 布局探测 / 分派、
  Python 探测逻辑、autogen TU 数量

- Affected code：
  - **950（v3）**：`csrc/ascend950/flash_attn_npu_3/{kernel_common.hpp,fwd_kernel.cpp,fwd_dispatch.hpp,
    fwd_dispatch_impl.hpp,fd_combine.hpp,autogen/generate_kernels.py,qk_matmul.hpp,init_outputs.hpp,
    rescale_o.hpp,tiling.cpp,tiling_from_tensors.hpp,mha_fwd.cpp,flash_api.cpp}`、
    `flash_attn_npu_3/flash_attn_npu_interface_950.py`
  - **950（v4）**：`csrc/ascend950/flash_attn_npu_4/` 对应文件、`flash_attn_npu_4/flash_attn_npu_interface_950.py`
  - **910（v2）**：`csrc/ascend910/flash_attn_npu/{kernel_common.hpp,fwd_dispatch.hpp,fwd_dispatch_impl.hpp,
    fwd_kernel_launch.hpp,mha_fwd_kvcache.cpp,qk_matmul.hpp,init_outputs.hpp,rescale_o.hpp,combine_scale.hpp,
    flash_api.cpp,autogen/generate_kernels.py}`、`flash_attn_npu/flash_attn_npu_interface.py`
  - **910（v3）**：`csrc/ascend910/flash_attn_npu_3/{kernel_common.hpp,fwd_dispatch.hpp,fwd_dispatch_impl.hpp,
    mha_fwd_kvcache.cpp,qk_matmul.hpp,init_outputs.hpp,rescale_o.hpp,combine_scale.hpp,flash_api.cpp,
    autogen/generate_kernels.py}`、`flash_attn_npu_3/flash_attn_npu_interface.py`
  - **910（v4）**：`csrc/ascend910/flash_attn_npu_4/` 同上对应文件
    （v4 无 `k_new/v_new`、无 `rotary`，有 `blockTableStride` / `gather_kv_indices`）、
    `flash_attn_npu_4/flash_attn_npu_interface.py`
  - **测试**：950 `tests/test_flash_attn_npu_v3.py`、`tests/test_flash_attn_npu_v4.py`、
    `docs/enable-qkv-bnsd/test_kv_seqlen_allocated.py`；910 `tests/test_flash_attn_npu_v2_bnsd_910.py`、
    `tests/test_flash_attn_npu_v3_bnsd_910.py`、`tests/test_flash_attn_npu_v4_bnsd_910.py`

- 明确不改：对外 Python 公开签名、`fa_metadata.aicpu`、`FAInferTilingData`（v3/v4）与
  `FAInferStaticTilingData` / `FAInferRuntimeData`（910 v4）字段集合、950 与 910 两侧的其它后端行为
  （BNSD 未启用时既有 BSND / TND 行为与数值完全不变）

- 明确改动：`autogen/` TU 数量按平台各增 fwd 实例（950 v3/v4 各 4 → 6；910 v3 fwd 4 → 6、bwd 4 不变、
  v4 fwd 4 → 6、v2 `fwd_dispatch` 4 → 6 且 `fwd_combo` 32 → 48）；
  `FAIKernelParams` 中布局布尔量删除，仅保留 `int64_t kvSeqlenAllocated`

- 编译 / 验证环境（两代分别）：
  - **950**：conda 环境 `ascend`，CANN `~/miniconda3/Ascend/cann/set_env.sh`（9.1.0），
    `FLASH_ATTN_BUILD_NPU=950`
  - **910**：`source /data/.sysvol/cann/cann-9.3.0/set_env.sh`（`ASCEND_HOME_PATH` 指向同一路径）+
    含 torch_npu 的 conda 环境（实测 `zzy311new`：torch 2.7.1 / torch_npu 2.7.1.post5 / Ascend910B3），
    构建命令 `FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_BUILD_VERSION=all python setup.py install`

## ADDED Requirements

### Requirement: BNSD 布局输入（950 与 910 共用）
系统 SHALL 在各后端（950 v3/v4；910 v2/v3/v4）的 `flash_attn_func` / `flash_attn_varlen_func` /
`flash_attn_with_kvcache` 中接受 BNSD 形式的 Q（`(B, H, S, D)`）与 KV（`(B, H_kv, S_alloc, D)`，
`with_kvcache` 下为 `k_cache` / `v_cache`），并 SHALL 保证返回的 `out` 形状与调用方传入的 `q` 形状一致。

> 平台差异：950 的 v3 入口为 `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache`，
> v4 无 `flash_attn_with_kvcache`；910 v3 三个入口齐全，v4 无 `flash_attn_with_kvcache`，
> v2 仅 `flash_attn_func` 与 `flash_attn_with_kvcache`（varlen / packed 按 BSND / TND 解读）。

#### Scenario: BNSD 转置视图输入（零拷贝）
- **WHEN** 调用方传入 `bnsd_q.transpose(1, 2)` / `bnsd_k.transpose(1, 2)` / `bnsd_v.transpose(1, 2)`
  （即 BSND 形状、strides 为 `(H*S*D, D, S*D, 1)` 的非连续视图），且三者全部命中 stride 判据
  （910 侧另要求未提供 `page_table` / `cu_seqlens_k`）
- **THEN** 系统把这组视图零拷贝还原为连续 BNSD 交给 kernel，不产生任何 `.contiguous()` 拷贝，
  数值与等价 BSND 调用一致（910 侧为逐元素相等），且返回的 `out` 形状等于传入的 `q` 形状

#### Scenario: 连续 BNSD 直接传入
- **WHEN** q 为 `(B, H, S, D)`、k/v 为 `(B, H_kv, S_alloc, D)` 的连续张量直接传入（未做 `transpose(1, 2)`）
- **THEN** 不判定为 BNSD（连续张量的 stride 是 shape 的纯函数，shape 无法与 BSND 区分），
  按 BSND 解读并原样返回传入 shape；`out.shape == q.shape`

#### Scenario: 混合输入
- **WHEN** Q、K、V 中只有部分命中 stride 判据（例如 q 为 BNSD 转置视图而 v 为普通 BSND）
- **THEN** 不做 stride 判定，整体按 BSND 解读；数值仍正确，仅可能多一次 `contiguous()` 拷贝

#### Scenario: 退化连续张量
- **WHEN** 传入的 BSND 张量恰好连续（含 `S == H == 1` 等退化形状）
- **THEN** `is_contiguous()` 短路生效，永远不会被误判为 BNSD 转置视图

#### Scenario: 不参与 BNSD 的输入
- **WHEN** 提供 `page_table`（paged KV）、`cu_seqlens_k`、rank-3 Q（TND）或 rank-3 KV（cu-packed）
- **THEN** 对应入口不做 BNSD 解读（BNSD KV 要求非 paged、rank-4）

#### Scenario: 非 stride 的 rank-4 输入
- **WHEN** rank-4、非 paged 的静态 Q 与 rank-4 KV 未命中 stride 判据
- **THEN** 按 BSND 解读（不做 shape 推断），`out` 原样以传入 shape 返回，不抛错

#### Scenario: KV 已分配长度大于实际使用长度
- **WHEN** BNSD KV 的 `S_alloc` 大于 `cache_seqlens` / `seqused_k` 给出的实际长度
- **THEN** 只读取实际长度范围，结果正确，且 `S_alloc`（910 侧为 `k.size(2)`）仅用于 head-plane / batch
  步长推导；`max(cache_seqlens) > S_alloc` 时报错

#### Scenario: BNSD 反向
- **WHEN** 对 950 或 910 的 BNSD 前向结果执行 backward（相应后端的 autograd）
- **THEN** 抛出明确 `NotImplementedError`，说明该后端 BNSD 仅支持前向

### Requirement: 布局探测（950 与 910 共用）
系统 SHALL 采用「stride 严格匹配」这一唯一信号自动推断布局（rank-4、非连续、且
`stride(3) == 1 && stride(1) == shape[3] && stride(2) == shape[1] * shape[3]`），
要求 Q、K、V 三者全部命中才判定 BNSD，且 SHALL NOT 要求调用方传入任何布局参数。

#### Scenario: stride 命中
- **WHEN** Q、K、V 三者全部命中 stride 判据
- **THEN** 采用零拷贝 BNSD 路径

#### Scenario: 混合输入
- **WHEN** Q、K、V 中只有部分命中 stride 判据（例如 q 为 BNSD 转置视图而 v 为普通 BSND）
- **THEN** 不做 stride 判定，整体按 BSND 解读；数值仍正确，仅可能多一次 `contiguous()` 拷贝

#### Scenario: 退化连续张量
- **WHEN** 传入的 BSND 张量恰好连续（含 `S == H == 1` 等退化形状）
- **THEN** `is_contiguous()` 短路生效，永远不会被误判为 BNSD 转置视图

#### Scenario: 非 stride 的 rank-4 输入
- **WHEN** rank-4、非 paged 的静态 Q 与 rank-4 KV 未命中 stride 判据
- **THEN** 按 BSND 解读（不做 shape 推断），`out` 原样以传入 shape 返回，不抛错

#### Scenario: 不参与 BNSD 的输入
- **WHEN** 提供 `page_table`（paged KV）、`cu_seqlens_k`、rank-3 Q（TND）或 rank-3 KV（cu-packed）
- **THEN** 对应入口不做 BNSD 解读（BNSD KV 要求非 paged、rank-4）

#### Scenario: 探测先于 contiguity 处理（910）
- **WHEN** 910 入口按既有逻辑对操作数做 `maybe_contiguous`（v2/v3/v4 只按末维 `stride(-1)` 判断）
- **THEN** 探测在原始输入上进行、在其之前完成；BNSD 路径下还原后的连续 BNSD 张量是 no-op，
  不产生额外拷贝

### Requirement: 公开接口不新增布局参数（950 与 910 共用）
系统 SHALL 保持 `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` 及 packed 变体的公开签名不变。

#### Scenario: 拒绝显式参数
- **WHEN** 调用方传入 `bnsd=` / `q_bnsd=` / `kv_bnsd=` 关键字
- **THEN** Python 抛出 `TypeError`（未知关键字），布局只能由自动探测决定

#### Scenario: 向后兼容
- **WHEN** 未启用 BNSD（既有 BSND / TND 调用）
- **THEN** 入参语义、输出形状、LSE 形状与数值与改动前完全一致

### Requirement: 【950】BNSD 前向约束
系统 SHALL 在 950 的 BNSD 路径上关闭不适用快路径，并在反向拒绝 BNSD。

#### Scenario: scheduler metadata
- **WHEN** 探测到 BNSD 布局
- **THEN** 复用与 BSND 相同的 `scheduler_metadata`（AICPU metadata 只承载数字、不含布局语义，
  尺寸一律取 `_detect_layout` 检测前的 BSND 形状）；显式传入的 metadata 同样有效
- **AND** host 侧 metadata 分支的 KV 上界按布局取序列维（BNSD 读 `k.size(2)`，非 BNSD 读 `k.size(1)`）

#### Scenario: DN 快路径
- **WHEN** 布局为 BNSD 且 `mask_category == NO_MASK`
- **THEN** DN 快路径与 BSND/TND 同样生效：DN 的 `loadQGM` 通过新增的 `qHeadStride` 形参
  （BNSD 为 `maxQSeqlen * embed_`，BSND/TND 为 `embed_`）适配各布局的 Q head 平面间距，
  它同时是 Nd2Nz 的 `srcNdMatrixStride`
- **AND** `kUseDnFastPath = (MaskCat == MaskCategory::NO_MASK) && IsDN`，
  因此 `FAInferDn<BNSD>` 在非 masked 时被实例化（masked 仍走 `FAInfer`）

#### Scenario: KV 长度校验
- **WHEN** `max(cache_seqlens)` 或 `max(seqused_k)` 大于 BNSD KV 的 `S_alloc`
- **THEN** 报错，提示序列维不足以覆盖实际长度

#### Scenario: 布局互斥校验
- **WHEN** BNSD 与 FlashDecode 同时被选中
- **THEN** FlashDecode 不启用（FD 仅用于 paged KV，与 BNSD 互斥）

#### Scenario: BNSD 反向
- **WHEN** 对 BNSD 前向结果执行 backward
- **THEN** 抛出明确 `NotImplementedError`，说明 950 后端 BNSD 仅支持前向

### Requirement: 【910】前向 kernel 按布局推导寻址
系统 SHALL 在 910 三个后端的编译期布局参数（`inputLayout kLayout`）驱动下推导前向寻址，
BSND / TND 的取值与改动前逐项一致：

- Q：`strideQ = (BNSD) ? embed : qHeads*embed`；`qHeadStride = (BNSD) ? maxQSeqlen*embed : embed`
- O：`strideO = (BNSD) ? embedV : qHeads*embedV`；`oHeadStride = (BNSD) ? maxQSeqlen*embedV : embedV`
- K/V：`strideK = (BNSD) ? embed : kvHeads*embed`（V 用 `embedV`）；
  `kvHeadStrideK = (BNSD) ? kvSeqlenAllocated*embed : embed`（V 同理）；
  batch 推进 `kBatchStride = (BNSD) ? kvHeads*kvHeadStrideK : kvPhysicalSeqlen*strideK`
- GM offset：`gmOffsetQ = qBOffset + qS*strideQ + qNStartIdx*qHeadStride`（O/K/V 同理）；
  batch 基址 `qBOffset = (BNSD) ? BIdx*qHeads*qHeadStride : prevQSeqlenSum*strideQ`
- KV 实际长度：BNSD 与 BSND 一致，取 `gActualKvseqlen.GetValue(BIdx)`（per-batch 值），
  不使用 TND 的 cu 差分

#### Scenario: BSND / TND 回归
- **WHEN** `kLayout == inputLayout::BSND` 或 `inputLayout::TND`
- **THEN** 上述每个量的取值与改动前完全相同，数值不变

#### Scenario: BNSD 平面步长
- **WHEN** `kLayout == inputLayout::BNSD`
- **THEN** head plane 宽为常量 `S_alloc*D`，batch 步长也为常量；`S_alloc` 来自 `kvSeqlenAllocated` 而非 `maxKvSeqlen`

### Requirement: 【910】LSE 与输出写回
系统 SHALL 使 910 的 BNSD 的 LSE 与输出写回与 BSND 语义一致，且 SHALL NOT 从 `qBOffset / strideQ`
反推 batch 索引（BNSD 下 `strideQ == embed`，反推会错）。

#### Scenario: LSE
- **WHEN** 布局为 BNSD
- **THEN** kernel 侧 LSE 仍为 `(B, H, S)`，其 batch/head 基址由循环内 batch 计数与 `qHeads` 直接给出
  （910 现有公式 `lseBOffset = prevQSeqlenSum*qHeads`、`lseHeadStride = maxQSeqlen` 在
  `prevQSeqlenSum = BIdx*maxQSeqlen` 下取值与 BSND 相同），各公开入口对 LSE 的既有形状约定不变

#### Scenario: 输出 / FD combine 写回
- **WHEN** 布局为 BNSD
- **THEN** per-head 目的地址按 `oHeadStride` 计算（BNSD 为 `maxQSeqlen*embedV`）；
  BSND 下 `oHeadStride == embedV`，与改动前逐项等价；FlashDecode（仅 paged KV）与 BNSD 天然互斥，
  `combine_scale.hpp` 的 `headStride` 分支无需为 BNSD 新增语义

#### Scenario: GM→L1 拷贝
- **WHEN** 布局为 BNSD
- **THEN** `qk_matmul.hpp` 中把 head 平面间距写成常量 `qHeads*embed` 的位置改为按布局推导
  （BNSD 用 `maxQSeqlen*embed`），L1 目的布局不变

### Requirement: 【910】边界与拒错
系统 SHALL 在 910 的 BNSD 上关闭不适用的路径并给出明确错误信息。

#### Scenario: paged KV
- **WHEN** 提供 `page_table`（paged KV）
- **THEN** 永不判为 BNSD；FlashDecode 亦要求 paged KV，两者互斥

#### Scenario: append-KV 与 q_v
- **WHEN** v3 的 `k_new` / `v_new`（append-KV）或 `q_v` 非空且探测为 BNSD
- **THEN** host 报错拒绝（BNSD 仅覆盖非 append 的标准 KV 路径，append-KV 的 nd2nz 写回不在本次范围）

#### Scenario: KV 序列维不足
- **WHEN** `max(cache_seqlens)` 或 `max(seqused_k)` 大于 BNSD KV 的 `S_alloc`
- **THEN** 报错，提示序列维不足以覆盖实际长度

#### Scenario: rank / 布局互斥校验
- **WHEN** host 收到 `layout` 码
- **THEN** 校验 `layout ∈ {0,1,2}`、`(layout == TND) ⟺ cu_seqlens_q 非空`、
  BNSD 要求 q/k/v 均 rank-4 且 k/v 的 (batch, heads, seqlen) 三维一致

## MODIFIED Requirements

### Requirement: host 侧布局枚举（仅 950）
用 `FwdLayout { TND = 0, BSND = 1, BNSD = 2 }`（`csrc/ascend950/flash_attn_npu_{3,4}/tilingdata.h`）
替换原 `bool is_tnd`，取值与 `kernel_common.hpp` 的 `Format { TND = 0, BSND = 1, BNSD = 2 }` 逐项一致，
因此 host 侧可用 `static_cast<Format>(layout_enum)` 直接得到编译期模板实参。
`fill_inference_context` 增加 `FwdLayout layout` 入参；
`tiling.cpp` 的两处 cumulated-kv 读取分支只在 `FwdLayout::TND` 时执行，
BNSD 下要求 per-batch KV 长度列表（拒绝 `batch + 1` 的累加形式）。**未新增 `FAInferTilingData` 字段**。

> 910 无 host 侧 `FwdLayout`：host 持 `int64_t layout`（`0=TND` / `1=BSND` / `2=BNSD`），
> 用三元表达式 / `switch` 显式映射到 kernel 的 `inputLayout { BSND = 0, TND = 1, BNSD = 2 }`（编号不同）。

### Requirement: 前向 dispatch 与 autogen（950 与 910 分列）
- **950**：`launch_fwd_impl` 的模板参数由 `<DType, bool IS_TND>` 改为 `<DType, Format kLayout>`，
  `launch_fwd` 按 `a.layout` 做 `(fp16, bf16) × (TND, BSND, BNSD)` 共 6 路分派；
  `FWD_KERNEL_LAUNCH` 把 `kLayout` 同时喂给 kernel 的 `qFormat` 与 `kvFormat`。
  `autogen/generate_kernels.py` 的 `LAYOUTS` 增加 `("bnsd", "BNSD", "Format::BNSD")`，
  生成 6 个 per-(dtype, layout) TU（`setup.py` 用 glob 收集，无需改动）。
- **910**：三个后端的 fwd dispatch 模板参数由 `bool IS_TND` 改为 `inputLayout kLayout`：
  `autogen/generate_kernels.py` 的 `LAYOUTS` 条目携带布局令牌（fwd 三值：`inputLayout::BSND` /
  `inputLayout::TND` / `inputLayout::BNSD`），fwd TU 由 4 增至 6（v2 的 `fwd_combo` 由 32 增至 48）；
  反向（v3 的 `bwd_dispatch_*`、v2 的 `fag_general_dispatch_*` / `varlen_bwd_dispatch_*`）布局列表保持 2 项，
  不生成 BNSD 反向 TU。`setup.py` 用 `glob(autogen/*.cpp)` 收集源码，无需改动。

### Requirement: kernel 入口参数与 `FwdLaunchArgs`（950 与 910 共用骨架）
- **950**：`FAIKernelParams` 新增 `int64_t kvSeqlenAllocated`（BNSD KV 的 `S_alloc`，其余布局填 0），
  不新增任何布局布尔量；`FAInfer` / `FAInferDn` 的 `CATLASS_GLOBAL` 签名与 `FWD_KERNEL_LAUNCH` 宏
  同步增加该参数，布局由模板参数 `Format qFormat, Format kvFormat` 承载。
  `FAFlashDecodeCombine` 的 kernel 签名不变（FD 仅用于 paged KV，与 BNSD 互斥），
  其 `FAIKernelParams` 聚合初始化的 `kvSeqlenAllocated` 填 `0`。
  `FwdLaunchArgs` 新增 `layout`（`Format`）与 `kv_seqlen_allocated`（int64，BNSD KV 的 `S_alloc`）运行时段。
- **910**：三份 `FAIKernelParams` 新增 `int64_t kvSeqlenAllocated`，构造参数同步；
  `FwdLaunchArgs` 新增 `inputLayout layout` 与 `int64_t kv_seqlen_allocated`；`FWD_LAUNCH` 宏把二者传入 kernel 例化。
  布局由单一模板参数 `inputLayout kLayout`（`INPUT_LAYOUT`）承载。
  两代均不再有运行期 `bnsd` 布尔量。

### Requirement: 【950】kernel 步长推导
以编译期 `qFormat` / `kvFormat`（`if constexpr`）推导步长，BSND/TND 取值与改动前逐项一致：

- Q：`strideQ = (qFormat == Format::BNSD) ? embed_ : qHeads_ * embed_`；
  `qHeadStride = (qFormat == Format::BNSD) ? maxQSeqlen * embed_ : embed_`
- O：`strideO = (qFormat == Format::BNSD) ? embedV_ : qHeads_ * embedV_`；
  `oHeadStride = (qFormat == Format::BNSD) ? maxQSeqlen * embedV_ : embedV_`
- K/V：`strideK = (kvFormat == Format::BNSD) ? embed_ : kvHeads_ * embed_`（V 同理用 `embedV_`）；
  `kvHeadStrideK = (kvFormat == Format::BNSD) ? kvSeqlenAllocated * embed_ : embed_`（V 同理）；
  batch 累加 `kBatchStride = (kvFormat == Format::BNSD) ? kvHeads_ * kvHeadStrideK : kvPhysicalSeqlen * strideK`
- GM offset：`gmOffsetQ = qBOffset + qSOffset * strideQ + qNStartIdx * qHeadStride`；
  `gmOffsetK = kBOffset + kvHeadIdx * kvHeadStrideK`（O/V 同理）
- LSE offset：BNSD 时 `(curBatch * qHeads_ + qNStartIdx) * maxQSeqlen + qSOffset`，
  其余沿用现有 BSND/TND 公式（用循环中已有的 batch 计数 `curBatch`，不用 `qBOffset / strideQ` 反推）
- BNSD KV 的实际长度：`kvNumTokens = gActualKvseqlen.GetValue(batch_ - 1)`
  （per-batch 值，不使用 TND 的 cu 差分）；
  同一处原 `if constexpr (kvFormat == Format::TND) { if (!params.bnsd) {...} }` 的死守卫一并删除，
  直接按 `kvFormat` 分支

#### Scenario: BSND/TND 回归
- **WHEN** `qFormat == kvFormat == Format::BSND`（或 `Format::TND`）
- **THEN** 所有步长与偏移的取值与改动前完全相同，数值不变

#### Scenario: BNSD 平面步长
- **WHEN** `qFormat == kvFormat == Format::BNSD`
- **THEN** 每 batch 内 head plane 宽 `S_alloc * D`（`S_alloc` 为常量，因此 batch 步长也是常量）

### Requirement: Nd2Nz 源步长与输出写回
`qk_matmul.hpp` 的 `loadQGM` 增加运行时的 head-plane 源间距参数（BSND 传 `embed_`，BNSD 传 `maxQSeqlen * embed_`），
L1 目的布局保持不变；`qk_matmul_DN.hpp` 的 `loadQGM` 增加同一参数并作为 Nd2Nz 的 `srcNdMatrixStride`
（改动前硬编码为 `embed`，仅在 BSND/TND 成立），因此 BNSD 也支持 DN 快路径。
`init_outputs.hpp` 与 `rescale_o.hpp` 的写回路径增加 `oHeadStride` 运行时参数，
缺省语义等价于 `embedV_`，保证 BSND 路径行为不变。

### Requirement: host 布局分派（910）
910 三个后端的 host 底层入口新增 `int64_t layout`：
- v3 / v4 `flash_api.cpp::mha_fwd`：原 `is_varlen_q` 处的 `launch_fwd<true>/<false>` 改为按 `layout` 三分派；
  BNSD 下 `num_heads = sizes[1]`、`seqlen_q = sizes[2]`、`num_heads_k = k.size(1)`、`kv_seqlen_allocated = k.size(2)`
- v2 `flash_api.cpp::mha_fwd`（BSND rank-4 入口）与 `mha_fwd_kvcache`（rank-4 cache 入口）：同样新增 `layout`
  并派生上述量；`mha_varlen_fwd`（rank-3 / TND）不参与 BNSD
- `out` 沿用既有 `torch::empty_like(q)` 分配，因此 BNSD 下自然得到 `(B,H,S,D)`（910 强制 `v.size(-1) == head_size_q`），
  Python 侧再转置回调用方形状
- `get_scheduler_metadata` 与 metadata 复用：BNSD 复用与 BSND 相同的 metadata（只承载数字、不含布局语义），
  尺寸一律取探测**之前**的 BSND 形状；host 侧 metadata 分支的 KV 上界按布局取序列维（BNSD 读 `k.size(2)`）

### Requirement: Python 侧探测与调用点（950 与 910 分列）
两代均提供统一的探测辅助函数，并用与 C++ 一致的布局码
（`_LAYOUT_TND = 0` / `_LAYOUT_BSND = 1` / `_LAYOUT_BNSD = 2`）把探测结果透传给 kernel：

- `_transposed_bnsd_view(t)`：rank-4、非连续、且
  `stride(3) == 1 && stride(1) == shape[3] && stride(2) == shape[1] * shape[3]` 时返回 `t.transpose(1, 2)`，否则 `None`。
- `_detect_layout(q, k, v, page_table, cu_seqlens_k) -> (layout, q, k, v, out_transposed)`：
  仅尝试三者全部命中 stride 判据（命中则返回还原后的连续 BNSD 与 `out_transposed = True`），
  否则返回原样操作数、`out_transposed = False` 与 `_LAYOUT_BSND`。

调用点在返回前按 `out_transposed` 把 `out` 转置回调用方形状。

- **950**：v3/v4 的 `flash_attn_npu_interface_950.py` 实现上述辅助；`_flash_attn_forward` 与其 fake、以及
  C++ `fwd` 绑定以底层 `layout: int` 形参把探测结果透传给 kernel（不再有 `bnsd` 布尔量）；
  `_training_forward` / `FlashAttnFunc` / `FlashAttnVarlenFunc` 一律用 `ctx.layout` 记录并在 backward 首行拒绝 `_LAYOUT_BNSD`。
  v3 的 `flash_attn_varlen_func`（rank-3 packed）恒为 `_LAYOUT_TND`；
  v4 的 `FlashAttnVarlenFunc` 在 `cu_seqlens_q is not None` 时为 `_LAYOUT_TND`，否则为 `_LAYOUT_BSND`。
- **910**：v2 / v3 / v4 三份 `flash_attn_npu_interface.py` 各自新增同一套辅助函数。
  自定义算子（v2/v3/v4 的 `_flash_attn_forward` 及其 fake）与 pybind 底层入口
  （`flash_attn_npu(.fwd/.fwd_kvcache)`、`flash_attn_npu_3.fwd`、`flash_attn_npu_4.fwd`）
  以 `layout: int` 形参承接探测结果；autograd 用 `ctx.layout` 记录，backward 首行拒绝 `_LAYOUT_BNSD`。
  v3 `flash_attn_varlen_func` 恒为 `_LAYOUT_TND`；v4 的 `FlashAttnVarlenFunc` 在
  `cu_seqlens_q is not None` 时为 `_LAYOUT_TND`，否则为 `_LAYOUT_BSND`。

两代不变式一致：`layout == _LAYOUT_TND ⟺ cu_seqlens_q is not None`（host 侧同一校验）。

## REMOVED Requirements

无
