# 950 后端 QKV 支持 BNSD 布局（仅前向）Spec

## Why
Ascend 950 后端的 `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` 目前只接受
BSND（`(B, S, H, D)`）与 TND（varlen packed）布局。推理侧 Q 与 KV cache 常以
BNSD（`(B, H, S, D)`，per-head plane）形式存放，调用方需要 `q.transpose(1, 2)` 转成 BSND 形状再喂入，
这不仅在长序列 / 长 cache 下产生明显额外拷贝与显存开销，也让「输入到底是 BSND 还是 BNSD 转置视图」
这一语义在接口上不可表达。需要在 950 的 Python 接口层与 CANN 内核层原生支持 BNSD 前向，
且**不改变对外 Python 接口参数**。

本次为**独立重写**：不复用、不合入 `bsnd` 分支的任何代码或提交，v3 与 v4 两个 950 后端各自按本设计实现。

## What Changes
- 新增 `Format::BNSD` / host 侧 `FwdLayout`：把原来的 `bool is_tnd` 重构为三值枚举
  `FwdLayout { TND = 0, BSND = 1, BNSD = 2 }`，统一承载 q/k/v 输入布局语义
  （BNSD 与 TND 天然互斥：BNSD 要求 rank-4 query）。
- **单一编译期布局参数 `Format`**：Q、K、V 理论上同一种输入格式，因此只用一个 `Format` 值承载布局语义，
  并且它是 kernel 的**编译期模板参数**（`qFormat` / `kvFormat`），由 kernel 内 `if constexpr` 推导
  token / head-plane / batch 步长。业务分支不再依赖任何运行期 `bnsd` 布尔量。
  dispatch TU 由 4 个增至 6 个（`(fp16, bf16) × (TND, BSND, BNSD)`），
  在 `autogen/generate_kernels.py` 中新增 `bnsd` 行即可，`setup.py` 用 glob 收集无需改动。
- **对外 Python 公开签名不变**：`flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache`
  均不新增 `bnsd`（或 `q_bnsd` / `kv_bnsd`）参数。布局完全由自动探测决定，探测信号为：

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
- **不修改** `FAInferTilingData` 的字段集合，也不触碰 `fa_metadata.aicpu`（预编译 AICPU 产物的 ABI 不动），
  布局信息全部走 kernel 入口参数与 host 侧 `FwdLayout`。
- kernel 寻址改造：Q/O 的 token 步长与 head-plane 步长、K/V 的 per-head plane 与 batch 步长、
  `gmOffset{Q,K,V,O}`、LSE offset；输出写回与 Nd2Nz 源步长参数化。
- BNSD **仅前向**：v3 的 autograd backward 遇 BNSD 抛明确 `NotImplementedError`，不实现 BNSD 反向。
- v4 后端同样改造其 `flash_attn_func` / `flash_attn_varlen_func`（v4 无 `flash_attn_with_kvcache`）；
  v4 varlen query 仅 rank-3，因此 packed（`cu_seqlens_q` 存在）路径恒为 `FwdLayout::TND`，
  否则为 `FwdLayout::BSND`。
- C++ 侧 `mha_fwd` 的底层入口参数由 `std::optional<bool> bnsd_` 换成 `int64_t layout`
  （0 = TND、1 = BSND、2 = BNSD，与 `FwdLayout` / `Format` 取值一致，Python 公开层不透出）：
  host 只做取值范围校验与不变式校验 `layout == TND ⟺ cu_seqlens_q is not None`，
  布局探测结果由 Python 侧一次算出后透传，host 不再重复探测。
- 不改动 910 后端；BNSD 未启用时既有 BSND/TND 行为与数值完全不变。

## Impact
- Affected specs: 950 v3/v4 前向 kernel 寻址、kernel 入口参数、host 布局探测、Python 探测逻辑
- Affected code:
  - v3：`csrc/ascend950/flash_attn_npu_3/{kernel_common.hpp,fwd_kernel.cpp,fwd_dispatch.hpp,fwd_dispatch_impl.hpp,
    fd_combine.hpp,autogen/generate_kernels.py,qk_matmul.hpp,init_outputs.hpp,rescale_o.hpp,tiling.cpp,
    tiling_from_tensors.hpp,mha_fwd.cpp,flash_api.cpp}`、
    `flash_attn_npu_3/flash_attn_npu_interface_950.py`
  - v4：`csrc/ascend950/flash_attn_npu_4/` 对应文件、`flash_attn_npu_4/flash_attn_npu_interface_950.py`
  - 测试：`tests/test_flash_attn_npu_v3.py`、`tests/test_flash_attn_npu_v4.py`、
    `docs/enable-950-qkv-bnsd/test_kv_seqlen_allocated.py`
- 明确不改：对外 Python 公开签名、`fa_metadata.aicpu`、`tilingdata.h` 的 `FAInferTilingData` 字段集合、910 后端
- 明确改动：`autogen/` TU 数量由 4 增至 6（新增 `bnsd` 编译实例）；`FAIKernelParams` 中 `bool bnsd` 删除，
  仅保留 `int64_t kvSeqlenAllocated`
- 编译环境：conda 环境 `ascend`，CANN `~/miniconda3/Ascend/cann/set_env.sh`（9.1.0），`FLASH_ATTN_BUILD_NPU=950`

## ADDED Requirements

### Requirement: BNSD 布局输入
系统 SHALL 在 Ascend 950 后端的 `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` 中
接受 BNSD 形式的 Q（`(B, H, S, D)`）与 KV（`(B, H_kv, S_alloc, D)`，`with_kvcache` 下为 `k_cache`/`v_cache`），
并 SHALL 保证返回的 `out` 形状与调用方传入的 `q` 形状一致。

#### Scenario: 连续 BNSD 直接传入
- **WHEN** q 为 `(B, H, S, D)`、k/v 为 `(B, H_kv, S_alloc, D)` 的连续张量直接传入（未做 `transpose(1, 2)`）
- **THEN** 不判定为 BNSD（shape 无法与 BSND 区分），按 BSND 解读并原样返回传入 shape；
  `out.shape == q.shape`

#### Scenario: BNSD 转置视图输入（零拷贝）
- **WHEN** 调用方传入 `bnsd_q.transpose(1, 2)` / `bnsd_k.transpose(1, 2)` / `bnsd_v.transpose(1, 2)`
  （即 BSND 形状、strides 为 `(H*S*D, D, S*D, 1)` 的非连续视图），且三者全部命中 stride 判据
- **THEN** 系统把这组视图零拷贝还原为连续 BNSD 交给 kernel，不产生任何 `.contiguous()` 拷贝，
  数值与等价 BSND 调用一致，且返回的 `out` 形状等于传入的 `q` 形状

#### Scenario: KV 已分配长度大于实际使用长度
- **WHEN** BNSD KV 的 `S_alloc` 大于 `cache_seqlens` / `seqused_k` 给出的实际长度
- **THEN** 只读取实际长度范围，结果正确，且 `S_alloc` 仅用于平面步长推导

### Requirement: 布局探测
系统 SHALL 采用「stride 严格匹配」这一唯一信号自动推断布局，
且 SHALL NOT 要求调用方传入任何布局参数。

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

### Requirement: 公开接口不新增布局参数
系统 SHALL 保持 `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` 的公开签名不变。

#### Scenario: 拒绝显式参数
- **WHEN** 调用方传入 `bnsd=` / `q_bnsd=` / `kv_bnsd=` 关键字
- **THEN** Python 抛出 `TypeError`（未知关键字），布局只能由自动探测决定

#### Scenario: 向后兼容
- **WHEN** 未启用 BNSD（既有 BSND / TND 调用）
- **THEN** 入参语义、输出形状、LSE 形状与数值与改动前完全一致

### Requirement: BNSD 前向约束
系统 SHALL 在 BNSD 路径上关闭不适用快路径，并在反向拒绝 BNSD。

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

## MODIFIED Requirements

### Requirement: host 侧布局枚举
用 `FwdLayout { TND = 0, BSND = 1, BNSD = 2 }`（`tilingdata.h`）替换原 `bool is_tnd`，
取值与 `kernel_common.hpp` 的 `Format { TND = 0, BSND = 1, BNSD = 2 }` 逐项一致，
因此 host 侧可用 `static_cast<Format>(layout_enum)` 直接得到编译期模板实参。
`fill_inference_context` 增加 `FwdLayout layout` 入参；
`tiling.cpp` 的两处 cumulated-kv 读取分支只在 `FwdLayout::TND` 时执行，
BNSD 下要求 per-batch KV 长度列表（拒绝 `batch + 1` 的累加形式）。**未新增 `FAInferTilingData` 字段**。

### Requirement: 950 前向 dispatch
`launch_fwd_impl` 的模板参数由 `<DType, bool IS_TND>` 改为 `<DType, Format kLayout>`，
`launch_fwd` 按 `a.layout` 做 `(fp16, bf16) × (TND, BSND, BNSD)` 共 6 路分派；
`FWD_KERNEL_LAUNCH` 把 `kLayout` 同时喂给 kernel 的 `qFormat` 与 `kvFormat`。
`autogen/generate_kernels.py` 的 `LAYOUTS` 增加 `("bnsd", "BNSD", "Format::BNSD")`，
生成 6 个 per-(dtype, layout) TU（`setup.py` 用 glob 收集，无需改动）。
`FwdLaunchArgs` 新增 `kv_seqlen_allocated`（int64，BNSD KV 的 `S_alloc`）运行时段，
布局本身由 `layout` 字段（`Format`）承载，不再有 `bnsd` 运行期布尔量。

### Requirement: kernel 入口参数
`FAIKernelParams` 新增 `int64_t kvSeqlenAllocated`（BNSD KV 的 `S_alloc`，其余布局填 0），
不新增任何布局布尔量；`FAInfer` / `FAInferDn` 的 `CATLASS_GLOBAL` 签名与 `FWD_KERNEL_LAUNCH` 宏
同步增加该参数，布局由模板参数 `Format qFormat, Format kvFormat` 承载。
`FAFlashDecodeCombine` 的 kernel 签名不变（FD 仅用于 paged KV，与 BNSD 互斥），
其 `FAIKernelParams` 聚合初始化的 `kvSeqlenAllocated` 填 `0`。

### Requirement: 950 kernel 步长推导
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

### Requirement: Python 侧探测与调用点
v3/v4 的 `flash_attn_npu_interface_950.py` 提供统一的探测辅助函数，并用与 C++ 取值一致的布局码
（`_LAYOUT_TND = 0` / `_LAYOUT_BSND = 1` / `_LAYOUT_BNSD = 2`）把探测结果透传给 kernel：

- `_transposed_bnsd_view(t)`：rank-4、非连续、且
  `stride(3) == 1 && stride(1) == shape[3] && stride(2) == shape[1] * shape[3]` 时返回 `t.transpose(1, 2)`，否则 `None`。
- `_detect_layout(q, k, v, page_table, cu_seqlens_k) -> (layout, q, k, v, out_transposed)`：
  仅尝试三者全部命中 stride 判据（命中则返回还原后的连续 BNSD 与 `out_transposed = True`），
  否则返回原样操作数、`out_transposed = False` 与 `_LAYOUT_BSND`。

调用点在返回前按 `out_transposed` 把 `out` 转置回调用方形状。
`_flash_attn_forward` 与其 fake、以及 C++ `fwd` 绑定以底层 `layout: int` 形参把探测结果透传给 kernel
（不再有 `bnsd` 布尔量）；`_training_forward` / `FlashAttnFunc` / `FlashAttnVarlenFunc` 一律用
`ctx.layout` 记录并在 backward 首行拒绝 `_LAYOUT_BNSD`。
v3 的 `flash_attn_varlen_func`（rank-3 packed）恒为 `_LAYOUT_TND`；
v4 的 `FlashAttnVarlenFunc` 在 `cu_seqlens_q is not None` 时为 `_LAYOUT_TND`，否则为 `_LAYOUT_BSND`。
不变式：`layout == _LAYOUT_TND ⟺ cu_seqlens_q is not None`（host 侧同一校验）。

## REMOVED Requirements
无
