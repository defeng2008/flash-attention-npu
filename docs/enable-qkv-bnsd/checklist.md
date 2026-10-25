# Checklist

> 覆盖两代：`## 通用（950 与 910 共用）` / `## 950` / `## 910`。
> 每个平台章节下保留原文档的分区标题（三级标题）。

## 通用（950 与 910 共用）

- [x] 使用**单一**编译期布局参数承载 Q/K/V 三者布局（未区分 `q_bnsd` / `kv_bnsd`）：
      950 最终形态为 `Format kLayout`（模板参数）、910 为 `inputLayout kLayout`；两代均无任何运行期 `bnsd` 布尔量
- [x] 两代对外 Python 公开签名（`flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` /
      packed 变体）均**未新增** `bnsd` / `q_bnsd` / `kv_bnsd` 参数；传入这些关键字时抛 `TypeError`
- [x] 两代 `FAInferTilingData`（950 v3/v4、910 v3/v4）与 `FAInferStaticTilingData` / `FAInferRuntimeData`（910 v4）
      字段集合均未改动
- [x] 两代 `fa_metadata.aicpu` 预编译产物均未重新生成、未改动
- [x] 两代 BNSD 只实现前向，未实现 BNSD 反向

## 950

### 设计约束（本次独立重写）

- [x] 未复用、未合入 `bsnd` 分支的任何提交或代码，v3 与 v4 各自独立实现
- [x] host 侧用 `FwdLayout { TND = 0, BSND = 1, BNSD = 2 }` 三值枚举替换原 `bool is_tnd`，
      取值与 `Format` 逐项一致，可直接 `static_cast` 互转
- [x] 模板实参为 `launch_fwd_impl<DType, Format kLayout>`；`autogen/` TU 由 4 增至 6
      （`(fp16, bf16) × (TND, BSND, BNSD)`），`setup.py` 用 glob 收集无需改动

### v3 CANN 层

- [x] `kernel_common.hpp` 的 `Format` 含 `BNSD = 2`，`FAIKernelParams` 含 `int64_t kvSeqlenAllocated`
      （无 `bool bnsd` 成员）
- [x] `FAInfer` 与 `FAInferDn` 的 `CATLASS_GLOBAL` 签名、`FWD_KERNEL_LAUNCH` 宏、`FwdLaunchArgs` 三处参数一致
- [x] `fd_combine.hpp` 的 `FAIKernelParams` 聚合初始化已补齐（`kvSeqlenAllocated = 0`），无参数个数不匹配
- [x] `fwd_kernel.cpp` 中 Q/O 的 token 与 head-plane 步长按 `if constexpr (qFormat == Format::BNSD)` 推导
- [x] `fwd_kernel.cpp` 中 K/V 的 per-head plane 步长按 `kvFormat == Format::BNSD` 与 `kvSeqlenAllocated` 推导，
      batch 步长同步
- [x] `GetKvSeqLen` 已删除 `bool bnsd` 形参，改按 `kvFormat` 分支；两处 `if (!params.bnsd)` 死守卫已删除
- [x] `gmOffsetQ` / `gmOffsetO` / `gmOffsetK` / `gmOffsetV` 含 BNSD 正确寻址
- [x] LSE offset 的 BNSD 分支使用循环内 `curBatch` 且含完整 `S` 步长，BSND/TND 公式未变
- [x] BNSD 时 KV 物理长度取 per-batch 值，不使用 TND 的 cu 差分
- [x] `loadQGM`（非 DN）支持运行时 head-plane 源间距；`qk_matmul_DN.hpp` 未被破坏
- [x] `init_outputs.hpp` / `rescale_o.hpp` 支持 `oHeadStride`，BSND 取值与改动前等价
- [x] K/V 的 tile 读取（`LayoutK`/`LayoutV`/`kShapeRow` 等）在 BNSD 下寻址正确

### v3 host 层

- [x] `mha_fwd.cpp` 的 `mha_fwd` 接受 `int64_t layout`（0 = TND / 1 = BSND / 2 = BNSD，底层入口），
      `flash_api.cpp` 绑定随签名自动同步
- [x] host 只做取值范围校验与不变式校验 `layout == FwdLayout::TND ⟺ cu_seqlens_q is not None`；
      探测（stride 严格匹配）在 Python 侧一次完成，host 不重复探测
- [x] 探测结果正确修正 `num_heads` / `seqlen_q` / `num_heads_k`，且与 `head_size` 校验一致
- [x] BNSD KV 校验 `max(seqused_k) <= S_alloc`，k/v 的 (batch, heads, seqlen) 维一致
- [x] paged KV 与 rank-3 cu-packed KV/TND query 永不判为 BNSD
- [x] BNSD 复用与 BSND 相同的 metadata：尺寸取检测前的 BSND 形状，host metadata 分支的 KV 上界按布局取序列维
- [x] BNSD 支持 DN 快路径：DN 版 `loadQGM` 接收 `qHeadStride` 作为 `srcNdMatrixStride`（BNSD 为 `maxQSeqlen * embed`）
- [x] `fill_inference_context` 支持 `FwdLayout::BNSD`（kv 列表为 per-batch，拒绝 batch+1 形式）
- [x] `tiling.cpp` 的 cumulated kv 读取分支只在 `FwdLayout::TND` 时执行，且未新增 tiling 字段

### v3 Python 层

- [x] 存在 `_transposed_bnsd_view()`：`is_contiguous()` 短路 + stride 三条件严格匹配
- [x] 存在 `_detect_layout()`：仅 stride 命中（q/k/v 全部命中才判定）时选 BNSD，否则恒返回 `_LAYOUT_BSND`；
      返回 `(_LAYOUT_TND/_LAYOUT_BSND/_LAYOUT_BNSD, q, k, v, out_transposed)`
- [x] stride 命中时零拷贝还原为连续 BNSD（`t.transpose(1, 2)`），不产生 `.contiguous()` 拷贝
- [x] 混合输入（部分命中）整体按 BSND 处理，数值仍正确
- [x] 三个公开入口（`flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache`）**都没有** `bnsd` 参数
- [x] 传入 `bnsd=` 关键字时由 Python 抛 `TypeError`（已验证）
- [x] `_flash_attn_forward` 及其 fake 用底层 `layout: int` 形参把探测结果透传给 `fwd`，fake 形状推导正确
- [x] `_training_forward` / `FlashAttnFunc` / `FlashAttnVarlenFunc` 记录 `ctx.layout`，
      并在 backward 首行拒绝 `_LAYOUT_BNSD`（不再有 `ctx.bnsd`）
- [x] 返回前按 `out_transposed` 把 `out` 转置回调用方传入的 q 形状
- [x] `flash_attn_with_kvcache` 支持 Q 与非 paged rank-4 `k_cache`/`v_cache` 为 BNSD；paged/rank-3 不解读
- [x] BNSD 前向在需要 grads 时 backward 抛出明确 `NotImplementedError`（已验证）
- [x] 未启用 BNSD 时既有 BSND/TND 调用的行为与形状不变

### v4 后端

- [x] v4 的 `Format`、kernel 入口参数、`FwdLaunchArgs`、宏实参与 v3 同构且一致
- [x] v4 kernel 步长/偏移/LSE/输出写回支持 BNSD
- [x] v4 host 按 `int64_t layout` 校验与透传，且公开 API 未新增布局参数
- [x] v4 `FlashAttnVarlenFunc` 在 `cu_seqlens_q is not None` 时为 `_LAYOUT_TND`，否则为 `_LAYOUT_BSND`
- [x] v4 未新增 `flash_attn_with_kvcache`
- [x] v4 的 autogen TU 数由 4 增至 6（与 v3 一致）
- [x] v4 三个 autograd 的 backward 返回元组个数已随 forward 入参个数同步调整

### 构建与验证

- [x] 使用 conda `ascend` + `~/miniconda3/Ascend/cann/set_env.sh`（CANN 9.1.0）以 `FLASH_ATTN_BUILD_NPU=950` 编译成功
- [x] 未使用 910 目标，未使用 `/usr/local/Ascend` 的 CANN 9.2.0-beta.2
- [x] 两个动态库 `flash_attn_npu_3_950` 与 `flash_attn_npu_4_950` 均链接成功
- [x] v3/v4 BSND 回归：与参考实现数值一致（fp16/bf16 精度内）
- [x] v3/v4 stride 零拷贝 BNSD：结果与 BSND 路径逐元素相等，且输出形状 == 传入 q 形状
- [x] v3/v4 连续 rank-4 输入（未命中 stride）按 BSND 解读，行为与既有 BSND 路径一致
- [x] v3 `flash_attn_with_kvcache` 在 stride BNSD 且 `cache_seqlens < S_alloc` 时数值正确
- [x] causal mask 与 LSE 在 BNSD 下正确
- [x] 公开 API 拒绝 `bnsd=`；BNSD backward 按预期报错
- [x] v3 BSND 回归套件 `tests/test_flash_attn_npu_v3.py` 结果确认：345 passed / 99 skipped / 0 failed
- [x] v4 BSND 回归套件 `tests/test_flash_attn_npu_v4.py` 结果确认：320 passed / 4 skipped / 0 failed
- [x] BNSD 用例已落进仓库测试目录：`docs/tests/test_flash_attn_npu_v3_bnsd.py`（8 passed）与
      `docs/tests/test_flash_attn_npu_v4_bnsd.py`（6 passed），覆盖 stride 零拷贝路径、连续 rank-4 回退 BSND、
      causal、LSE 布局、GQA 多头组、forward-only 拒绝、BNSD kvcache 分配平面长度
- [x] BNSD 支持 `scheduler_metadata` 后：实测 BNSD 调用 `get_scheduler_metadata` 恰好 1 次且尺寸取 BSND 语义；
      v3/v4 metadata 套件（15 passed / 80 skipped、45 passed / 6 skipped）与主回归（345/99、320/4）全绿

### 去 `bnsd` 重构（Task 8）

- [x] 全仓 `grep bnsd` 在 `csrc/` 与 `flash_attn_npu_*/` 下仅剩 `_transposed_bnsd_view` 辅助函数名与
      `autogen/generate_kernels.py` 的 `("bnsd", ...)` 生成键，无任何 `bnsd` 变量或形参
- [x] 重构后全量重编（`FLASH_ATTN_BUILD_NPU=950`）通过（0 error，两个 `.so` 链接成功）
- [x] 重构后 v3 回归 `tests/test_flash_attn_npu_v3.py`（345 passed / 99 skipped / 0 failed）
      与 v4 回归 `tests/test_flash_attn_npu_v4.py`（320 passed / 4 skipped / 0 failed）无 failed
- [x] 重构后 `docs/enable-qkv-bnsd/test_kv_seqlen_allocated.py`（BSND + BNSD 两用例）通过（2 passed）
- [x] `docs/` 与 `.trae/specs/` 的 spec / tasks / checklist / implementation 与最终实现一致（TU 4→6、
      无 `params.bnsd`、`launch_fwd_impl<DType, Format>`、`_detect_bnsd`→`_detect_layout`、`layout: int` 边界）

## 910

### 编译期布局契约

- [x] 910 三份 `kernel_common.hpp`（v2 `flash_attn_npu/`、v3 `flash_attn_npu_3/`、v4 `flash_attn_npu_4/`）的 `inputLayout` 枚举均含 `BSND = 0` / `TND = 1` / `BNSD = 2`；其编号与 Python 侧 `_LAYOUT_*`（`0=TND` / `1=BSND` / `2=BNSD`）**不同**，host 侧必须显式映射，不可 `static_cast` 互转
- [x] 三份 `FAIKernelParams` 均新增 `int64_t kvSeqlenAllocated`，构造参数同步；BSND / TND 路径下填 0，BNSD 路径下等于 KV 的 `S_alloc`
- [x] v2 的 `fag_layout.hpp` 与 `fag_common/common_header.h` 中的 FAG 布局常量**未改动**（反向仍只有 BSND / TND 两值）

### dispatch 与 autogen

- [x] 三个后端的 `launch_fwd` / `launch_fwd_dtype` / `launch_fwd_impl` 模板参数由 `bool IS_TND` 改为 `inputLayout kLayout`（v2 的 `fwd_kernel_launch.hpp` 同步）
- [x] 三个后端 `FwdLaunchArgs` 均新增 `inputLayout layout` 与 `int64_t kv_seqlen_allocated`，`FWD_LAUNCH` 宏把二者传入 kernel 例化
- [x] v3 `autogen/generate_kernels.py`：fwd `LAYOUTS` 3 项、**bwd `LAYOUTS` 仍为 2 项**；重新生成后 fwd TU 数为 6、bwd TU 数为 4
- [x] v4 `autogen/generate_kernels.py`：fwd `LAYOUTS` 3 项，fwd TU 数为 6
- [x] v2 `autogen/generate_kernels.py`：fwd `LAYOUTS` 3 项 → `fwd_dispatch` TU 6 个、`fwd_combo` TU 48 个；`fag_general_dispatch` / `varlen_bwd_dispatch` 列表仍为 2 项
- [x] `setup.py` 未改动（仍用 glob 收集 `autogen/*.cpp`），且三份 `.so` 名称与 arch（`dav-2201`）不变

### 前向 kernel 寻址（三后端逐条核对）

- [x] `kLayout == BSND` 与 `kLayout == TND` 时，`strideQ/strideO/strideK/strideV`、`qHeadStride/oHeadStride/kvHeadStrideK/kvHeadStrideV`、`qBOffset/kBOffset/vBOffset/oBOffset`、`gmOffsetQ/K/V/O`、KV 实际长度、LSE offset 的取值与改动前**逐项相同**（代码对照 + 数值回归双重确认）
- [x] `kLayout == BNSD` 时 `strideQ == embed`、`qHeadStride == maxQSeqlen*embed`（O 用 `embedV`）、`kvHeadStrideK == kvSeqlenAllocated*embed`（V 用 `embedV`）、batch 基址为 `BIdx*qHeads*qHeadStride`
- [x] `kvSeqlenAllocated` 来自 `k.size(2)`（非 `maxKvSeqlen`），KV 实际长度仍取 per-batch 的 `gActualKvseqlen.GetValue(BIdx)`，未误用 TND 的 cu 差分
- [x] `qk_matmul.hpp` 中把 head 平面间距写死为常量 `qHeads*embed` 的位置（`loadQGM` 的 `LayoutA denseSrc(rowNum, embed, qHeads*embed)` 等）已按布局推导；L1 目的布局未变
- [x] `init_outputs.hpp` / `rescale_o.hpp` 的 per-head 目标地址改用 `oHeadStride`；BSND 下 `oHeadStride == embedV` 保持等价
- [x] LSE 在 BNSD 下仍为 `(B,H,S)`，`lseBOffset` / `lseHeadStride` 取值与 BSND 相同，且 kernel 中**没有**从 `qBOffset / strideQ` 反推 batch 的写法
- [x] `combine_scale.hpp` 未为 BNSD 新增语义（FlashDecode 要求 paged KV，与 BNSD 天然互斥）

### host 侧

- [x] v3 / v4 `flash_api.cpp::mha_fwd` 新增 `int64_t layout` 形参并注册进 `PYBIND11_MODULE`；原 `is_varlen_q` 的二分派改为按 `layout` 三分派
- [x] v2 `mha_fwd` 与 `mha_fwd_kvcache` 新增 `int64_t layout` 形参并注册进 `PYBIND11_MODULE`；`mha_varlen_fwd` / packed 入口不参与 BNSD
- [x] host 校验 `layout ∈ {0,1,2}`、`(layout == TND) ⟺ cu_seqlens_q 非空`、BNSD 要求 q/k/v 均 rank-4、非 paged、`k_new/v_new/q_v` 为空
- [x] BNSD 下派生量正确：`num_heads = q.size(1)`、`seqlen_q = q.size(2)`、`num_heads_k = k.size(1)`、`kv_seqlen_allocated = k.size(2)`；`max(cache_seqlens) > S_alloc` 时报错
- [x] `out` 仍由 `torch::empty_like(q)` 分配，BNSD 下自然得到 `(B,H,S,D)`；Python 侧按 `out_transposed` 转置回调用方形状
- [x] `get_scheduler_metadata` 与 metadata 复用路径尺寸一律取探测**之前**的 BSND 形状；metadata 分支的 KV 上界在 BNSD 下读 `k.size(2)`

### Python 侧

- [x] v2 / v3 / v4 三份接口均定义 `_LAYOUT_TND = 0` / `_LAYOUT_BSND = 1` / `_LAYOUT_BNSD = 2`（与 host 侧 `int64_t layout` 取值一致；与 kernel 侧 `inputLayout` 编号不同，见上）
- [x] `_transposed_bnsd_view(t)`：要求 rank-4、非连续、`stride(3)==1 && stride(1)==shape[3] && stride(2)==shape[1]*shape[3]`，否则返回 `None`
- [x] `_detect_layout` 仅在 Q/K/V **三者全部命中**时返回 `(_LAYOUT_BNSD, 连续 BNSD 三元组, True)`，否则 `(_LAYOUT_BSND, 原样操作数, False)`；提供 `page_table` / `cu_seqlens_k` 时不做 BNSD 解读
- [x] 探测在既有 `maybe_contiguous` 逻辑**之前**、在原始输入上进行；BNSD 路径还原出的连续 BNSD 张量对 `maybe_contiguous` 是 no-op（不产生额外拷贝）
- [x] 自定义算子 `_flash_attn_forward` 及其 fake、以及 pybind 底层入口（`flash_attn_npu.fwd` / `.fwd_kvcache`、`flash_attn_npu_3.fwd`、`flash_attn_npu_4.fwd`）均以 `layout: int` 承接探测结果；fake 按传入操作数形状推导 `out` / LSE
- [x] autograd `forward` 记录 `ctx.layout`，`backward` 首行对 `_LAYOUT_BNSD` 抛 `NotImplementedError`（含「910 后端 BNSD 仅支持前向」字样）
- [x] v3 `flash_attn_varlen_func` 恒传 `_LAYOUT_TND`；v4 `FlashAttnVarlenFunc` 为 `cu_seqlens_q is not None ? _LAYOUT_TND : _LAYOUT_BSND`
- [x] v2 接口中 `layout=q.layout`（tensor layout，即 `torch.strided`）的既有用法未被破坏

### 功能验证

- [x] v3 `docs/tests/test_flash_attn_npu_v3_bnsd_910.py`（910 门槛）通过：stride 零拷贝路径、与等价 BSND 逐元素相等、连续 rank-4 回退 BSND、causal、LSE 形状与数值、forward-only 拒绝、`S_alloc > cache_seqlens`、GQA `H_KV >= 2`
- [x] v4 `docs/tests/test_flash_attn_npu_v4_bnsd_910.py`（910 门槛）通过
- [x] v2 `docs/tests/test_flash_attn_npu_v2_bnsd_910.py`（910 门槛）通过，且 packed / varlen 入口未被解释为 BNSD
- [x] BNSD 路径的 `out` 形状等于调用方传入的 `q` 形状；LSE 形状符合各入口既有约定
- [x] 与 950 既有 BNSD 用例（`docs/tests/test_flash_attn_npu_v{3,4}_bnsd.py` 中 950 门槛部分）互不干扰，950 上不误跑 910 用例

### 回归与静态核对

- [x] `source /data/.sysvol/cann/cann-9.3.0/set_env.sh` + `FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_BUILD_VERSION=all python setup.py install` 全量编译 0 error，三个 `.so` 就地就位
- [x] TU 数核对：v3 fwd 6 + bwd 4、v4 fwd 6、v2 fwd_dispatch 6 + fwd_combo 48、fag/bwd 不变
- [x] `tests/test_flash_attn_npu_v2.py` / `v3.py` / `v4.py` 全量运行 0 failed，BSND / TND 数值与改动前一致（记录 passed / skipped 数字）
- [x] `grep -rn '\bbnsd\b' csrc/ascend910/` 仅剩 `autogen/generate_kernels.py` 的 `("bnsd", ...)` 生成键，以及 v2 `fwd_dispatch_impl.hpp` 的 `FWD_SELECT_COMBO_8(fp16/bf16, bnsd)` **编译期宏 token**（无任何运行期 `bnsd` 变量或形参）
- [x] `grep -rn 'BNSD' csrc/ascend910/` 只出现在三份 `kernel_common.hpp` 枚举、`flash_api.cpp`（host 布局码与校验）、`fwd_dispatch*.hpp` / `fwd_kernel_launch.hpp`（编译期分派令牌）、`mha_fwd_kvcache.cpp`（按布局推导寻址）、`qk_matmul.hpp` / `init_outputs.hpp` / `rescale_o.hpp` 的 `if constexpr` 布局分支，以及生成的 TU 中；`combine_scale.hpp` 无 BNSD 语义
