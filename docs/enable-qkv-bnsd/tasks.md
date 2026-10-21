# Tasks

> 本清单合并两代任务，两份 tasks 分别对应：
> - **950 侧**：950 提交 `a514357`（分支 `950bnsd`），覆盖 `csrc/ascend950/` 的 v3 / v4；
> - **910 侧**：把同一设计移植到 910（`csrc/ascend910/` 的 v2 / v3 / v4）的 BNSD 移植工作。

## 950 后端

- [x] Task 1: v3 内核入口与布局参数打通
  - [x] SubTask 1.1: `csrc/ascend950/flash_attn_npu_3/kernel_common.hpp`：`Format` 增加 `BNSD = 2`；
        `FAIKernelParams` 增加 `int64_t kvSeqlenAllocated` 字段并同步构造函数（无布局布尔量）
  - [x] SubTask 1.2: `csrc/ascend950/flash_attn_npu_3/fwd_kernel.cpp`：`FAInfer` 与 `FAInferDn` 的
        `CATLASS_GLOBAL` 签名增加该参数并传入 `FAIKernelParams` 构造；布局由模板参数
        `Format qFormat, Format kvFormat` 承载
  - [x] SubTask 1.3: `csrc/ascend950/flash_attn_npu_3/fwd_dispatch.hpp`：`FwdLaunchArgs` 增加 `layout`（`Format`）
        与 `kv_seqlen_allocated`；`launch_fwd` 按 `a.layout` 做 `(fp16, bf16) × (TND, BSND, BNSD)` 6 路分派
  - [x] SubTask 1.4: `csrc/ascend950/flash_attn_npu_3/fwd_dispatch_impl.hpp`：模板参数 `<DType, bool IS_TND>` 改为
        `<DType, Format kLayout>`，扩展 `FWD_KERNEL_LAUNCH` 宏实参列表；
        `autogen/generate_kernels.py` 的 `LAYOUTS` 增加 `bnsd` 行，TU 数 4 → 6
  - [x] SubTask 1.5: `csrc/ascend950/flash_attn_npu_3/fd_combine.hpp`：`FAIKernelParams` 聚合初始化补入
        `kvSeqlenAllocated = 0`（FlashDecode 与 BNSD 互斥）

- [x] Task 2: v3 kernel 步长与偏移改造
  - [x] SubTask 2.1: `fwd_kernel.cpp`：按 `if constexpr (qFormat == Format::BNSD)` 推导 `strideQ`/`qHeadStride`、
        `strideO`/`oHeadStride`；按 `if constexpr (kvFormat == Format::BNSD)` 推导 `strideK`/`strideV`、
        `kvHeadStrideK/V`、K/V 的 batch 累加量
  - [x] SubTask 2.2: `fwd_kernel.cpp`：`gmOffsetQ` / `gmOffsetO` / `gmOffsetK` / `gmOffsetV` 使用新的 head-plane 步长；
        paged 分支逻辑不变
  - [x] SubTask 2.3: `fwd_kernel.cpp`：LSE offset 增加 BNSD 分支（用循环内 `curBatch`），且 offset 含完整 `S` 步长
        （避免歧义形状下越界写入其他 batch 的 LSE），BSND/TND 公式保持不变
  - [x] SubTask 2.4: `fwd_kernel.cpp`：`kvNumTokens` 在 `kvFormat == Format::BNSD` 时取 per-batch 值
        （`gActualKvseqlen.GetValue(batch_ - 1)`，屏蔽 TND 的 cu 差分）；删除原
        `if constexpr (kvFormat == Format::TND) { if (!params.bnsd) {...} }` 死守卫
  - [x] SubTask 2.5: `qk_matmul.hpp`：`loadQGM`（非 DN）增加运行时 head-plane 源间距参数并用于 Nd2Nz 源 stride；
        `qk_matmul_DN.hpp` 签名不动，调用点按 `IsDN` 分支分别传参
  - [x] SubTask 2.6: `init_outputs.hpp` 与 `rescale_o.hpp`（含 `ScatterGroupedOutput` 调用链）增加 `oHeadStride`
        运行时参数；BSND 下传 `embedV_`，BNSD 下传 `maxQSeqlen * embedV_`
  - [x] SubTask 2.7: 核对 `qShapeCol` / `oShapeCol` / `kShapeCol` / `vShapeCol`（row pitch 语义）在 BNSD 下正确

- [x] Task 3: v3 host 探测、校验与透传
  - [x] SubTask 3.1: `csrc/ascend950/flash_attn_npu_3/mha_fwd.cpp`：`mha_fwd` 底层入口参数为 `int64_t layout`
        （0 = TND、1 = BSND、2 = BNSD）
  - [x] SubTask 3.2: host 只做取值范围校验与不变式校验 `layout == FwdLayout::TND ⟺ cu_seqlens_q is not None`；
        布局探测（stride 严格匹配）已上移到 Python 侧，host 不再重复探测；
        按 `layout` 修正 `num_heads` / `seqlen_q` / `num_heads_k` 并与 `head_size` 校验
  - [x] SubTask 3.3: 计算 `kv_seqlen_allocated`（BNSD KV 的 `k.size(1)`）并校验 `max(seqused_k) <= S_alloc`、
        k/v 的 (batch, heads, seqlen) 维一致；BNSD KV 时按需生成 per-batch kv 列表（不做 cumsum）
  - [x] SubTask 3.4: `csrc/ascend950/flash_attn_npu_3/tiling_from_tensors.hpp`：`fill_inference_context` 的
        `is_tnd` 入参改为 `FwdLayout layout`，BNSD 下 kv 列表保持 per-batch 且拒绝 batch+1 形式
  - [x] SubTask 3.5: `csrc/ascend950/flash_attn_npu_3/tiling.cpp`：`FAInferContext` 的 `isTnd` 改为 `FwdLayout layout`，
        TND 下读取 cumulated kv 长度的两处分支只在 `FwdLayout::TND` 时执行；**不新增 `FAInferTilingData` 字段**
  - [x] SubTask 3.6: DN 快路径对 BNSD 同样开放（`enableDN` 不再排除 BNSD；DN 版 `loadQGM` 用 `qHeadStride`
        作 `srcNdMatrixStride`）、取消 FlashDecode 候选；metadata 分支的 KV 上界按布局取序列维；
        填充 `FwdLaunchArgs` 的 `layout`（`static_cast<Format>(layout_enum)`）与 `kv_seqlen_allocated`；
        确认输出按 BNSD 平面分配
  - [x] SubTask 3.7: `csrc/ascend950/flash_attn_npu_3/flash_api.cpp`：`m.def("fwd", &mha_fwd, ...)` 随新签名更新

- [x] Task 4: v3 Python 接口层
  - [x] SubTask 4.1: `flash_attn_npu_3/flash_attn_npu_interface_950.py` 新增 `_transposed_bnsd_view()` 与
        `_detect_layout()`：stride 严格匹配（要求 q/k/v 全部命中，`is_contiguous()` 短路），
        命中则零拷贝 `transpose(1, 2)` 还原连续 BNSD；否则原样按 BSND 返回；
        返回与 C++ 取值一致的布局码 `_LAYOUT_TND/BSND/BNSD`
  - [x] SubTask 4.2: `_flash_attn_forward` 与 `_flash_attn_forward_fake` 用底层 `layout: int` 形参透传，
        fake 按 `layout == _LAYOUT_BNSD` 推导 out/LSE 形状
  - [x] SubTask 4.3: `FlashAttnFunc` / `FlashAttnVarlenFunc`：用探测结果计算 `seqused_k`，BNSD 复用同一
        scheduler_metadata（尺寸取检测前的 BSND 形状），记录 `ctx.layout`，backward 首行拒绝 BNSD
  - [x] SubTask 4.4: `flash_attn_func` / `flash_attn_varlen_func` **不新增任何布局参数**；
        在返回前按 `out_transposed` 把 `out` 转置回调用方传入的 q 形状
  - [x] SubTask 4.5: `flash_attn_with_kvcache` **不新增任何布局参数**：支持 Q 为 BNSD 转置视图、
        非 paged rank-4 `k_cache`/`v_cache` 为 BNSD；paged 与 rank-3 cache 不做 BNSD 解读；
        BNSD 复用自动 metadata（尺寸取检测前的 BSND 形状）、校验 KV 平面长度；更新 docstring（含「out 形状等于传入的 q 形状」）

- [x] Task 5: v4 后端同构改造（`flash_attn_func` / `flash_attn_varlen_func`，无 kvcache）
  - [x] SubTask 5.1: 阅读 `csrc/ascend950/flash_attn_npu_4/` 的 `kernel_common.hpp`、`fwd_kernel.cpp`、
        `fwd_dispatch.hpp`、`fwd_dispatch_impl.hpp`，确认与 v3 同构
  - [x] SubTask 5.2: 按 Task 1 方式增加 `Format::BNSD`、kernel 入口参数、`FwdLaunchArgs` 字段、宏实参，
        `launch_fwd_impl` 模板参数改为 `Format kLayout`，`autogen/` TU 数 4 → 6
  - [x] SubTask 5.3: 按 Task 2 方式改造 v4 kernel 的步长/偏移/LSE/`loadQGM`/输出写回
  - [x] SubTask 5.4: 按 Task 3 方式改造 v4 的 `mha_fwd.cpp` 校验与透传（v4 无 kvcache）
  - [x] SubTask 5.5: 按 Task 4.1/4.2/4.4 方式改造 `flash_attn_npu_4/flash_attn_npu_interface_950.py`
        （v4 `FlashAttnVarlenFunc` 在 `cu_seqlens_q is not None` 时为 `_LAYOUT_TND`，否则 `_LAYOUT_BSND`；
        `FlashAttnFunc.forward` 返回前按 `out_transposed` 转置 `out`）

- [x] Task 6: 编译 950 双后端（conda `ascend` + CANN `~/miniconda3/Ascend/cann/set_env.sh`）
  - [x] SubTask 6.1: 加载 CANN 9.1.0（禁止 `/usr/local/Ascend` 的 9.2.0-beta.2），以 `FLASH_ATTN_BUILD_NPU=950`
        编译 `flash_attn_npu_3_950` 与 `flash_attn_npu_4_950`（禁止 910 目标）
  - [x] SubTask 6.2: 编译失败时查看完整日志（不得用 tail 截断），修复后重编
  - [x] SubTask 6.3: 若修复发生在编译中途，`touch` 受影响源文件触发强制重编

- [x] Task 7: BNSD 前向自测
  - [x] SubTask 7.1: 用 `sg HwHiAiUser` 刷新补充组快照后运行自测
  - [x] SubTask 7.2: v3 `flash_attn_func`：BSND 回归、stride 零拷贝 BNSD、causal、数值与 out/LSE 形状
  - [x] SubTask 7.3: v3 `flash_attn_with_kvcache`：BSND 回归 + stride 零拷贝 BNSD（`cache_seqlens < S_alloc`）
  - [x] SubTask 7.4: 公开 API 拒绝 `bnsd=`、BNSD backward 报错
  - [x] SubTask 7.5: v4 `flash_attn_func` 的 BNSD 用例与 BSND 回归
  - [ ] SubTask 7.6: 把 BNSD 用例补充进 `tests/test_flash_attn_npu_v3.py` 与 `tests/test_flash_attn_npu_v4.py`
  - [x] SubTask 7.7: 运行 v3 BSND 回归套件 `tests/test_flash_attn_npu_v3.py`，确认未破坏原有功能

- [x] Task 8: 去运行期 `bnsd` 变量，改为按 `Format` 决定业务分支
  - [x] SubTask 8.1: `kernel_common.hpp`（v3/v4）：删除 `FAIKernelParams::bnsd` 成员与构造形参，保留 `kvSeqlenAllocated`
  - [x] SubTask 8.2: `fwd_dispatch.hpp`（v3/v4）：`FwdLaunchArgs` 的 `bool bnsd` 改为 `Format layout`；
        `launch_fwd` 按 `layout` 做 3 `Format` × 2 dtype = 6 路分派
  - [x] SubTask 8.3: `fwd_dispatch_impl.hpp`（v3/v4）：模板参数 `<DType, bool IS_TND>` → `<DType, Format kLayout>`；
        `kUseDnFastPath` 追加 `&& (kLayout != Format::BNSD)` 剪掉 `FAInferDn<BNSD>`；宏实参删 `a.bnsd`
  - [x] SubTask 8.4: `autogen/generate_kernels.py`（v3/v4）：新增 `("bnsd", "BNSD", "Format::BNSD")` 并重新生成 6 个 TU
  - [x] SubTask 8.5: `fwd_kernel.cpp`（v3/v4）：所有 `params.bnsd` 判断改为 `if constexpr`；
        `GetKvSeqLen` 删 `bool bnsd` 形参；删两处 `if (!params.bnsd)` 死守卫
  - [x] SubTask 8.6: `fd_combine.hpp`（v3/v4）：聚合初始化的 `false` 默认实参改为 `0`
  - [x] SubTask 8.7: `mha_fwd.cpp`（v3/v4）：入参 `std::optional<bool> bnsd_` → `int64_t layout`，删除 host 重复探测
  - [x] SubTask 8.8: Python（v3/v4）`flash_attn_npu_interface_950.py`：新增 `_LAYOUT_*` 常量，
        `_detect_bnsd` → `_detect_layout`，`_flash_attn_forward[_fake]`/`_training_forward`/`FlashAttnFunc`/
        `FlashAttnVarlenFunc`/`flash_attn_with_kvcache` 全部改用 `layout`，`ctx.bnsd` → `ctx.layout`
  - [x] SubTask 8.9: `docs/enable-qkv-bnsd/test_kv_seqlen_allocated.py` 同步 `_detect_layout` 与 `_LAYOUT_BNSD`
  - [x] SubTask 8.10: 全量重编（CANN 9.1.0 + `FLASH_ATTN_BUILD_NPU=950`）并跑 v3/v4 回归 + kv_seqlen_allocated 用例

### Task Dependencies（950）

- Task 2 依赖 Task 1
- Task 3 依赖 Task 2（`FwdLaunchArgs` 与 kernel 参数形态）
- Task 4 依赖 Task 3（`fwd` pybind 签名）
- Task 5 依赖 Task 1~4 的设计定稿（可与 Task 3/4 并行推进，但需先完成 5.1 的结构核对）
- Task 6 依赖 Task 4、Task 5
- Task 7 依赖 Task 6
- Task 8 依赖 Task 6（重编验证），与 Task 8.1~8.9 的源码改动为同一批次
- Task 1、Task 5.1 可并行

## 910 后端

- [x] Task 1: 三份 Python 接口接入 BNSD 布局探测（v2 / v3 / v4）
  - [x] SubTask 1.1: 在 `flash_attn_npu_3/flash_attn_npu_interface.py` 增加模块常量
        `_LAYOUT_TND = 0` / `_LAYOUT_BSND = 1` / `_LAYOUT_BNSD = 2`、`_transposed_bnsd_view(t)`、
        `_detect_layout(q, k, v, page_table, cu_seqlens_k) -> (layout, q, k, v, out_transposed)`
  - [x] SubTask 1.2: v3 自定义算子 `flash_attn_npu_3::_flash_attn_forward` 及其 fake 增加 `layout: int`
        形参（fake 按传入操作数形状推导 `out` / LSE，不依赖 BSND 读法），pybind 调用处
        `flash_attn_npu_3.fwd(...)` 同步透传
  - [x] SubTask 1.3: v3 调用点接入：`FlashAttnFunc.forward`（含 `flash_attn_func`）与
        `flash_attn_with_kvcache` 调 `_detect_layout`（with_kvcache 传入 `page_table`）、
        `ctx.layout = layout`、返回前按 `out_transposed` 转置 `out`、backward 首行拒绝 `_LAYOUT_BNSD`；
        `FlashAttnVarlenFunc` / `flash_attn_varlen_func` 恒为 `_LAYOUT_TND`；
        BNSD 下 `k_new`/`v_new`/`q_v` 非空时不走 BNSD（留给 host 报错或在此提前报错）
  - [x] SubTask 1.4: `flash_attn_npu_4/flash_attn_npu_interface.py` 同上（v4 无 `with_kvcache`、
        无 `k_new/v_new`；`FlashAttnVarlenFunc` 为 `cu_seqlens_q is not None ? TND : BSND`）
  - [x] SubTask 1.5: `flash_attn_npu/flash_attn_npu_interface.py` 同上，仅覆盖
        `flash_attn_func`（→ `flash_attn_npu.fwd`）与 `flash_attn_with_kvcache`（→ `flash_attn_npu.fwd_kvcache`）；
        varlen / packed 变体不参与 BNSD；确认 `layout=q.layout`（tensor layout）不受影响
  - [x] SubTask 1.6: 三个接口的自定义算子 `get_scheduler_metadata` 路径与 metadata 复用保持尺寸取
        探测前的 BSND 形状；公开签名（`flash_attn_func` / `flash_attn_varlen_func` /
        `flash_attn_with_kvcache` / packed 变体）**不新增布局参数**

- [x] Task 2: 910 v3 前向 kernel 支持 BNSD（`csrc/ascend910/flash_attn_npu_3/`）
  - [x] SubTask 2.1: `kernel_common.hpp`：`inputLayout` 增加 `BNSD = 2`；`FAIKernelParams` 增加
        `int64_t kvSeqlenAllocated`（构造参数同步，其余布局填 0）
  - [x] SubTask 2.2: `fwd_dispatch.hpp` / `fwd_dispatch_impl.hpp`：`launch_fwd<kLayout>`、
        `launch_fwd_dtype<DType, inputLayout kLayout>`、`launch_fwd_impl` 同改；
        `FwdLaunchArgs` 增加 `inputLayout layout` 与 `int64_t kv_seqlen_allocated`；
        `FWD_LAUNCH` 宏把 `kLayout` 与 `kvSeqlenAllocated` 传入 kernel 例化
  - [x] SubTask 2.3: `autogen/generate_kernels.py`：`LAYOUTS` 条目携带 fwd 布局令牌，
        **fwd 列表 3 项（bnsd 新增）、bwd 列表保持 2 项**；重新生成后 fwd TU 4→6、bwd TU 4 不变；
        `setup.py` 不需改动
  - [x] SubTask 2.4: `mha_fwd_kvcache.cpp`（前向 kernel）：按 spec 的「前向 kernel 按布局推导寻址」
        改造 `strideQ/strideO/strideK/strideV`、新增 `qHeadStride/oHeadStride/kvHeadStrideK/kvHeadStrideV`、
        batch 基址 `qBOffset/kBOffset/vBOffset/oBOffset`、`gmOffsetQ/K/V/O`、KV 实际长度与 LSE offset
  - [x] SubTask 2.5: `qk_matmul.hpp`（`loadQGM` 的 head 平面间距 `qHeads*embed` 参数化）、
        `init_outputs.hpp` / `rescale_o.hpp`（per-head 目标地址改用 `oHeadStride`）；
        `combine_scale.hpp` 仅核对（FD 与 BNSD 互斥）
  - [x] SubTask 2.6: `flash_api.cpp`（host `mha_fwd`）：新增 `int64_t layout` 形参并注册进 pybind；
        校验 `layout ∈ {0,1,2}`、`(layout == TND) ⟺ cu_seqlens_q 非空`、BNSD 要求 rank-4 且非 paged、
        `k_new/v_new/q_v` 为空；按布局派生 `num_heads/seqlen_q/num_heads_k`、
        `kv_seqlen_allocated = k.size(2)`、`max(cache_seqlens) <= S_alloc`；
        把 `launch_fwd<true>/<false>` 改为按 `layout` 三分派；metadata 分支 KV 上界按布局取序列维
  - [x] SubTask 2.7: 编译通过（`FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_BUILD_VERSION=v3`）并跑
        `tests/test_flash_attn_npu_v3.py` / `test_flash_attn_npu_v3_metadata.py`，确认 BSND/TND 零回归

- [x] Task 3: 910 v4 前向 kernel 支持 BNSD（`csrc/ascend910/flash_attn_npu_4/`）
  - [x] SubTask 3.1: `kernel_common.hpp`：`inputLayout` 增加 `BNSD = 2`；`FAIKernelParams` 增加
        `int64_t kvSeqlenAllocated`
  - [x] SubTask 3.2: `fwd_dispatch.hpp` / `fwd_dispatch_impl.hpp`：同 Task 2.2（v4 的 `FwdLaunchArgs`
        含 `blockTableStride` / `tilingDevice` / `runtimeDevice`）
  - [x] SubTask 3.3: `autogen/generate_kernels.py`：fwd `LAYOUTS` 3 项，重新生成 fwd TU 4→6
        （脚本只产 fwd；bwd TU 不动）
  - [x] SubTask 3.4: `mha_fwd_kvcache.cpp`：同 Task 2.4 的步长/偏移/LSE 改造
        （v4 无 `k_new/v_new`，batch 基址含 `blockBOffset = BIdx*blockTableStride`）
  - [x] SubTask 3.5: `qk_matmul.hpp` / `init_outputs.hpp` / `rescale_o.hpp`：同 Task 2.5
  - [x] SubTask 3.6: `flash_api.cpp`（host `mha_fwd`）：同 Task 2.6 的 `layout` 形参、校验与派生
        （v4 无 append-KV / rotary；`gather_kv_indices` 与 BNSD 互斥关系需核对）
  - [x] SubTask 3.7: 编译通过（`FLASH_ATTN_BUILD_VERSION=v4`）并跑 `tests/test_flash_attn_npu_v4.py` /
        `test_flash_attn_npu_v4_metadata.py`，确认 BSND/TND 零回归

- [x] Task 4: 910 v2 前向 kernel 支持 BNSD（`csrc/ascend910/flash_attn_npu/`）
  - [x] SubTask 4.1: `kernel_common.hpp`：`inputLayout` 增加 `BNSD = 2`；`FAIKernelParams` 增加
        `int64_t kvSeqlenAllocated`；**`fag_layout.hpp` 与 `fag_common/common_header.h` 的 FAG 布局常量不改**
  - [x] SubTask 4.2: `fwd_dispatch.hpp` / `fwd_dispatch_impl.hpp` / `fwd_kernel_launch.hpp`：
        `launch_fwd_impl<DType, inputLayout kLayout>`，`FwdLaunchArgs` 增加 `layout` 与
        `kv_seqlen_allocated`；`fwd_combo` 的分派令牌由 `inputLayout::BNSD` 正确路由
  - [x] SubTask 4.3: `autogen/generate_kernels.py`：fwd `LAYOUTS` 3 项（`fwd_dispatch` 4→6、
        `fwd_combo` 32→48），fag / varlen_bwd 列表保持 2 项；重新生成并核对 TU 数
  - [x] SubTask 4.4: `mha_fwd_kvcache.cpp`（`FAInfer` 前向 kernel）：按布局改造 `strideQ/strideK/strideO`、
        head-plane 间距、per-batch 偏移（`prevQSeqlenSum` 分支）、`gmOffsetQ/K/V/O` 与 LSE offset
  - [x] SubTask 4.5: `qk_matmul.hpp`（`loadQGM` 内 `LayoutA denseSrc(rowNum, embed, qHeads*embed)` 与
        `getKVOffset` 的 `strideKV` 用法）、`init_outputs.hpp` / `rescale_o.hpp` / `combine_scale.hpp`
        的 per-head 目标地址
  - [x] SubTask 4.6: `flash_api.cpp`：`mha_fwd` 与 `mha_fwd_kvcache` 新增 `int64_t layout` 形参并注册进
        pybind；BNSD 下 `num_heads = sizes[2]` 改为 `sizes[1]`、`seqlen_q = sizes[2]`、
        `seqlen_k/num_heads_k = k.size(2)/k.size(1)`、`kv_seqlen_allocated = k.size(2)`；
        rank-4 中间维 stride 校验（`k.dim()==4` 时 k/v 的 (batch, heads, seqlen) 一致）；
        `mha_bwd` 对 rank-4 BNSD 入口直接拒绝（BNSD 仅前向）
  - [x] SubTask 4.7: 编译通过（`FLASH_ATTN_BUILD_VERSION=v2`）并跑 `tests/test_flash_attn_npu_v2.py` /
        `test_flash_attn_npu_v2_metadata.py`，确认 BSND/TND 零回归

- [x] Task 5: 新增 910 BNSD 用例
  - [x] SubTask 5.1: 新增 `tests/test_flash_attn_npu_v3_bnsd.py`（910 门槛：设备名含 `Ascend910`；
        与既有 950 同名文件区分——若命名冲突则用 `_910` 后缀）。最终落库文件名带 `_910` 后缀：
        `tests/test_flash_attn_npu_v3_bnsd_910.py`；覆盖 stride 零拷贝路径与 BSND 逐元素相等、
        连续 rank-4 回退 BSND、causal、LSE 形状/数值、forward-only 拒绝、`flash_attn_with_kvcache` 的
        `S_alloc > cache_seqlens` 分配平面长度、GQA 多头组（`H_KV >= 2`，因 `H_KV == 1` 时转置视图会被
        `is_contiguous()` 短路而探测失效）
  - [x] SubTask 5.2: 新增 `tests/test_flash_attn_npu_v4_bnsd.py` 对应用例（v4 无 kvcache 入口）；
        最终落库为 `tests/test_flash_attn_npu_v4_bnsd_910.py`
  - [x] SubTask 5.3: 新增 `tests/test_flash_attn_npu_v2_bnsd.py` 对应用例
        （`flash_attn_func` 与 `flash_attn_with_kvcache`；packed / varlen 入口不做 BNSD 解读）；
        最终落库为 `tests/test_flash_attn_npu_v2_bnsd_910.py`
  - [x] SubTask 5.4: 三个用例文件在 910 上全部通过；同时确认既有 `tests/test_flash_attn_npu_v{2,3,4}.py`
        中的 950-only 用例与 BNSD 用例互不干扰

- [x] Task 6: 全量编译与回归验证
  - [x] SubTask 6.1: `source /data/.sysvol/cann/cann-9.3.0/set_env.sh` +
        `FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_BUILD_VERSION=all python setup.py install` 全量编译 0 error，
        三个 `.so`（`flash_attn_npu` / `flash_attn_npu_3` / `flash_attn_npu_4`）就地就位；
        核对 TU 数：v3 fwd 6 + bwd 4、v4 fwd 6、v2 fwd_dispatch 6 + fwd_combo 48 + fag/bwd 不变
  - [x] SubTask 6.2: 跑三个新增 BNSD 用例（全绿）与 `test_flash_attn_npu_v2/v3/v4.py` 全量
        （BSND/TND 与改动前一致，0 failed），记录 passed/skipped 数字
  - [x] SubTask 6.3: 静态核对：`grep -rn '\bbnsd\b' csrc/ascend910/` 仅剩 `autogen/generate_kernels.py`
        的 `("bnsd", ...)` 生成键（无任何运行期 `bnsd` 变量/形参）；
        `grep -rn 'BNSD' csrc/ascend910/` 只出现在 `kernel_common.hpp` 枚举、`qk_matmul/init_outputs/
        rescale_o/combine_scale` 的 `if constexpr` 与生成的 TU 中；
        `FAInferTilingData` / `FAInferStaticTilingData` / `FAInferRuntimeData` 字段集合未变、`fa_metadata.aicpu` 未改

- [x] Task 7: 修复 v2 接口因布局探测引入的非对齐 head_dim 回归
      （Task 6 回归暴露：`test_fa_func_ops` 中 `head_size` 为 59 / 111 的用例数值全错）
  - [x] SubTask 7.1: 定界根因——`FlashAttnFunc.forward` 中「padding 非对齐 head_dim」的代码块被挪到
        `batch_size, seqlen_q, num_heads, head_size = q.shape` 之后，导致传给
        `get_scheduler_metadata` 的 `head_size` 由 padding 后的 64 变成原始 59，AICPU 侧
        scheduler metadata 与实际 kernel 工作划分不一致
  - [x] SubTask 7.2: 修复——`_detect_layout` 提前到 padding 之前，padding 之后再按 `layout` 分支
        解包（BNSD 取 `(B,H,S,D)` 读法），使 `get_scheduler_metadata` 拿到 kernel 实际运行的
        （padding 后）head dim；BNSD 的 `batch_size/num_heads/seqlen_q/seqlen_k` 也随布局取对
  - [x] SubTask 7.3: 验证——最小复现用例通过；`tests/test_flash_attn_npu_v2.py -k "59 or 111"`
        38 passed；`tests/test_flash_attn_npu_v2_bnsd_910.py` 12 passed

- [x] Task 8: 修复 v3 前向 FlashDecode 分支未透传 `outputStride` / `oHeadStride` 导致的 GM 越界写
      （Task 6 回归暴露：`test_fa_kvcache_ops` 中 `[...-20000-59-...-TND-True--1--1-2.0-1-False]`
      崩溃 `ACL stream synchronize failed, error code:507015`，plog 报 MTE 非法 GM 地址）
  - [x] SubTask 8.1: 定界根因——`flash_api.cpp` 的 `flashDecodeFlag` 条件
        （`paged_KV && is_varlen_q && !is_local && maxQSeqlen*groupSize <= 128 && maxQSeqlen <= 16 &&
        maxKVSeqlen >= 1024 && (isLongSeq || isShortSeq)`）在该用例下为真，于是 kernel 走
        `rescale_o.hpp` 的 FD 分支；而 `fa_split.h` 的 `isSplitKV` 是**逐 task** 的
        （`coveredS2 > 0 && coveredS2 < curKSBlockNum`），FD kernel 内仍存在 `isSplitKV == false`
        的 task，这些 task 走 `CopyOToGm` 写完整 O。v3 `mha_fwd_kvcache.cpp` 的 FD 调用点
        （`flashDecodeFlag != 0U` 分支）**未传**新增的 `outputStride` / `oHeadStride`，两者取
        `operator()` 的默认值 0，`DataCopyExtParams(..., (outputStride - embed) * 2, ...)` 的
        gap 由 0 减 `embed` 下溢成巨值 → MTE 访存非法 GM 地址
  - [x] SubTask 8.2: 修复——v3 FD 调用点补传
        `static_cast<uint32_t>(strideO), static_cast<uint32_t>(oHeadStride)`；
        并去掉 v3 `rescale_o.hpp` `operator()` 中尾部形参的默认值（因 C++ 不允许「有默认值的形参
        后面跟无默认值形参」，实际做法是把 `splitParams` 起的整组尾部形参默认值一并去掉，两处调用点
        均已全显式传参；FD 分支中间 5 个参数按原默认值显式传入 `0, 0, 0, 0, 1`，语义不变）；
        同时核对 v2 / v4 的 FD 调用点（v2 已传 `oHeadStride`、v4 的 `oHeadStride` 为必填形参，
        均无需改动）
  - [x] SubTask 8.3: 验证——重编 v3 后单跑该用例 1 passed；`tests/test_flash_attn_npu_v3.py`
        全量由 443 failed 降为 3 failed（525 passed, 11 skipped），剩余 3 项为另一处回归，
        见 Task 9

- [x] Task 9: 修复 v3 host `layout`/`cu_seqlens_q` 一致性校验提前触发导致的错误文案回归
      （Task 8 修复后 v3 全量暴露：`TestVarlenValidation::test_lowlevel_varlen_validation`
      的 `native-q_2d` / `native-empty_batch` / `native-empty_tokens` 3 项 Regex 不匹配）
  - [x] SubTask 9.1: 定界根因——测试的 native 分支用
        `inspect.signature(api._flash_attn_forward_fake).bind(**inputs)` + `apply_defaults()`
        再直调 `flash_attn_npu_3.fwd`，fake 的 `layout` 取了默认值 `_LAYOUT_BSND(1)`，而输入带
        `cu_seqlens_q`；本轮新增在 `flash_api.cpp` L185 的
        `TORCH_CHECK((layout == 0) == cu_seqlens_q_.has_value(), ...)` 排在既有形状校验
        （L197 `q must be 3-dimensional`、L198 `batch size must be positive`、L199
        `total_q must be positive`）**之前**，于是新文案抢先抛出，原有错误信息被遮蔽
  - [x] SubTask 9.2: 修复——把该一致性 `TORCH_CHECK` 移到 L196-L200 的
        `if (cu_seqlens_q_.has_value())` 形状校验块**之后**（仍早于任何按布局派生的逻辑），
        让非法形状仍报原有错误、形状合法但 layout 与 `cu_seqlens_q` 矛盾的调用仍被拒绝；
        不改测试文件，不改 v2 / v4（v2 全量 376 passed 0 failed、v4 370 passed 0 failed 无此问题）
  - [x] SubTask 9.3: 验证——重编 v3 后 `TestVarlenValidation` 35 passed；
        `tests/test_flash_attn_npu_v3.py` + `test_flash_attn_npu_v3_metadata.py` 全量
        528 passed, 11 skipped, **0 failed**

### Task Dependencies（910）

- [Task 2] / [Task 3] / [Task 4] 均依赖 [Task 1]（布局契约与 `layout` 参数形状先冻结）
- [Task 3] / [Task 4] 与 [Task 2] 可并行（三个后端目录彼此独立、无共享头）
- [Task 5] 依赖 [Task 2] [Task 3] [Task 4]
- [Task 6] 依赖 [Task 5]
- [Task 7] 依赖 [Task 6]（其回归运行暴露了该缺陷）
- [Task 8] 依赖 [Task 6]（其回归运行暴露了该缺陷）；须在 [Task 6] 重跑前完成
- [Task 9] 依赖 [Task 8]（Task 8 修复后重跑 v3 全量才暴露出该回归）
