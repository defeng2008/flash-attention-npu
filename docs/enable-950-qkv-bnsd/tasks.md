# Tasks

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
  - [x] SubTask 8.9: `docs/enable-950-qkv-bnsd/test_kv_seqlen_allocated.py` 同步 `_detect_layout` 与 `_LAYOUT_BNSD`
  - [x] SubTask 8.10: 全量重编（CANN 9.1.0 + `FLASH_ATTN_BUILD_NPU=950`）并跑 v3/v4 回归 + kv_seqlen_allocated 用例

# Task Dependencies
- Task 2 依赖 Task 1
- Task 3 依赖 Task 2（`FwdLaunchArgs` 与 kernel 参数形态）
- Task 4 依赖 Task 3（`fwd` pybind 签名）
- Task 5 依赖 Task 1~4 的设计定稿（可与 Task 3/4 并行推进，但需先完成 5.1 的结构核对）
- Task 6 依赖 Task 4、Task 5
- Task 7 依赖 Task 6
- Task 8 依赖 Task 6（重编验证），与 Task 8.1~8.9 的源码改动为同一批次
- Task 1、Task 5.1 可并行
