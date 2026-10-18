# Checklist

## 设计约束（本次独立重写）
- [x] 未复用、未合入 `bsnd` 分支的任何提交或代码，v3 与 v4 各自独立实现
- [x] 使用**单一**布局参数承载 Q/K/V 三者布局（未区分 `q_bnsd` / `kv_bnsd`）：
      最终形态为编译期 `Format kLayout`（模板参数），无任何运行期 `bnsd` 布尔量
- [x] host 侧用 `FwdLayout { TND = 0, BSND = 1, BNSD = 2 }` 三值枚举替换原 `bool is_tnd`，
      取值与 `Format` 逐项一致，可直接 `static_cast` 互转
- [x] 模板实参为 `launch_fwd_impl<DType, Format kLayout>`；`autogen/` TU 由 4 增至 6
      （`(fp16, bf16) × (TND, BSND, BNSD)`），`setup.py` 用 glob 收集无需改动
- [x] 未修改 `FAInferTilingData` 字段集合，未改动 `fa_metadata.aicpu`
- [x] 对外 Python 公开签名未新增任何布局参数（无 `bnsd` / `q_bnsd` / `kv_bnsd`）
- [x] BNSD 只实现前向，未实现 BNSD 反向

## v3 CANN 层
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

## v3 host 层
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

## v3 Python 层
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

## v4 后端
- [x] v4 的 `Format`、kernel 入口参数、`FwdLaunchArgs`、宏实参与 v3 同构且一致
- [x] v4 kernel 步长/偏移/LSE/输出写回支持 BNSD
- [x] v4 host 按 `int64_t layout` 校验与透传，且公开 API 未新增布局参数
- [x] v4 `FlashAttnVarlenFunc` 在 `cu_seqlens_q is not None` 时为 `_LAYOUT_TND`，否则为 `_LAYOUT_BSND`
- [x] v4 未新增 `flash_attn_with_kvcache`
- [x] v4 的 autogen TU 数由 4 增至 6（与 v3 一致）
- [x] v4 三个 autograd 的 backward 返回元组个数已随 forward 入参个数同步调整

## 构建与验证
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
- [x] BNSD 用例已落进仓库测试目录：`tests/test_flash_attn_npu_v3_bnsd.py`（8 passed）与
      `tests/test_flash_attn_npu_v4_bnsd.py`（6 passed），覆盖 stride 零拷贝路径、连续 rank-4 回退 BSND、
      causal、LSE 布局、GQA 多头组、forward-only 拒绝、BNSD kvcache 分配平面长度
- [x] BNSD 支持 `scheduler_metadata` 后：实测 BNSD 调用 `get_scheduler_metadata` 恰好 1 次且尺寸取 BSND 语义；
      v3/v4 metadata 套件（15 passed / 80 skipped、45 passed / 6 skipped）与主回归（345/99、320/4）全绿

## 去 `bnsd` 重构（Task 8）
- [x] 全仓 `grep bnsd` 在 `csrc/` 与 `flash_attn_npu_*/` 下仅剩 `_transposed_bnsd_view` 辅助函数名与
      `autogen/generate_kernels.py` 的 `("bnsd", ...)` 生成键，无任何 `bnsd` 变量或形参
- [x] 重构后全量重编（`FLASH_ATTN_BUILD_NPU=950`）通过（0 error，两个 `.so` 链接成功）
- [x] 重构后 v3 回归 `tests/test_flash_attn_npu_v3.py`（345 passed / 99 skipped / 0 failed）
      与 v4 回归 `tests/test_flash_attn_npu_v4.py`（320 passed / 4 skipped / 0 failed）无 failed
- [x] 重构后 `docs/enable-950-qkv-bnsd/test_kv_seqlen_allocated.py`（BSND + BNSD 两用例）通过（2 passed）
- [x] `docs/` 与 `.trae/specs/` 的 spec / tasks / checklist / implementation 与最终实现一致（TU 4→6、
      无 `params.bnsd`、`launch_fwd_impl<DType, Format>`、`_detect_bnsd`→`_detect_layout`、`layout: int` 边界）
