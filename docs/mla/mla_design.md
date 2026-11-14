# FA3 / FA4 MLA 前向设计

> 分支：`mlav3`（基于 `origin/main` 82a9a8d）。设备：Ascend 910（dav-2201）。
> 设计原则：MLA 是 **FA 前向流水内的一种形状签名**，复用同一套 tiling、分核与 kernel 计算原语，不引入与 FA 平行的 ABI。

## 1. 形态判定

MLA 不做显式 flag，完全由张量形状签名判定，判定点在 host 侧 `mha_fwd`（v3/v4 各一份 `flash_api.cpp`）。

| 形态 | 签名 | 计算 |
| --- | --- | --- |
| prefill / unabsorbed（训练口径） | 不传 `qv`，`q.last == k.last == 192`，`v.last == 128` | 单段 QK（192 宽，rope 已在末维），PV 128 宽，支持 GQA |
| decode / absorbed（推理口径） | 传 `qv`（`q.last == k.last == 64`，`qv.last == v.last == 512`，KV 头数 = 1） | 两段 score 累加进同一 S：`Q_nope·K_nopeᵀ ⊕ Q_rope·K_ropeᵀ`，PV 的 V 即 latent，输出 512 宽 |

判定表达式：

```
has_q_v  = q_v_.has_value()
head_size_v = has_q_v ? q_v.size(-1) : v.size(-1)
is_mla   = has_q_v || (head_size_og != head_size_v)
```

即「**传了 `qv`**」或「**q/k 头维与 v 头维不等**」任一成立即进入 MLA。后者是 prefill 的判据，避免为训练口径新增参数。

非 MLA 输入（`q.last == v.last` 且无 `qv`）不进入任何 MLA 分支，行为与改动前逐位一致。

### 1.1 为什么不需要新增 tiling 字段

`embeddingSize`（QK 宽度）与 `embeddingSizeV`（输出/PV 宽度）在 FA 中**本来就是两个独立的 runtime 字段**，只是此前 host 侧恒把 `embeddingSizeV` 写成与 `embeddingSize` 相同。MLA 只需要把这个赋值改成真实值 `head_size_v`，tiling 结构体的字段、偏移与语义均不变：

- v3 `FAInferTilingData`：未改动。
- v4 `FAInferStaticTilingData` / `FAInferRuntimeData`：未改动。
- absorbed 形态的 576 宽 QK 不落 tiling，由 kernel 用 `embedQK = embed + embedV` 现场派生。
- latent query 指针 `qv` 属于 per-forward 入参，走 kernel 启动参数（`FAIKernelParams.qv`），不占 tiling 位。

这样既满足「严格复用 FA 结构体」，也避免动 `tilingdata.h` 触发的跨编译单元 ABI 风险。

## 2. Host 侧改动

### 2.1 `flash_api.cpp`

v3 / v4 改动对称，差异仅在 v4 多一条 `sinkDevice` 与 v4 自己的 layout 检测函数名：

1. 形态判定与校验（见 §1）。MLA 额外校验：
   - `!has_q_v || head_size_og <= 256`：absorbed 形态的 rope Q 头维必须 ≤ 256。
   - `head_size_v ∈ [1, 512]`：输出/latent 宽度上限。
   - absorbed：`q_v.size(-1) == v.size(-1)` 且 dtype 与 q 一致、**全连续**（kernel 按 packed BSND/BNSD stride 读 latent query）。
2. 输出分配：非 MLA 仍 `empty_like(q)`；BNSD（含 MLA）统一分配连续 `(B,S,H,dv)`；MLA 非 BNSD 复制 q 的 shape 后把末维替换为 `head_size_v`（BSND `(B,S,H,dv)`、TND `(total,H,dv)`）。BNSD 分支必须排在 MLA 分支之前：MLA BNSD 的 `q.sizes()` 是 `(B,H,S,dq)`，先走 MLA 分支会把输出错当成 `(B,H,S,dv)` 而静默错形状。
3. `embeddingSizeV = head_size_v`（原为 `head_size_og`）。
4. `splitCtx.head_size_v = head_size_v`：分核成本模型与 FlashDecode workspace 偏移都按输出宽度计。
5. `wsSplit = partialRowsUpper * (head_size_v + 1) * sizeof(float)`：FD 归约 workspace 按 v 宽分配。
6. `fwd_args.qvDevice = has_q_v ? q_v.data_ptr() : nullptr`：唯一新增的 fwd 入参。

`fa_split.h::splitBN2S1GS2` 本身未改动，MLA 与普通 FA 共用同一套 `coreInfo` / `splitInfo` 与同一套 FlashDecode 归约规划。

AICPU metadata 同样无需改动：`fa_metadata.aicpu` 从 `metaArgs->embeddingSizeV` 读 `ctx.head_size_v`，host 侧把 `embeddingSizeV` 写对即自动一致。

### 2.2 Python 接口

v3 `flash_attn_npu_interface.py`：

- `_flash_attn_forward` 把 `qv = maybe_contiguous(qv)` 改为 `qv = qv.contiguous()`：kernel 按 packed BSND/BNSD stride 读 latent query，只保证末维连续不够——BNSD 视图的 head stride 也不连续，仅归一化末维会让 kernel 按固定 packed stride 读取时行错位、结果静默错误。
- `_detect_layout` 增加 `qv` 入参并返回 6 元组：absorbed 形态 KV 头数恒为 1，transposed BNSD 的 k/v 视图退化为连续存储、无法单独判定布局，改由 `q` + `qv`（带真实 head 轴）判定；判定为 BNSD 时 k/v 仍零拷贝 re-view 成 `(B,Hk,S,D)`。
- `FlashAttnFunc.forward` / `FlashAttnVarlenFunc.forward` 的 `get_scheduler_metadata` 调用把 `headdim_v` 从 `head_size` 改为 `v.shape[-1]`：MLA prefill 的 v 宽度（128）与 q/k（192）不同，metadata 的 workspace 规划必须按 v 宽。
- `flash_attn_with_kvcache` 的 `qv` 语义沿用原有实现（`(q.last + qv.last) ** -0.5` 默认 scale）；BNSD 下额外拒绝 append-KV，并校验 `cache_seqlens` 不超过 BNSD 的 KV 序列维。

v4 `flash_attn_npu_interface.py`：

- `_flash_attn_forward` 的 `qv` 归一化同 v3（`contiguous()`）。
- `_detect_layout` 的改造与 v3 对称。
- `_flash_attn_forward_fake` 的输出 shape 改为 `list(q.shape)` 后末维置为 `v.shape[-1]`；原 `empty_like(q)` 在 MLA 下形状错误。
- v4 不新增 `flash_attn_with_kvcache`；decode 通过 `flash_attn_func` / `flash_attn_varlen_func` 的 `qv` + `page_table` / `seqused_k` 表达。

### 2.3 Dispatch

`FwdLaunchArgs` 末尾加 `uint8_t* qvDevice`，`FWD_LAUNCH` 宏末位透传，`launch_fwd_dtype` 取值后交给 `FAInfer(...)` 的末位默认参数 `GM_ADDR qv = nullptr`。

dispatch 的模板轴（`dtype × layout × paged × mask × softcap × flashDecode`）**不新增 MLA 维**：MLA 与普通 FA 走同一份编译出的 kernel，head dim 仍是 runtime 轴。因此 `autogen/generate_kernels.py` 与 `setup.py` 无需改动，编译目标数量不变。

## 3. Kernel 侧改动

### 3.1 参数与模式开关

`kernel_common.hpp::FAIKernelParams` 末尾新增 `GM_ADDR qv`（构造函数尾参默认 `nullptr`），全仓只有 `mha_fwd_kvcache.cpp` 的 `FAInfer` 会传它。

`FAInferKernel` 在 `init` 中派生：

```
mlaAbsorbed = (params.qv != nullptr);
embedQK     = mlaAbsorbed ? (embed + embedV) : embed;
headDimSplit = embedQK > 256;      // 原为 embed > 256
qL1DynNum    = RoundUp(embedQK, BLOCK_SIZE);
```

absorbed 下 `embed=64`、`embedV=512`、`embedQK=576`。`headDimSplit` 因此恒为真，`dSliceNum = ceil(576 / 256) = 3`：slice 0/1 覆盖 latent `[0,256)/[256,512)`，slice 2 起点 512 ≥ `mlaLatentDim`，自动落到 rope 源。

### 3.2 Q 双源

`qk_matmul.hpp::loadQGM` 增加 MLA 分支，复用新抽出的 `loadQSegmentToL1`（与 V3 原实现逐字等价）：

```
loadQSegmentToL1(gA,        0,           mlaLatentDim, ...)  // latent qv → L1 列 [0,512)
loadQSegmentToL1(mlaRopeQ,  mlaLatentDim, embed - mlaLatentDim, ...) // rope q → L1 列 [512,576)
```

两段写入**同一个** 576 宽 zN 块的相邻列区间，`LayoutAInL1::GetOffset` 的 C0 块落点与 `headDimSplitQK` 的 L1A 偏移天然对齐，后续 QK matmul 不需要感知双源。

### 3.3 K 双源

`copyKHeadDimSliceToL1` 按 `dOffset = dSlice * l1KDynamic` 选源：

```
if (mlaAbsorbed && dOffset >= mlaLatentDim) {
    src = mlaRopeK;          // rope K
    srcStride = mlaRopeKStride;
    srcPageStride = mlaRopeKPageStride;
    dOffset -= mlaLatentDim;
}
```

latent 段的 K 源就是 **V cache**（`gV`），因此 QK 的 `layoutB`、token stride、page stride 全部切到 PV 口径；rope 段用单独注册的 `gK` 及其 stride。paged 场景下 `gOffset` 的 page 项也要按源切换（latent 用 `valuePageStride`，rope 用 `keyPageStride`），所以 `pageStride` 与 `strideKV` 都随源重算。

调用点（`mha_fwd_kvcache.cpp`）：

```
AscendC::GlobalTensor<ElementK> gQKSrcLatent = mlaAbsorbed ? gV[gmOffsetV] : gQKSrc[qkSrcOffset];
LayoutK layoutQKSrc = mlaAbsorbed ? layoutKLatent : layoutKTemp;
const uint32_t qkStrideKV = mlaAbsorbed ? strideV : strideK;
```

并配合 `blockMmadQK.SetPageStride(mlaAbsorbed ? valuePageStride : keyPageStride)` 与三个 setter（`SetMlaLatentDim` / `SetMlaRopeQ` / `SetMlaRopeK`）。

### 3.4 PV、epilogue 与 FD 归约

V 宽度本来就是 `embedV`，absorbed 下即 512，PV 流水无需改动。需要改的是**此前误用 `embed` 的三处输出侧几何**：

| 位置 | 原值 | 改为 |
| --- | --- | --- |
| `LayoutUpdate layoutUpdate` | `embed, embedRound` | `embedV, embedRoundV` |
| `LayoutLse layoutgmLo`（FD 归约） | `embed * qNBlockSize` | `embedV * qNBlockSize` |
| `strideQv` / `qvHeadStride`（新增） | — | BNSD：`embedV` / `maxQSeqlen*embedV`；否则 `qHeads*embedV` / `embedV` |

prefill 形态 `embed=192 / embedV=128` 恰好暴露了这些旧假设，所以这组修改对 prefill 也是必需的（此前输出宽度被错当 192）。

latent query 的地址基址 `gmOffsetQv` 必须按 layout 分支（`mha_fwd_kvcache.cpp`）：BNSD 下 batch 基址是 `BIdx * qHeads * qvHeadStride`（head 维是真实的物理轴），BSND/TND 下仍是 `prevQSeqlenSum * strideQv`。漏掉这一分支时 B>1 的 absorbed BNSD 会读到错误的 batch 基址。

`combine_kernel.cpp` / `combine_scale.hpp` 未改动：FD 归约的 partial workspace 由 host 按 `head_size_v + 1` 预留，kernel 侧按 `embedV` 读写，两侧一致。

### 3.5 两段 score 共用累加器

absorbed 的 QK 是**一次 576 宽 matmul**，不是两次 matmul 相加：Q/K 的 latent 与 rope 两段在 L1 内拼成连续的 576 宽 zN/nZ 块，`headDimSplitQK` 按 256 列切片依次累加进同一个 S 累加器（`dSlice==0 && kL0Idx==0` 时初始化，其余累加）。这样 online softmax 看到的是一份完整 score，无需感知 MLA。

## 4. 接口契约

| 项 | v3 | v4 |
| --- | --- | --- |
| prefill 入口 | `flash_attn_func` / `flash_attn_varlen_func` | 同左 |
| decode 入口 | `flash_attn_with_kvcache`（含 `qv`） | `flash_attn_func` / `flash_attn_varlen_func` + `qv` + `page_table` / `seqused_k` |
| 输出末维 | `qv.last`（有 qv）或 `v.last` | 同左 |
| 默认 `softmax_scale`（有 qv） | `(q.last + qv.last) ** -0.5` | `(q.last + qv.last) ** -0.5` |
| 默认 `softmax_scale`（无 qv） | `q.last ** -0.5` | `q.last ** -0.5` |
| layout | TND(0) / BSND(1) / BNSD(2) | TND(0) / BSND(1) / BNSD(2) |
| `softmax_lse` | 带 `qv` 时 `(..., s, h)` | 同左 |

**scale 口径必须显式传**：prefill 的正确口径是 `192 ** -0.5`，absorbed decode 的 score 覆盖完整 576 宽内积，口径是 `576 ** -0.5`。两者不是同一个数，训推一致性测试必须显式传入统一 scale（见 §6）。

## 5. 约束（明确拒绝）

| 约束 | 触发点 | 结果 |
| --- | --- | --- |
| append-KV（`k_new`/`v_new`） | v3 `TORCH_CHECK(!appendKV)` | 报错 |
| `q.last > 256` 且带 `qv` | `TORCH_CHECK(head_size_og <= 256)` | 报错 |
| `v.last ∉ [1,512]` | `TORCH_CHECK(head_size_v ∈ [1,512])` | 报错 |
| 反向 | bwd 包装层拒绝不等 head dim | `ValueError` |
| `gather_kv_indices`（v4） | `flash_api.cpp` 直接 `TORCH_CHECK` | 报错 |

未使用的既有参数（`q_descale` / `k_descale` / `v_descale`）在 MLA 下必须为 `None`；带量化输入会被既有 dtype 校验拒绝。

BNSD 不再被拒绝（原 `TORCH_CHECK(!is_bnsd)` 已删除）：MLA 的 prefill 与 absorbed 两种形态都支持 BNSD 输入，输出连续 `(B,S,H,dv)`，与非 MLA BNSD 约定一致。唯一例外是 v3 `flash_attn_with_kvcache` 的 BNSD 分支仍然拒绝 append-KV（`k`/`v` 非空），因为该分支不更新 cache。absorbed 的 BNSD 判定依赖 `Sq > 1`（见 `mla_bnsd_design.md`）：`Sq == 1` 时 transposed BNSD 视图本身就是连续存储，不带任何 stride 证据，此时 BNSD 与 BSND 是同一块内存，无需也无法区分。

## 6. 训推一致性口径

同一 latent 状态两条路：

- prefill：`q = [q_nope(128) | q_rope(64)]`、`k = [k_nope | k_rope]`、`v = latent @ W_uvᵀ`（128 宽）。
- decode：`qv = q_nope @ W_uk`（512）、`q = q_rope`（64）、`k = k_rope`（64）、`v = latent`（512）。

二者是同一双线性形式，因此

```
out_prefill ≈ (out_decode @ W_uvᵀ)
```

验收阈值：bf16 下相对 L2 < 5e-3，并另用 FP64 CPU golden 分别交叉验证两条路。实测（v3/v4 相同）：prefill-vs-golden 2.283e-3、decode-vs-golden 2.184e-3、prefill-vs-decode 4.115e-3。

注意：`k_nope = latent @ W_ukᵀ` 与 `v = latent @ W_uvᵀ` 在 golden 中使用**同一次 bf16 舍入后的值**，否则 bf16 输入误差会被放大到超出阈值。

## 7. 相关文件

- 宿主：`csrc/ascend910/flash_attn_npu_{3,4}/flash_api.cpp`、`fwd_dispatch.hpp`、`fwd_dispatch_impl.hpp`
- Kernel：`csrc/ascend910/flash_attn_npu_{3,4}/kernel_common.hpp`、`mha_fwd_kvcache.cpp`、`qk_matmul.hpp`
- Python：`flash_attn_npu_{3,4}/flash_attn_npu_interface.py`
- 测试：`tests/test_flash_attn_npu_v3_mla.py`、`tests/test_flash_attn_npu_v4_mla.py`、`tests/test_flash_attn_npu_mla_consistency.py`、`tests/perf/bench_mla_vs_fia.py`
