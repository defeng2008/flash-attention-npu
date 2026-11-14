# MLA BNSD 支持设计

> 分支：`mlav3`（基于 `origin/main` 82a9a8d）。设备：Ascend 910（dav-2201）。
> 关联文档：[mla_design.md](./mla_design.md)（MLA 主体设计，本文是其 BNSD 扩展）。
> 设计目标：**MLA 全场景（prefill / unabsorbed + decode / absorbed）原生支持 BNSD**，复用普通 FA3/FA4 已有的 BNSD 通路，不新增 ABI、不新增 kernel 模板轴、不改 tiling 结构体。

## 1. 目标与范围

让 MLA 像普通 FA3/FA4 一样接受 BNSD 视角的输入，即：

```
连续 BNSD (B, H, S, D).transpose(1, 2)  ->  非连续 (B, S, H, D) 视图  ->  送入算子
```

输出形态与现有非 MLA BNSD 约定保持一致（见 §3.2）。

| 形态 | 是否支持 BNSD | 选路 | 说明 |
| --- | --- | --- | --- |
| MLA prefill / unabsorbed，`Hk > 1`（GQA） | ✅ 支持 | 原生 BNSD（`layout=2`） | host 两处改动，kernel 零改动（§4.2 / §4.3） |
| MLA prefill / unabsorbed，`Hk == 1` | ✅ 支持 | 退化为 BSND（`layout=1`） | `Hk==1` 使 k/v 转置视图连续（§2.1），k/v 零拷贝，仅 q 物化 |
| **MLA decode / absorbed（传 `qv`）** | ✅ **本次支持** | **原生 BNSD（`layout=2`）** | `_detect_layout` 改造 + kernel `gmOffsetQv` + host 放开，见 §4.5 |
| 非 MLA | 既有能力 | —— | 行为不变 |

> `Hk` 指 KV 头数。MLA absorbed 形态恒有 `Hk == 1`；prefill 形态可为 1（MQA）或 >1（GQA）。

## 2. 现状（实测）

现有非 MLA BNSD 通路在 host 与 kernel 中已经完整，MLA 只是在 host 侧被两条 `TORCH_CHECK` 挡掉。实测（当前构建，910）：

```
输入 = contiguous BNSD (2,4,32,64).transpose(1,2)        # 非连续 (B,S,H,D)
-> out.shape == (2,32,4,64)，out.is_contiguous() == True
-> out.transpose(1,2).is_contiguous() == False
-> 数值与等价 BSND 输入逐位一致
```

即既有约定是「**输出与传入视图同 shape、连续存储**」，而不是「输出 `(B,H,S,D)`」。

MLA 当前的 BNSD 阻断点：

| 位置 | 内容 | 影响形态 |
| --- | --- | --- |
| v3 `flash_api.cpp:382` / v4 `flash_api.cpp:268` | `TORCH_CHECK(!is_bnsd, "MLA forward does not support BNSD layout; use TND (0) or BSND (1)")` | prefill + absorbed |
| v3 `flash_api.cpp:188` | `TORCH_CHECK(!q_v_.has_value(), "BNSD layout does not support a separate q_v tensor")` | absorbed |
| v3 接口 `1530-1535` | `out_transposed` 分支在 `qv is not None` 时抛 `NotImplementedError` | absorbed |
| Python `_detect_layout` | 依赖 q **和** k **和** v 三者同时为 BNSD 视图，absorbed 的 k/v（`Hk==1`）无法参与判定 | absorbed |

kernel 侧其实**已经备好 MLA 的 BNSD 寻址**：

- `mha_fwd_kvcache.cpp:254-269`（v3）/ `358-368`（v4）的 stride 派生块里，`strideQv` / `qvHeadStride` 已按 `INPUT_LAYOUT` 分支给出 BNSD 取值（`264-265` / `361-362`）。
- `qBOffset`（v3 `688` / v4 `651`）、`kBOffset` / `vBOffset`（v3 `699-700` / v4 `653-654`）均已是 layout-aware。
- O 的输出寻址被强制为 BSND（v3 `285-290`、`692-693`），保证输出恒为连续 `(B,S,H,dv)`。
- **唯一缺口**：`gmOffsetQv`（v3 `740-745` / v4 `703-708`）的 batch 基准写死为 BSND/TND 形式，注释亦标注 “MLA rejects BNSD”。该值**只有 absorbed 分支会消费**（`loadQGM(gQv...)`）。

### 2.1 一条关键事实：`Hk == 1` 的转置视图是连续的

`(B, 1, S, D).transpose(1, 2)` 得到的 `(B, S, 1, D)` 视图**是连续的**——size-1 维会吸收 stride 差异。实测：

```
shape=(1,4,1,64)  strides=(256,64,256,1)  is_contiguous() == True
```

因此 `_transposed_bnsd_view` 会对这类张量提前返回 `None`（`flash_attn_npu_interface.py:50` 的 `t.is_contiguous()` 短路）。这就是 **absorbed 无法靠 k/v 判定 BNSD 的根因**：absorbed 恒有 `Hk == 1`，k/v 的 BNSD 视图与「连续 BSND cache」在内存上完全等价，靠 stride 无法区分。

结论：**absorbed 的 layout 必须由带有真实 head 轴的 `q` 与 `qv` 共同判定**（§4.4）。同时因为 `Hk==1` 下 k/v 两种布局内存等价，BNSD 判定后直接把 k/v 以 `transpose(1,2)` 重新视图即可，**k/v 仍然零拷贝**。

## 3. 输入 / 输出约定

### 3.1 输入

调用方持有连续 BNSD：

| 张量 | 连续 BNSD shape | 传入视图 |
| --- | --- | --- |
| `q`（prefill：`q_nope|q_rope`；absorbed：`q_rope`） | `(B, H, Sq, Dq)` | `q.transpose(1,2)` |
| `qv`（仅 absorbed） | `(B, H, Sq, Dv)` | `qv.transpose(1,2)` |
| `k` / cache（`k_rope`） | `(B, Hk, S, Dq)` | `k.transpose(1,2)` |
| `v` / cache（`latent` / `v`） | `(B, Hk, S, Dv)` | `v.transpose(1,2)` |

判定为 BNSD 时，`_detect_layout` 把 q/k/v（及 qv）都还原成**连续的 BNSD 视图**交给 host（host 侧 `q.is_contiguous()` 校验 `flash_api.cpp:321` 因而通过）。

### 3.2 输出（与非 MLA 逐字一致）

输出为**连续张量**，`shape` 与调用方传入的视图**同形**，末维换成 `dv`：

| 调用方传入视图 | MLA 输出 |
| --- | --- |
| `(B, S, H, Dq)`（由 `(B,H,S,Dq).transpose(1,2)` 得到） | `(B, S, H, dv)`，`is_contiguous() == True` |

对 prefill：`dv = 128`；对 absorbed：`dv = qv.last = 512`。

> kernel 把 O 的寻址强制为 BSND（v3 `285-290`），host 的 BNSD 分支按 `(batch_size, seqlen_q, num_heads, head_size_v)` 分配，二者一致，故该约定对 prefill / absorbed 统一成立。

## 4. 设计

### 4.1 改动总览

| # | 文件 | 位置 | 改动 |
| --- | --- | --- | --- |
| 1 | v3 `flash_api.cpp` | `382` | 移除 MLA 的 `is_bnsd` 拒绝（保留其余 MLA 校验） |
| 2 | v3 `flash_api.cpp` | `188` | 移除 `!q_v_.has_value()` 的 BNSD 拒绝 |
| 3 | v3 `flash_api.cpp` | `404-407` | BNSD 输出末维改 `head_size_v` |
| 4 | v3 `flash_api.cpp` | `370` | `q_v` 校验由 `stride(-1)==1` 收紧为 `is_contiguous()` |
| 5 | v4 `flash_api.cpp` | `268` | 同 #1（v4 无 #2 对应检查） |
| 6 | v4 `flash_api.cpp` | `277-286` | 调整分支顺序：BNSD 分支置于 MLA 分支之前，末维用 `head_size_v` |
| 7 | v4 `flash_api.cpp` | `258` | 同 #4 |
| 8 | v3 `mha_fwd_kvcache.cpp` | `740-745` | `gmOffsetQv` batch 基准改 layout-aware（absorbed BNSD 核心） |
| 9 | v4 `mha_fwd_kvcache.cpp` | `703-708` | 同 #8 |
| 10 | v3/v4 `flash_attn_npu_interface.py` | `58-80` | `_detect_layout` 增加 absorbed 分支并返回 `qv` |
| 11 | v3/v4 `flash_attn_npu_interface.py` | `260` / `213` | `qv` 归一化为连续（BSND 回退路径） |
| 12 | v3 `flash_attn_npu_interface.py` | `1530-1535` | `out_transposed` 分支放开 `qv`（仍拒绝 append-KV） |
| 13 | v3/v4 `flash_attn_npu_interface.py` | `798` / `1527` | 适配 `_detect_layout` 新返回值 |
| 14 | 测试 | `tests/*mla*.py` | BNSD 拒绝用例改为「接受 + 数值正确」用例 |

### 4.2 Host（`flash_api.cpp`）——prefill / absorbed 共用

#### 4.2.1 放开 MLA 的 BNSD 拒绝

- v3 `382` / v4 `268` 的 `TORCH_CHECK(!is_bnsd, ...)` 删除。
- v3 `188` 的 `TORCH_CHECK(!q_v_.has_value(), ...)` 删除（v4 的 BNSD 块 `234-239` 本就没有该检查）。

MLA 与 layout 正交的校验全部保留：`head_size_v ∈ [1,512]`、absorbed 的 `Dq ≤ 256`、dtype FP16/BF16、`Dq == Dk`。

#### 4.2.2 输出分配

统一为「BNSD 分支优先、末维用 `head_size_v`」。非 MLA 下 `head_size_v == head_size_og`（v3 `397` 已断言），故对既有路径零影响。

v3（现 `404-416`，BNSD 已在 MLA 之前，只需改末维）：

```cpp
} else if (is_bnsd) {
    // BNSD (B,H,S,D) input: the kernel reads BNSD but writes contiguous BSND.
    // MLA 的 dv 与 q 的 head dim 不同，故用 head_size_v（非 MLA 时二者相等）。
    out = torch::empty(
        {batch_size, seqlen_q, num_heads, static_cast<int64_t>(head_size_v)}, q.options());
} else if (is_mla) {
    auto out_sizes = q.sizes().vec();
    out_sizes.back() = head_size_v;
    out = torch::empty(out_sizes, q.options());
} else {
    out = torch::empty_like(q);
}
```

v4（现 `277-289`，MLA 在 BNSD 之前，必须倒序，否则 MLA+BNSD 会按 `q.sizes()` 得到错误的 `(B,H,Sq,dv)` 且静默）：

```cpp
} else if (is_bnsd_layout) {          // 提到 is_mla 之前
    out = torch::empty(
        {q.size(0), q.size(2), q.size(1), static_cast<int64_t>(head_size_v)}, q.options());
} else if (is_mla) {
    auto out_sizes = q.sizes().vec();
    out_sizes.back() = head_size_v;
    out = torch::empty(out_sizes, q.options());
} else {
    out = torch::empty_like(q);
}
```

#### 4.2.3 `q_v` 连续性校验收紧

v3 `370` / v4 `258` 现为：

```cpp
TORCH_CHECK(q_v.stride(-1) == 1, "Input tensor q_v must have contiguous last dimension");
```

BNSD 下 kernel 用**固定 BNSD stride**（`qvHeadStride = maxQSeqlen*embedV`）读 qv，仅末维连续不足以保证正确，故收紧为：

```cpp
TORCH_CHECK(q_v.is_contiguous(), "Input tensor q_v must be contiguous because FA3/FA4 use packed Qv strides");
```

与同文件对 `q` 的 `is_contiguous()` 校验（v3 `321`）对齐。

#### 4.2.4 已天然正确、无需改动的项

| 项 | 位置 | 说明 |
| --- | --- | --- |
| `num_heads_k` | v3 `330` / v4 `344` | 已 BNSD-aware |
| `kv_seqlen_allocated` | v3 `333-334` / v4 `442` | BNSD 取 `k.size(2)`（head-plane pitch） |
| `seqlen_k_val` / `kvSeqlenBound` | v3 `425` / `506` | 已 BNSD-aware |
| `fwd_args.layout` | v3 `798-799` / v4 `731` | 已映射 `BNSD` |
| `launch_fa_infer` 分派 | v3 `829-842` / v4 `762-764` | 已含 BNSD 分支 |
| `softmax_lse` | v3 `476` / v4 `392` | `(B,H,S)`，与 layout 无关 |
| `embeddingSizeV` / `splitCtx.head_size_v` / `wsSplit` | v3 `605/685/571`；v4 `567/620/516` | 只依赖 `head_size_v`，与 layout 正交 |
| AICPU metadata | `fa_metadata.aicpu` | 签名不含 layout，调度只依赖 `(B,H,S,S2)` |

### 4.3 Kernel（`mha_fwd_kvcache.cpp`）——absorbed BNSD 唯一改动

`gmOffsetQv` 的 batch 基准改为 layout-aware，与 `qBOffset` 的 BNSD 形式（`BIdx * qHeads * qHeadStride`）对齐：

```cpp
uint64_t qvBOffset;
if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
    // BNSD: qv shares Q's (batch, head, seq) indexing; its head plane is
    // maxQSeqlen*embedV, so the batch base must include the qHeads factor.
    qvBOffset = static_cast<uint64_t>(BIdx) * qHeads * qvHeadStride;
} else {
    qvBOffset = static_cast<uint64_t>(prevQSeqlenSum) * strideQv;
}
uint64_t gmOffsetQv = qvBOffset +
                      static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQv +
                      static_cast<uint64_t>(qNStartIdx) * qvHeadStride;
```

落点：v3 `740-745`、v4 `703-708`，并删除 “MLA rejects BNSD” 注释。

- 该量只在 `mlaAbsorbed` 分支被 `loadQGM(gQv, ...)` 消费，**prefill 路径不受影响**（`mlaAbsorbed == false` 时 Q/K/V 走已 BNSD-aware 的 `qBOffset/kBOffset/vBOffset`）。
- `strideQv` / `qvHeadStride` 的 BNSD 取值（`embedV` / `maxQSeqlen*embedV`）在 stride 派生块中**已存在**，无需新增。

### 4.4 Python 接口

#### 4.4.1 `_detect_layout` 改造（absorbed 的关键）

现有 `_detect_layout` 要求 q、k、v **三者同时**为 BNSD 视图。absorbed 的 k/v 因 `Hk==1` 退化为连续（§2.1），永远判定不出 BNSD。改造为「**absorbed 由 q 与 qv 判定，k/v 重新视图**」，并把 `qv` 纳入返回值：

```python
def _detect_layout(q, k, v, page_table, cu_seqlens_k, qv=None):
    if k is not None and v is not None and v.dim() != k.dim():
        raise RuntimeError(...)

    if q.dim() != 4 or k.dim() != 4 or page_table is not None or cu_seqlens_k is not None:
        return _LAYOUT_BSND, q, k, v, qv, False

    if qv is not None:
        # Absorbed MLA: KV heads == 1, so a transposed BNSD k/v view collapses to a
        # contiguous buffer and cannot select the layout on its own. Decide from the
        # two tensors that carry a real head axis, q and qv.
        q_bnsd = _transposed_bnsd_view(q)
        qv_bnsd = _transposed_bnsd_view(qv)
        if q_bnsd is not None and qv_bnsd is not None:
            # Re-view k/v as BNSD (B, Hk, S, D). With Hk == 1 this is bit-identical
            # storage to a contiguous BSND cache, so it stays zero-copy.
            return (_LAYOUT_BNSD, q_bnsd, k.transpose(1, 2), v.transpose(1, 2), qv_bnsd, True)
        return _LAYOUT_BSND, q, k, v, qv, False

    q_bnsd = _transposed_bnsd_view(q)
    k_bnsd = _transposed_bnsd_view(k)
    v_bnsd = _transposed_bnsd_view(v)
    if q_bnsd is not None and k_bnsd is not None and v_bnsd is not None:
        return _LAYOUT_BNSD, q_bnsd, k_bnsd, v_bnsd, qv, True
    return _LAYOUT_BSND, q, k, v, qv, False
```

要点：

- 返回值由 5 元组变 6 元组（新增 `qv`）。`_detect_layout(...)[0]` 的调用点（`L490`）不受影响。
- **仅当 `qv is not None` 时**启用新分支，**prefill 与非 MLA 判定逻辑逐字不变**，避免把「q 是 BNSD 视图但 k/v 是普通 BSND」的非 MLA 调用误判为 BNSD。
- 判定成功后交给 host 的 q/k/v/qv 全部是**连续 BNSD**，`_prepare_forward_qkv` 的 `.contiguous()` 为 no-op，**零额外拷贝**。

#### 4.4.2 `qv` 归一化（混合布局回退）

即使走 BSND 回退（例如 q 为 BNSD 视图但 qv 为连续 BSND，或反向），`qv` 也必须整体连续——现 `maybe_contiguous(qv)`（v3 `260` / v4 `213`）**只归一化末维**，会让 kernel 用 packed stride 错读。改为：

```python
qv = qv.contiguous() if qv is not None else None
```

- BNSD 路径：`_detect_layout` 已返回连续 BNSD qv，`.contiguous()` 是 no-op。
- BSND 路径：物化为 packed BSND 缓冲区（与 `q` 的处理一致）。
- `qv is None`（prefill / 非 MLA）：无影响。

> 这是「全场景」的后半句：只要 q / qv 中**有任意一个**不是 BNSD 视图，就走 BSND 并物化，绝不静默错读。

#### 4.4.3 `out_transposed` 分支放开 `qv`

v3 接口 `1530-1535` 现为：

```python
if out_transposed:
    if k is not None or v is not None or qv is not None:
        raise NotImplementedError(
            "flash_attn_npu_3: 910 backend BNSD layout does not support "
            "append-KV (k/v) or qv"
        )
```

改为只拒绝 append-KV（`qv` 是 absorbed 的合法输入）：

```python
if out_transposed:
    if k is not None or v is not None:
        raise NotImplementedError(
            "flash_attn_npu_3: 910 backend BNSD layout does not support append-KV (k/v)"
        )
```

其后的 `cache_seqlens` 上界检查保留（`bsnd_seqlen_k` 在检测前取自 `k_cache.shape[1]`，对 BSND/BNSD 视图都是序列维）。

#### 4.4.4 调用点适配

| 入口 | 行 | 改动 |
| --- | --- | --- |
| `flash_attn_func`（含 `qv`） | v3 `798` | 解包 6 元组，把 `qv` 传给 `_detect_layout`，并在 `_flash_attn_forward` 调用点（`824`）传检测后的 `qv` |
| `flash_attn_with_kvcache` | v3 `1527` | 同上（`qv` 传入 `_detect_layout`，并回写 `qv`） |
| `FlashAttnFunc.forward` | v3 `490` | 仅取 `[0]`，无需改 |

`_flash_attn_forward` 内的 `elif layout == _LAYOUT_BNSD: batch_size, _, seqlen_q, _ = q.shape`（v3 `248-249`）已适配 BNSD 形状，无需改。

v4 侧对称改动 `flash_attn_npu_4/flash_attn_npu_interface.py`：`_detect_layout` 定义（同类行）与 `qv` 归一化、以及 `FlashAttnFunc.forward` 的调用点。差别在 v4 **没有** `flash_attn_with_kvcache` 入口，absorbed decode 走 `flash_attn_func` / `flash_attn_varlen_func`，因此 v4 不存在 `out_transposed` 的消费点——BNSD 输出由 host 直接分配成连续 `(B,S,H,dv)`，与调用方传入视图同形，无需再转置。

#### 4.4.5 全场景选路矩阵

| 场景 | q | qv | k / v cache | 选路 | 拷贝 |
| --- | --- | --- | --- | --- | --- |
| 标准 BNSD（absorbed） | BNSD 视图 | BNSD 视图 | `Hk==1` 连续 | 原生 BNSD | 无 |
| 标准 BNSD（prefill，`Hk>1`） | BNSD 视图 | —— | BNSD 视图 | 原生 BNSD | 无 |
| 标准 BNSD（prefill，`Hk==1`） | BNSD 视图 | —— | 连续 | BSND | 仅 q |
| 全连续 BSND | 连续 | 连续 | 连续 | BSND | 无 |
| 混合（q BNSD 视图、qv 连续） | BNSD 视图 | 连续 | 连续 | BSND | 仅 q |
| 混合（qv BNSD 视图、q 连续） | 连续 | BNSD 视图 | 连续 | BSND | 仅 qv |
| dim0 带间隙的 cache | 视图 | 视图 | 非连续 | 原生 BNSD | 仅 k/v（`_prepare_forward_qkv` 物化） |

## 5. 约束（更新）

在 [mla_design.md §5](./mla_design.md) 的拒绝表基础上：

| 约束 | 触发点 | 结果 | 变化 |
| --- | --- | --- | --- |
| ~~BNSD 布局~~ | ~~v3 `flash_api.cpp:382` / v4 `268` `TORCH_CHECK(!is_bnsd)`~~ | ~~报错~~ | **移除**（prefill + absorbed 均支持） |
| ~~BNSD 布局 + `q_v`（absorbed）~~ | ~~v3 `flash_api.cpp:188`~~ | ~~报错~~ | **移除** |
| ~~BNSD 布局 + `qv`（kvcache 入口）~~ | ~~v3 接口 `1531` `NotImplementedError`~~ | ~~报错~~ | **仅保留对 append-KV 的拒绝** |
| BNSD + paged KV | v3 `flash_api.cpp:186` / v4 `237` | 报错 | 不变 |
| BNSD + append-KV（`k_new/v_new`） | v3 `flash_api.cpp:187` / 接口 `1531` | 报错 | 不变 |
| BNSD + `cu_seqlens_k` | v4 `flash_api.cpp:238` | 报错 | 不变 |
| MLA absorbed + BNSD | 无 | 支持 | **新增能力** |

MLA 与 layout 正交的既有约束（`Dq == Dk`、`Dv ∈ [1,512]`、absorbed `Dq ≤ 256`、absorbed KV 头数=1、FP16/BF16、`softmax_scale` 显式传入）全部不变。

## 6. 影响面与风险

- **改动面**：prefill 只需 host 两处 + v4 分支顺序；absorbed 额外需要 `_detect_layout` 分支、kernel `gmOffsetQv`、host 两处放开、接口三处适配。tiling / AICPU / dispatch / 模板轴均不动，编译产物数量不变。
- **零额外拷贝**：标准 BNSD 场景下 q/k/v/qv 全是连续 BNSD 视图，`.contiguous()` 均为 no-op；只有混合非连续场景才物化。
- **非 MLA 路径零影响**：`_detect_layout` 的新分支以 `qv is not None` 为前提；`head_size_v == head_size_og` 保证输出形状与数值与改动前逐位一致。
- 主要风险点：
  1. **v4 分支顺序**：若 BNSD 分支留在 MLA 之后，MLA+BNSD 会按 `q.sizes()` 得到 `(B,H,Sq,dv)`，形状错误且静默 → 必须有回归用例。
  2. **kernel `gmOffsetQv`**：漏改会使 absorbed BNSD 静默错读（batch 基准少乘 `qHeads`）→ 必须有跨 batch 的回归用例。
  3. **`_detect_layout` 适用范围**：新分支必须严格限定在 `qv is not None`，否则会把非 MLA 的「q 是 BNSD 视图、k/v 是普通 BSND」误判为 BNSD。
- `out_transposed` 在 `FlashAttnFunc.forward` 中赋值后未被消费属既有历史行为（输出保持 CBSND 视角），不属本次改动。

## 7. 验证计划（已落地）

新增测试（v3 与 v4 对称；absorbed decode 在 v3 走 `flash_attn_with_kvcache`、在 v4 走 `flash_attn_func`）。所有 BNSD 用例统一用 `_to_bnsd_view(x) = x.transpose(1, 2).contiguous().transpose(1, 2)`，即「连续 BNSD 存储 + 非连续 BSND 视图」。

| 用例 | 构造 | 落点 |
| --- | --- | --- |
| MLA prefill BNSD（`Hk == H`） | `q/k=(2,64,8,192)`、`v=(2,64,8,128)` 的 BNSD 视图，`causal=True` | `test_mla_prefill_bnsd` |
| MLA prefill BNSD + GQA（`Hk > 1`，覆盖 `qBOffset`/`kBOffset`/`vBOffset` 的 BNSD 分支） | `q=(2,96,16,192)`、`k/v=(2,96,4,·)` | `test_mla_prefill_bnsd_gqa` |
| **MLA absorbed BNSD 跨 batch**（覆盖 `gmOffsetQv` 的 batch 基准，§4.3） | `B=2`、`Sq=4`、q/qv/cache 全 BNSD 视图 | `test_mla_decode_bnsd_cross_batch` |
| 布局等价性（双后端） | BNSD 视图调用 vs 同值 BSND 调用 | `test_mla_accepts_bnsd`（consistency） |
| MLA prefill BNSD（`Hk == 1`）/ 混合布局 | —— | 不单独建：`Hk == 1` 时 k/v 的转置视图本身连续，`_detect_layout` 自动回退 BSND，数值与 BNSD 视图完全等价，由既有 BSND 用例覆盖 |
| MLA absorbed BNSD（`Sq == 1`） | —— | 不单独建：`Sq == 1` 时 BNSD 视图与 BSND 是同一块连续内存，不存在可区分的布局 |
| 非 MLA BNSD 回归 / MLA BSND·TND 回归 | 既有用例 + 三套件 | 全绿 |

既有「拒绝 BNSD」用例全部改造为「接受 + 数值正确」：`test_mla_rejects_bnsd`（consistency）→ `test_mla_accepts_bnsd`；v3 `test_mla_rejects_bnsd` → 上表三个新用例；v4 `test_mla_rejects_bnsd_prefill` / `test_mla_rejects_bnsd_decode` → 上表三个新用例。

回归命令（沿用工程既有口径）：

```bash
source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh
export LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/11:/usr/lib/aarch64-linux-gnu
ASCEND_RT_VISIBLE_DEVICES=0 /home/w00811046/miniconda3/envs/cann/bin/python -m pytest \
  tests/test_flash_attn_npu_v3_mla.py tests/test_flash_attn_npu_v4_mla.py \
  tests/test_flash_attn_npu_mla_consistency.py -q -p no:randomly
```

重编前必须删掉过期的 `fwd_dispatch_*` 目标文件（`build/temp.linux-aarch64-cpython-310/flash_attn_npu_{3,4}.flash_attn_npu_{3,4}.objs/fwd_dispatch_*.o` 与同名 `.o.cmdhash`）：setup.py 对它们用 mtime/`.cmdhash` 增量判据，不追踪被 include 的 `mha_fwd_kvcache.cpp`，只改 kernel 实现时这 6 个 `.o` 会被判定 up-to-date 而跳过重编，产物仍是旧 kernel。本次就因此出现了「v3 absorbed BNSD 只有 `b=1` 数值错、v4 全对」的假象——v4 的 `.o` 恰好是新一轮编出来的，v3 的不然。

## 8. 相关文件

- Host：`csrc/ascend910/flash_attn_npu_{3,4}/flash_api.cpp`
- Kernel：`csrc/ascend910/flash_attn_npu_{3,4}/mha_fwd_kvcache.cpp`（`gmOffsetQv`）
- Python：`flash_attn_npu_{3,4}/flash_attn_npu_interface.py`（`_detect_layout` / `_prepare_forward_qkv` / `qv` 归一化 / `out_transposed`）
- 测试：`tests/test_flash_attn_npu_v3_mla.py`、`tests/test_flash_attn_npu_v4_mla.py`、`tests/test_flash_attn_npu_mla_consistency.py`
- 关联设计：`docs/mla/mla_design.md`
