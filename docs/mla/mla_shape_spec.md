# MLA Shape 规格与跨实现对照

本文给出本仓（`flash_attn_npu_3` / `flash_attn_npu_4`）MLA 的**形状规格矩阵**，并与
参考实现 **daoailab（NVIDIA flash-attention）** 做逐项对照。

设计动机与模块划分见 [mla_design.md](./mla_design.md)；本文只聚焦「形状 → 校验 → tile 几何」。

引用约定：本仓文件用相对链接；daoailab 文件（`/home/w00811046/code/mla/daoailab`）用仓库相对路径。
无法从源码确认处标注「待确认」。

---

## 1. 形态判定（进入 MLA 路径的前置条件）

**Python 层** `_is_mla`
（[flash_attn_npu_interface.py#L39-L54](../../flash_attn_npu_3/flash_attn_npu_interface.py)）：

```python
_is_mla = qv is not None                       # 传 qv 一律走 MLA
       or (Dq == 192 and Dv == 128)            # 不传 qv：只认 prefill 签名，且保留非打包 query 前置条件
```

传 `qv` 但形状不匹配的输入由 native MLA 校验报错（不会落到普通 FA3）；不带 `qv` 的其余头维
组合交回普通 FA3。`IsMlaForward`（[flash_api.cpp#L169-L175](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)）
在 `fa_fwd` 处再按 `qv ? (Dq==64 && Dv==512) : (Dq!=Dv && Dq==192 && Dv==128)` 门控一次。

**C++ host 层**`CheckForwardBoundary`
（[mla_validation.h#L91-L97](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)）
再做一次边界判定，并补一条反向排除：`!qv && q.size(-1) == v.size(-1)` → 交回普通 FA3。

即**两种触发方式**：显式传 `qv`，或 Q=192 / V=128 的未吸收形态。

---

## 2. 两种形态的张量形状

| | **compressed（吸收 / absorbed）** | **prefill（未吸收）** |
|---|---|---|
| 触发 | 传了 `qv` | `qv = None` |
| `q` | `[..., H, 64]` | `[..., H, 192]` |
| `qv` | `[..., H, 512]` | — |
| `k` | `[..., H_kv, 64]` | `[..., H_kv, 192]` |
| `v` | `[..., H_kv, 512]` | `[..., H_kv, 128]` |
| KV 头数 | **恒 = 1** | ≥ 1，支持 GQA |
| 默认 `softmax_scale` | `1/√(64+512)` | `1/√192` |
| 校验位置 | [mla_validation.h#L132-L134](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h) | [mla_validation.h#L145-L146](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h) |

**形状耦合约束**

- `q.shape[-1] == k.shape[-1]`（[L101](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)）
- `qv` 与 `q` 除最后一维外形状必须完全一致（[L127-L131](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)）
- `k` / `v` 除 head dim 外形状必须一致（[L118-L120](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)）
- `heads % kvHeads == 0`

> **不支持 576 拼接式布局**。必须拆成 `q(64) + qv(512) + k(64) + v(512)` 四个独立张量，否则在
> `CheckForwardBoundary` 阶段就被拒（`q.size(-1) == 576 && v.size(-1) == 512` 不满足任一形态签名）。

---

## 3. 布局（layout）

**用户面** `layout` 枚举：`TND = 0 / BSND = 1 / BNSD = 2`。
**MLA 只接受 0 或 1**，BNSD 直接报错（[mla_forward.cpp#L401](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)）。

**内部** `MlaContext::layout` 是另一套编码
（[mla_metadata.h#L43](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h)），由 host 自动推导：

| 内部值 | 含义 | Q | KV |
|---|---|---|---|
| 0 | BSND 稠密 | 4D | 4D |
| 1 | TND 稠密 | 3D | 3D |
| 2 | 混合：TND Q + BSND KV | 3D | 4D |
| 3 | BSND Q + 分页 KV | 4D | 4D + `page_table` |
| 4 | TND Q + 分页 KV | 3D | 4D + `page_table` |

**禁止混搭**：`BSND Q + TND KV` 直接报错
（[L135](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)、[L147](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)）。

### 各布局具体形状

```
BSND 稠密:  q (B, Sq, H, Dq)      k (B, Sk, Hkv, Dk)   v (B, Sk, Hkv, Dv)
            qv(B, Sq, H, Dv)                            且 q.size(0) == k.size(0)

TND 稠密:   q (Tq, H, Dq)         k (Tk, Hkv, Dk)      v (Tk, Hkv, Dv)
            qv(Tq, H, Dv)         配套 cu_seqlens_q/k (B+1,) int32 + max_seqlen_q/k
                                  可选 seqused_q/k (B,) int32

分页 PA:    q/qv 同 BSND 或 TND
            k (Np, P, Hkv, Dk)    v (Np, P, Hkv, Dv)   ← 第 0 维是物理页号
            page_table (B, max_pages) int32
            逻辑容量 = max_pages × P
```

### BNSD 内存上的 BSND 视图（`transpose(1, 2)`）

调用方持有连续 BNSD 张量 `(B, H, S, D)` 时，可以 `.transpose(1, 2)` 得到
**BSND 逻辑形状 + BNSD 内存顺序**的非连续视图 `(B, S, H, D)`，直接传给 MLA：

- **被接纳为 BSND**，`layout` 仍为 `1`。MLA 不新增 BNSD 分支，C++ 依旧只接受 `0/1`。
- **strides 透传**：`stride(-1) == 1` 是唯一硬要求，外层 stride 经
  `MlaBsndTensorLayout{batchStride,rowStride,headStride}` 原样下发给 kernel 寻址，
  因此**不做 `.contiguous()` 物化**（零拷贝）。对齐安全：`D ∈ {64,128,192,512}` ⇒
  `rowStride = D`、`headStride = S*D`（元素）换算成字节均为 32B 倍数。
- **输出形状跟随调用方**：`out.shape == q.shape`，即调用方传入视图的 BSND 形状
  `(B, Sq, H, Dv)`；MLA **不回带** BNSD 形状。
- `lse` 恒为 `(B, H, Sq)`，与布局无关。
- **不适用**：TND（`cu_seqlens_q` 打包）与分页 KV（`page_table`）——分页 KV 的第 0 维是物理页号，
  不存在 BNSD 视图语义。
- 注意 **Hkv = 1 的陷阱**：compressed 形态下 `k`/`v` 的 `transpose(1, 2)` 结果仍是连续的
  （size-1 维度使 stride 无关），因此不能靠 `is_contiguous()` 判定「是否 BNSD 视图」，
  也不应对 MLA 施加「q/k/v 必须同时命中」之类的判定。

---

## 4. 维度与容量约束

容量上限（[mla_tiling.h#L18-L27](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tiling.h)）：

| 常量 | 值 | 约束对象 |
|---|---|---|
| `kMlaMaxAic` | **26** | 物理核数上限 `physicalCores` |
| `kMlaMaxKvLength` | **16384** | Sk / 分页逻辑容量 |
| `kMlaMaxQueryLength` | 2^30 = 1073741824 | Sq |
| `kMlaMaxQueryHeads` | 8388607 | H |
| `kMlaMaxDenseRows` | 2^32 − 256 | Sq × H |
| `kMlaMaxBatchRows` | 2^63 / 1024 | B × Sq × H |

分页专属（[mla_metadata.h#L135](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h)、
[mla_layout.h#L112-L117](../../csrc/ascend910/flash_attn_npu_3/mla/mla_layout.h)）：

- `pageSize P ∈ [16, 1024]` 且**必须是 16 的倍数**
- `Sk % pageSize == 0`
- `page_table` 与 `cu_seqlens_k` **互斥**
- 物理页数 `Np ≤ INT32_MAX + 1`（保证页号可用 int32 表示）

选型约束（[mla_metadata.h#L116-L119](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h)）：

- `heads % kvHeads == 0`
- `valueDim ∈ {128, 512}`
- **`valueDim == 512` 时 `kvHeads` 必须 = 1**
- `margin < physicalCores`（至少留一个物理 AIC）
- `(layout >= 3) != bool(pageSize)` 必须成立

---

## 5. Tile 几何（自动推导，不由用户传入）

来源 [mla_metadata.h#L140-L163](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h)：

| | compressed (Dv = 512) | prefill (Dv = 128) |
|---|---|---|
| `mBaseSize` | **256**（PA 可缩到 128） | 由 `prefillN` 对齐后反推：512 / 256 / 128 / 64 / 32 |
| `s2BaseSize` | **512**（受 pageSize 对齐裁到 ≤ 512） | `prefillN`：Sq ≤ 16 时按 `g` 取 8192 / 4096 / 2048，否则 512 |

硬校验：Dv = 512 时 `m ∈ [128, 256]` 且 `n ≤ 512`；Dv = 128 时必须精确等于推导值
（[L161-L163](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h)）。

---

## 6. dtype 与其他

| 项 | 规格 | 依据 |
|---|---|---|
| dtype | **仅 FP16 / BF16**，q/k/v/qv 必须一致 | [mla_validation.h#L107-L108](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h) |
| INT8 / FP8 | 拒绝 | [mla_validation.h#L104-L106](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h) |
| `q/k/v_descale` | 必须全为 None | [mla_forward.cpp#L417](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) |
| 最后一维 | 必须连续（`stride(-1) == 1`）；外层 stride 非负 | [mla_validation.h#L113](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h) |
| `num_splits` | 1，或 0..256 | [mla_forward.cpp#L412](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) |
| gradient | forward-only；MLA 仅实现前向，未提供反向 | — |
| 设备 | Ascend 910B / 910C（含 910_93） | [mla_forward.cpp#L56](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) |
| `softmax_scale` | 必须**显式传入**，不按 headdim 自动推导 | 见 §9 口径一致性 |

---

## 7. 典型形状样例

### compressed + BSND + 分页 decode

```
B=2, Sq=1, H=128, Hkv=1, P=64, max_pages=16

q          (2, 1, 128, 64)      fp16
qv         (2, 1, 128, 512)     fp16
k_cache    (1024, 64, 1, 64)    fp16      # Np = 1024 物理页
v_cache    (1024, 64, 1, 512)   fp16
page_table (2, 16)              int32
cache_seqlens (2,)              int32     # ≤ 16 × 64 = 1024

→ 内部 layout = 3 (BSND PA)
→ 逻辑容量 = 1024 ≤ 16384 ✓
→ out (2, 1, 128, 512)   lse (2, 128, 1) fp32
```

---

## 8. daoailab（NVIDIA flash-attention）对照

### 8.1 两套独立 MLA 技术栈

| | FA3 (Hopper) | FA4 (Blackwell CuTe) |
|---|---|---|
| 位置 | `daoailab/hopper/flash_api.cpp` | `daoailab/flash_attn/cute/interface.py` |
| 接口 | C++ `mha_fwd` + Python 包 | 纯 Python `flash_attn_func` 系列 |
| 形态判定 | `head_size_v != head_size` 走 diff-hdim 前向 | `is_deepseek_shape` / `is_deepseek_mla_absorbed_shape` |
| 反向 | 通用反向 | MLA 专用反向**只对 sparse 存在** |

### 8.2 形状合法性判据（源头）

FA3 `daoailab/hopper/flash_api.cpp:782-786`：

```
head_size_v != head_size 时只允许：
  (head_size ∈ (128, 192] 且 head_size_v ∈ (96, 128])   → 覆盖 192/128 prefill
  或 (head_size <= 64 且 head_size_v <= 512)            → 覆盖 64/512 absorbed
```

FA3 `qv` 额外约束 `daoailab/hopper/flash_api.cpp:1028-1042`：

```
head_size <= 64          # Dq 必须 ≤ 64
head_size_v >= 256       # Dv 必须 ≥ 256
dtype ∈ {fp16, bf16}     # 不支持 FP8
arch == 90               # 仅 Hopper
qv 形状 = (B, Sq, H, Dv) 或 varlen 下 (Tq, H, Dv)
```

FA4 `daoailab/flash_attn/cute/interface.py:126-127`：

```python
is_deepseek_shape              = (head_dim == 192 and head_dim_v == 128)
is_deepseek_mla_absorbed_shape = (head_dim == 64 or head_dim == head_dim_v) and head_dim_v == 512
```

`head_dim == head_dim_v` 这一支是给 **no-rope 变体**（`q = None`，Dq = Dv = 512）准备的。
注意这两个标志**只用于校验 head_dim 组合合法性**（`interface.py:136`），**并不触发 MLA kernel**。

### 8.3 absorbed / prefill 形状

**核心特征：V（latent 缓存）一物二用** —— 既是第二段 score 的 K，又是 PV 的 V，
因此**输出维 = latent 维 = 512**。

absorbed / decode（FA3 BSND）：

| 张量 | shape | 说明 |
|---|---|---|
| `q` | `(B, Sq, H, 64)` | rope query |
| `qv` | `(B, Sq, H, 512)` | absorbed nope query（预乘 W_UK） |
| `k_cache` | `(B, Sk, 1, 64)` | rope k，**Hkv 恒 = 1** |
| `v_cache` | `(B, Sk, 1, 512)` | latent c_kv（**不是真正的 V**） |
| `out` | `(B, Sq, H, 512)` | = latent 维，需再乘 W_UV |
| `softmax_lse` | `(B, H, Sq)` | |

默认 `softmax_scale = 1/√(64+512) = 1/√576`（`interface.py:760-764`）。
FA3 入口**不只有** `flash_attn_with_kvcache`：`flash_attn_func` 与 `flash_attn_varlen_func` 也都接受 `qv`。

prefill（未吸收）：

| 张量 | shape |
|---|---|
| `q` | `(B, Sq, H, 192)` = 128(nope) + 64(rope) |
| `k` | `(B, Sk, Hkv, 192)` |
| `v` | `(B, Sk, Hkv, 128)` |
| `qv` | **None** |
| `out` | `(B, Sq, H, 128)` |

`Hkv` 可 > 1，唯一约束是 `H % Hkv == 0`。**走通用 diff-hdim 前向，不是 MLA 专用 kernel。**

### 8.4 变体

| 变体 | 规格 |
|---|---|
| **no-rope（FA4 独有）** | `q = None`，只用 `qv`；`q_shape = qv.shape`（`interface.py:665`），`head_dim` 取自 qv = 512，`scale = 1/√512`；走 `(q=qv, k=v, v=v)` shared_kv 路由 |
| **varlen** | 走 `flash_attn_varlen_func`：去掉 batch 维，换 `cu_seqlens_q` + `seqused_k (B,) int32`（`flash_attn_with_kvcache` 只接受 BSND 形状的 q/qv） |
| **paged** | `k_cache (Np, P, 1, 64)` / `v_cache (Np, P, 1, 512)` / `page_table (B, max_pages) int32` |
| **sparse（FA4）** | `gather_kv_indices`，**仅 qv/absorbed 路径生效**；`num_splits > 1` 抛错（`interface.py:540-544`）、与 `page_table` 互斥、末维须 `%128 == 0` 且 dtype int32 |
| **lse 布局翻转** | 带 qv → `(B, Sq, H)`；不带 qv → `(B, H, Sq)`（`interface.py:776-780`） |

---

## 9. 三方差异总表

| 维度 | daoailab | 本仓（flash-attn-npu 3） |
|---|---|---|
| absorbed Dq / Dv | 64 / 512 | 64 / 512 ✅ 一致 |
| prefill Dq / Dv | 192 / 128 | 192 / 128 ✅ 一致 |
| absorbed Hkv | 恒 1（实务必然） | **强制 `kvHeads == 1`**（硬校验） |
| 576 拼接布局 | 不支持，必须拆 | 不支持，必须拆 ✅ 一致 |
| no-rope（`q = None`） | FA4 支持 | ❌ 不支持 |
| top-k 稀疏 `gather_kv_indices` | FA4 支持 | ❌ 不支持 |
| FP8 / INT8 | FA3 `qv` 拒 FP8；FA4 支持 fp8 | 全拒 |
| lse 布局 | 随 `qv` 翻转 | 固定 `[B, Hq, Sq]` / `[Hq, Tq]` |
| MLA 专用反向 | sparse 有 | 无（forward-only） |
| `Dq == Dv` 的处置 | FA4 视为 no-rope MLA 放行 | **判 false 交回普通 FA3** |
| dtype | FP16 / BF16（+ FA4 fp8） | 仅 FP16 / BF16 |
| 分核 / 调度 | 复用 GPU 通用 diff-hdim 前向 | 独立 MLA metadata + 三级分核 + 专用 kernel |

**最本质的一致点**：两边边界口径都是「Dk ≠ Dv 或带 qv ⇒ 可能进 MLA」，
absorbed / prefill 的形状签名完全相同。

**最本质的差异**：daoailab 把 MLA 当成**通用 head-dim 前向的一个特例**去复用；
本仓**另起一套 MLA 专用管线**，代价是不支持 no-rope / 稀疏 / FP8。

**口径一致性提醒**：`softmax_scale` 必须由调用方显式传入。若按 headdim 自动推导，
576 拼接布局会得到 `1/√576`，与 192 口径的 `1/√192` 相差约 1.73 倍。
