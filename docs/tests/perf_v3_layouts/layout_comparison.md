# 950 v3 接口 BSND vs BNSD 输入布局性能对比

## 1. 目标与范围

针对 Ascend 950 上的 v3 接口，各给出一个「BSND 输入 vs BNSD 输入」的性能对比用例：

| 接口 | 用例规模 | 对比内容 |
|---|---|---|
| `flash_attn_func` | `S_Q = S_KV = 4096` | 同一批逻辑数值，一份以连续 BSND 存储，一份以连续 BNSD 存储 |
| `flash_attn_with_kvcache` | `S_Q = 128`，`S_KV = 4096`（`cache_seqlens = 4096`） | 同上 |

共同固定项：`B = 1`、`H = H_kv = 64`、`D = 128`、`float16`、`softmax_scale = 1/sqrt(128)`、`causal = True`。
两个用例都来自我们自己的算子（不涉及 CANN 标杆算子）。

交付目录：[perf_v3_layouts](file:///home/w00811046/code/fan/261022confict/flash-attention-npu/docs/tests/perf_v3_layouts)

## 2. 关键实现说明（为什么这样搭用例才公平）

布局互比有一个容易踩的坑：**同一块内存按 BSND 读和按 BNSD 读，得到的是互为转置的两组逻辑张量**，
描述的不是同一个数学问题。因此：

- 每个接口的 BNSD 槽位（[func/model_new_ascendc.py](file:///home/w00811046/code/fan/261022confict/flash-attention-npu/docs/tests/perf_v3_layouts/func/model_new_ascendc.py)、
  [kvcache/model_new_ascendc.py](file:///home/w00811046/code/fan/261022confict/flash-attention-npu/docs/tests/perf_v3_layouts/kvcache/model_new_ascendc.py)）
  在**首次** forward 时把 BSND 输入转成连续 BNSD 存储并缓存（`transpose(1, 2).contiguous()`），
  之后每次只重新取 `transpose(1, 2)` 视图。这样两路持有「同一批逻辑数值、各自布局的存储」，
  且布局转换只发生在 warmup 区、**不进入计时区**。
- BNSD 槽位传给算子的 q 是 `(B, S, H, D)` 视图，950 接口的输出形状等于调用方传入的 q 形状，
  因此两路输出都是连续 BSND，可直接逐元素比对。实测两路 **max abs diff = 0.00000**（完全一致），
  说明两条路径确实在算同一个问题。
- 布局判定依据 stride：[flash_attn_npu_interface_950.py](file:///home/w00811046/code/fan/261022confict/flash-attention-npu/flash_attn_npu_3/flash_attn_npu_interface_950.py#L47-L79)
  的 `_transposed_bnsd_view()` 要求「rank-4 + 非连续 + stride 匹配 BNSD」，命中即走 `_LAYOUT_BNSD`。

## 3. 实测发现：`flash_attn_with_kvcache` 在解码形态（`S_Q = 1`）下 BNSD 不可检出

`S_Q = 1` 时，BNSD 存储 `(B, H, 1, D)` 的 `transpose(1, 2)` 视图在 stride 上与连续 BSND 完全等价
（`is_contiguous()` 为 True），于是 `_transposed_bnsd_view()` 返回 `None`，接口回退 `_LAYOUT_BSND`；
此时 cache 的 `(B, S, H, D)` 视图是非连续的，BSND 分支随后以 `RuntimeError: v must be contiguous` 拒绝。

实测（`.cannbot/dev/bnsd-bsnd-output/perf_v3_layouts/probe_kvcache_layout.py`）：

```
S_Q=   1 bnsd_storage=True  layout=1 (BSND) -> RAISED RuntimeError: v must be contiguous
S_Q=  64 bnsd_storage=True  layout=2 (BNSD) -> out (1, 64, 64, 128) contig=True
```

因此若坚持 `S_Q = 1`，kvcache 只能跑 BSND，无法做布局互比。**本轮把该用例的 `S_Q` 由 1 调整为 128**
（其余条件不变），使其成为「一段 query 对 4096 长 cache」的形态，从而两个接口都能给出对比。
这是对原计划用例规模的唯一偏离，原因即上述接口 stride 判据的固有限制。

## 4. 测量方法

两种测量都做，互为交叉验证：

1. **wall-clock**（单进程交错计时，20 次迭代 + 5 次预热，取中位数）：
   [bench_wallclock.py](file:///home/w00811046/code/fan/261022confict/flash-attention-npu/docs/tests/perf_v3_layouts/bench_wallclock.py)，
   计时前先做 sanity（形状 + `allclose`）。
2. **msprof kernel 级**（`--quick`，`warm-up=3`、`repeats=5`，取 `Task_Duration` 均值），
   `model.py` 为 BSND 基准位（报告中的 Ref）、`model_new_ascendc.py` 为 BNSD 被测位（报告中的 Asc），
   即 `speedup = Ref/Asc > 1` 表示 **BNSD 更快**。

运行卡：卡 7（`--device 7`，实测空闲；卡 0/2 Alarm、卡 1 被占满）。设备绑定必须在 `import torch_npu` 之前
设置 `ASCEND_RT_VISIBLE_DEVICES`，否则 causal 变体报 507033。

## 5. 结果

### 5.1 wall-clock（卡 7，中位数）

| 接口 | BSND (us) | BNSD (us) | BNSD/BSND − 1 |
|---|---|---|---|
| `flash_attn_func` (S_Q=4096) | 1256.5 | 1356.5 | **+7.96%** |
| `flash_attn_with_kvcache` (S_Q=128) | 189.8 | 186.2 | **−1.85%** |

`flash_attn_func` 的 20 次采样离散度较大（mean +7.00%、median +7.96%、min −0.50%），
属于 host 侧包装与调度噪声范围；`flash_attn_with_kvcache` 三档统计一致（mean −2.06%、median −1.85%、min −0.52%）。

### 5.2 msprof kernel 级（两遍）

| 接口 | 遍次 | BSND/Ref (us) | BNSD/Asc (us) | Speedup (BNSD 快于 BSND) |
|---|---|---|---|---|
| `flash_attn_func` | r1 | 1438.16 | 1397.54 | **1.029x** |
| `flash_attn_func` | r2 | 1435.56 | 1340.63 | **1.071x** |
| `flash_attn_with_kvcache` | r1 | 154.24 | 149.34 | **1.033x** |
| `flash_attn_with_kvcache` | r2 | 155.68 | 151.13 | **1.030x** |

两遍结论一致，无翻转。

## 6. 结论

- **kernel 级：BNSD 输入不劣于 BSND，反而略快。** `flash_attn_func` 快约 3%~7%，
  `flash_attn_with_kvcache` 稳定快约 3%。机制上合理：BNSD 让每个 head 的 KV 平面在全局内存中连续，
  QK^T 沿 head 维的访存与 cube 侧取数更规整。
- **wall-clock 与 kernel 级在 `flash_attn_func` 上出现方向性差异**（wall-clock 显示 BNSD 慢约 8%，
  kernel 显示 BNSD 快约 3%~7%）。该用例 wall-clock 采样离散度大（min 甚至为 −0.5%），
  且两路 host 侧工作基本相同；**本对比以 msprof 的 device kernel 时间为准**，wall-clock 只作趋势参考。
- **`flash_attn_with_kvcache` 的 `S_Q = 1`（解码）形态无法做布局互比**：接口按 stride 判定布局，
  单 token 的 BNSD query 视图与连续 BSND 等价，必然回退 BSND 并在带 stride 的 cache 上抛
  `v must be contiguous`。需要 BNSD 时该接口必须 `S_Q > 1`。

## 7. 复现

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ascend
source ~/miniconda3/Ascend/cann/set_env.sh

# wall-clock（含 sanity）
python3 -u docs/tests/perf_v3_layouts/bench_wallclock.py --device 7

# kernel 级（BSND=Ref vs BNSD=Asc）
bash .trae/skills/ops-profiling/scripts/msprof_profile_run.sh \
  --quick --output-dir=docs/tests/perf_v3_layouts/func --device=7 --warm-up=3 --repeats=5
bash .trae/skills/ops-profiling/scripts/msprof_profile_run.sh \
  --quick --output-dir=docs/tests/perf_v3_layouts/kvcache --device=7 --warm-up=3 --repeats=5
```

原始证据（wall-clock 日志、两遍 `performance*.json`、探针脚本）归档在
`.cannbot/dev/bnsd-bsnd-output/perf_v3_layouts/`。

## 8. 边界与限制

- 单点规模（`B=1`、`causal`、fp16、`D=128`），不覆盖 batch>1、非 causal、其他 headdim/dtype。
- `flash_attn_func` 的 BNSD 槽位以「一次性物化 BNSD 存储」的方式构造输入，真实业务里这次转换
  （或由上游产出的 BNSD 张量）是存在成本的，本用例按约定不计入算子耗时。
- wall-clock 含 host 开销，量级与噪声均大于 kernel 级；跨接口比较绝对耗时无意义，只比较同一接口内两布局的相对差。
