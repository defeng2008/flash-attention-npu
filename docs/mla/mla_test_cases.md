# FA3 / FA4 MLA 测试用例

> 环境变量：`ASCEND_RT_VISIBLE_DEVICES=0`；pytest 必须带 `-p no:randomly`（存在已知的顺序相关抖动）。
> 精度口径：bf16，输出相对 L2 < 5e-3。

## 1. 测试文件

| 文件 | 覆盖 | 用例数 |
| --- | --- | --- |
| `tests/test_flash_attn_npu_v3_mla.py` | V3 接口（func / varlen_func / with_kvcache） | 11 |
| `tests/test_flash_attn_npu_v4_mla.py` | V4 接口（func / varlen_func，decode 走 page_table/seqused_k） | 10 |
| `tests/test_flash_attn_npu_mla_consistency.py` | 训推一致性、FP64 golden、scale 口径、BNSD 布局、拒绝路径 | 16 |
| `tests/perf/bench_mla_vs_fia.py` | 性能（非 pytest，独立脚本） | — |
| `tests/test_flash_attn_npu_v3.py` / `_v4.py`（既有） | 非 MLA 零回归 | 589 / 412 |

参考实现统一在 CPU 上用 FP32 手写：

- prefill：`scores = scale · q @ kᵀ`，`out = softmax(scores) @ v`（支持 causal 与 GQA）。
- decode：`scores = scale · (qv @ v_latentᵀ + q_rope @ k_ropeᵀ)`，`out = softmax(scores) @ v_latent`。

BNSD 用例统一用 `_to_bnsd_view(x) = x.transpose(1, 2).contiguous().transpose(1, 2)`，即「连续 BNSD 存储 + 非连续 BSND 视图」，与调用方 `contiguous_bnsd.transpose(1, 2)` 的持有形态一致。注意 `Sq == 1` 时该视图本身连续，不带 stride 证据、判不出 BNSD（此时 BNSD 与 BSND 是同一块内存），因此 BNSD 用例的 `Sq` 必须 > 1。

## 2. V3 用例（`test_flash_attn_npu_v3_mla.py`）

| 用例 | 输入形状 | 场景 | 断言 |
| --- | --- | --- | --- |
| `test_mla_prefill_bsnd_func` | q/k `(1,198,32,192)`，v `(1,198,32,128)`，causal | 用户 prefill 场景 | out `(1,198,32,128)`，相对 L2 < 5e-3 |
| `test_mla_prefill_bsnd_func_default_scale` | q/k `(2,64,8,192)`，v `(2,64,8,128)` | 缺省 scale，期望 `192 ** -0.5` | 相对 L2 < 5e-3 |
| `test_mla_prefill_tnd_varlen` | `seq_lens=[198,130,7]`，heads 16，`cu_seqlens` 在 NPU | TND varlen + causal | out `(total,16,128)`，相对 L2 < 5e-3 |
| `test_mla_prefill_gqa` | q `(1,96,16,192)`，k `(1,96,4,192)`，v `(1,96,4,128)` | GQA + Dq≠Dv | 相对 L2 < 5e-3 |
| `test_mla_decode_bsnd_no_paged` | q `(1,1,32,64)`，qv `(1,1,32,512)`，cache `(1,1025,1,64/512)` | 非分页 absorbed decode | out `(1,1,32,512)`，相对 L2 < 5e-3 |
| `test_mla_decode_bsnd_default_scale` | heads 16，kv_len 512 | 缺省 scale，期望 `576 ** -0.5` | 相对 L2 < 5e-3 |
| `test_mla_decode_bsnd_paged` | kv_len 1025，page_size 64，17 页 | 分页 absorbed decode | 相对 L2 < 5e-3 |
| `test_mla_decode_rejects_append_kv` | 同时传 `k`/`v`（append-KV）与 `qv` | 拒绝路径 | `RuntimeError` |
| `test_mla_prefill_bnsd` | q/k `(2,64,8,192)`，v `(2,64,8,128)`，三者均为 BNSD 视图 | BNSD prefill | out `(2,64,8,128)`，相对 L2 < 5e-3 |
| `test_mla_prefill_bnsd_gqa` | q `(2,96,16,192)`，k/v `(2,96,4,·)`，BNSD 视图 | BNSD + GQA | 相对 L2 < 5e-3 |
| `test_mla_decode_bnsd_cross_batch` | B=2，Sq=4，q `(2,4,8,64)`、qv `(2,4,8,512)`、cache `(2,64,1,·)`，BNSD 视图 | BNSD absorbed decode，锁定 `gmOffsetQv` 的 batch 基址 | out `(2,4,8,512)`，逐 batch 相对 L2 < 5e-3 |

## 3. V4 用例（`test_flash_attn_npu_v4_mla.py`）

V4 没有 `flash_attn_with_kvcache`，decode 统一走 `flash_attn_func` / `flash_attn_varlen_func`。

| 用例 | 输入形状 | 场景 | 断言 |
| --- | --- | --- | --- |
| `test_mla_prefill_bsnd_func` | 同 V3 用户 prefill 场景 | prefill | 相对 L2 < 5e-3 |
| `test_mla_prefill_bsnd_func_default_scale` | 同 V3 | 缺省 scale = `192 ** -0.5` | 相对 L2 < 5e-3 |
| `test_mla_prefill_tnd_varlen` | 同 V3（`cu_seqlens` 须在 NPU） | TND varlen | 相对 L2 < 5e-3 |
| `test_mla_prefill_gqa` | 同 V3 | GQA | 相对 L2 < 5e-3 |
| `test_mla_decode_bsnd_no_paged` | q `(1,1,32,64)` + qv `(1,1,32,512)`，cache `(1,1025,1,·)` | 非分页 decode | 相对 L2 < 5e-3 |
| `test_mla_decode_bsnd_default_scale` | heads 16 | 缺省 scale = `576 ** -0.5` | 相对 L2 < 5e-3 |
| `test_mla_decode_paged_varlen` | 1025 tokens / page_size 64 / 17 页，TND + `seqused_k` + `page_table` | 分页 varlen decode | 相对 L2 < 5e-3 |
| `test_mla_prefill_bnsd` | q/k `(2,64,8,192)`，v `(2,64,8,128)`，三者均为 BNSD 视图 | BNSD prefill | out `(2,64,8,128)`，相对 L2 < 5e-3 |
| `test_mla_prefill_bnsd_gqa` | q `(2,96,16,192)`，k/v `(2,96,4,·)`，BNSD 视图 | BNSD + GQA | 相对 L2 < 5e-3 |
| `test_mla_decode_bnsd_cross_batch` | B=2，Sq=4，q/qv/cache 全为 BNSD 视图，decode 走 `flash_attn_func` | BNSD absorbed decode | out `(2,4,8,512)`，逐 batch 相对 L2 < 5e-3 |

## 4. 一致性与精度（`test_flash_attn_npu_mla_consistency.py`）

公共常量：`LATENT=512, NOPE=128, ROPE=64, HEADS=8, SEQLEN=197`；`BACKENDS=["flash_attn_npu_3","flash_attn_npu_4"]`，除 `gather_kv_indices` 与 layout 用例只测 v4 外都双后端参数化。

| 用例 | 目的 | 断言 |
| --- | --- | --- |
| `test_mla_prefill_decode_consistency` | 同一 latent 的 absorbed decode 输出经 `W_uvᵀ` 投影后应等于 prefill 输出 | 相对 L2 < 5e-3 |
| `test_mla_prefill_matches_fp64_golden` | prefill 对独立 FP64 CPU golden | 相对 L2 < 5e-3 |
| `test_mla_decode_matches_fp64_golden` | decode 对独立 FP64 CPU golden | 相对 L2 < 5e-3 |
| `test_mla_prefill_default_scale_is_192` | prefill 缺省 scale 必须是 `192 ** -0.5` | 相对 L2 < 5e-3 |
| `test_mla_decode_default_scale_is_576` | absorbed 缺省 scale 必须是 `576 ** -0.5`（**不是** 192 口径） | 相对 L2 < 5e-3 |
| `test_mla_accepts_bnsd` | BNSD 视图输入与同值 BSND 调用的结果必须一致（布局寻址等价性） | 相对 L2 < 5e-3 |
| `test_mla_backward_unsupported` | MLA 仅前向，`requires_grad` 反向被拒 | `RuntimeError` 或 `ValueError` |
| `test_mla_rejects_gather_kv_indices`（仅 v4） | 稀疏 gather 拒绝 | `RuntimeError` |
| `test_mla_rejects_invalid_layout`（仅 v4） | `_flash_attn_forward(layout=7)` 非法 layout 拒绝 | `RuntimeError` |

实测相对 L2（v3/v4 相同）：

| 比较 | 数值 |
| --- | --- |
| prefill vs FP64 golden | 2.283e-3 |
| decode vs FP64 golden | 2.184e-3 |
| prefill vs decode（经 `W_uvᵀ` 投影） | 4.115e-3 |
| decode 缺省 scale（576 口径）vs golden | 2.105e-3 |

## 5. 零回归

| 套件 | 结果 |
| --- | --- |
| `tests/test_flash_attn_npu_v3.py`（非 MLA） | 589 passed |
| `tests/test_flash_attn_npu_v4.py`（非 MLA） | 412 passed |

非 MLA 路径的判据是 `q.last == v.last` 且无 `qv`，此时 `is_mla=false`，`mlaAbsorbed=false`，`embedQK==embed`，所有 MLA 分支均不进入。

## 6. 性能脚本

`tests/perf/bench_mla_vs_fia.py`：

- 场景来自 `tests/perf/mla_user_scenarios.py`（prefill `198/198/32/32/192/192/128` causal；decode `1/1025/32/1/576/576/512` rope64 非 causal，`softmax_scale=192^-0.5`）。
- 同进程交错测量 5 种实现：`fia`、`fa3_nomla`、`fa3_mla`、`fa4_nomla`、`fa4_mla`。
- `WARMUP=10`，`ROUNDS=50`，取中位数。
- 两个门槛：`mla/nomla <= 1.05`（正式门槛）；`mla/fia` 仅作参考（见过程报告 §4）。

FIA 调用注意：prefill 用 `sparse_mode=3` 时**必须**传 2048×2048 上三角 bool mask；decode 用 `sparse_mode=0`，key/value 共用同一 latent 张量。

## 7. 运行方式

```bash
ASCEND_RT_VISIBLE_DEVICES=0 /home/w00811046/miniconda3/envs/cann/bin/python -m pytest \
  tests/test_flash_attn_npu_v3_mla.py tests/test_flash_attn_npu_v4_mla.py \
  tests/test_flash_attn_npu_mla_consistency.py -q -p no:randomly
```

性能：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 /home/w00811046/miniconda3/envs/cann/bin/python \
  tests/perf/bench_mla_vs_fia.py
```
