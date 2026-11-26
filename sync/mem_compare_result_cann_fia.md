# MLA 显存对比结果 — npu(FA3/FA4) vs CANN 9.2 FIA

- **日期**：2026-11-26
- **环境**：Ascend910B3，CANN 9.2.0，`torch_npu`（conda env `cann`）
- **套件**：`flash-attention-npu/docs/test/atk/gpu_mem_compare`
- **候选**：`flash_attn_npu_3` / `flash_attn_npu_4`（NPU，device 0）
- **标杆**：`torch_npu.npu_fused_infer_attention_score`（CANN FIA，NPU，device 0，节点名 `fia_mla`）
- **判据**：候选峰值分配显存 / 标杆峰值分配显存 ≤ **1.05**（yaml `mem: 1.1` 不在白名单 `[1.05,1.5,2.0]`，回退 1.05）；或当 (输入+输出) ≥ 500MiB 时 `|候选 − 标杆| ≤ L2` 亦可
- **执行**：`atk task --task memory_device --bm_device npu --single_process`

## 1. 总览

| 接口 | 被测接口 | 通过 | 用例数 | 候选显存 (MB) | FIA 标杆显存 (MB) | 候选/FIA | 内存是否达标 |
|---|---|---|---|---|---|---|---|
| `fa3_func` | `flash_attn_npu_3.flash_attn_func` | 12/12 | 12 | 621–699 | 826–1160 | 0.602–0.751 | Pass |
| `fa3_varlen` | `flash_attn_npu_3.flash_attn_varlen_func` | 12/12 | 12 | 619–699 | 820–1160 | 0.602–0.755 | Pass |
| `fa3_kvcache` | `flash_attn_npu_3.flash_attn_with_kvcache` | 12/12 | 12 | 621–699 | 822–1160 | 0.602–0.755 | Pass |
| `fa4_func` | `flash_attn_npu_4.flash_attn_func` | 12/12 | 12 | 658–702 | 1080–1160 | 0.605–0.611 | Pass |
| `fa4_varlen` | `flash_attn_npu_4.flash_attn_varlen_func` | 12/12 | 12 | 618–702 | 1008–1160 | 0.605–0.614 | Pass |
| **合计** | | **60/60** | **60** | | | | **通过率 100%** |

> 表中「分配显存」为 ATK 报告的 `npu_candidate_分配内存（MB）` / `npu_fia_mla_分配内存（MB）`。
> 全部 60 条用例的候选/FIA 比值均 < 1，未触及 1.05 门限，也无失败用例。

## 2. 逐用例明细

### 2.1 `fa3_func`（flash_attn_npu_3.flash_attn_func，12/12 通过）

| # | 形态 | dtype | causal | 输入形状 | 候选 (MB) | FIA (MB) | 候选/FIA | 结果 |
|---|---|---|---|---|---|---|---|---|
| 0 | decode | fp16 | True | `q[2, 4096, 32, 64]  k[2, 7680, 1, 64]  v[2, 7680, 1, 512]  qv[2, 4096, 32, 512]` | 620.5 | 826.0 | 0.751 | 通过 |
| 1 | decode | fp16 | True | `q[2, 4096, 32, 64]  k[2, 8192, 1, 64]  v[2, 8192, 1, 512]  qv[2, 4096, 32, 512]` | 620.5 | 826.0 | 0.751 | 通过 |
| 2 | decode | fp16 | True | `q[2, 4096, 32, 64]  k[2, 8192, 1, 64]  v[2, 8192, 1, 512]  qv[2, 4096, 32, 512]` | 620.5 | 826.0 | 0.751 | 通过 |
| 3 | decode | fp16 | True | `q[2, 4096, 32, 64]  k[2, 8192, 1, 64]  v[2, 8192, 1, 512]  qv[2, 4096, 32, 512]` | 620.5 | 826.0 | 0.751 | 通过 |
| 4 | prefill | bf16 | True | `q[2, 7680, 32, 192]  k[2, 7680, 32, 192]  v[2, 7680, 32, 128]` | 658.5 | 1088.0 | 0.605 | 通过 |
| 5 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k[2, 7680, 32, 192]  v[2, 7680, 32, 128]` | 658.5 | 1084.0 | 0.607 | 通过 |
| 6 | prefill | fp16 | False | `q[2, 7680, 32, 192]  k[2, 7680, 32, 192]  v[2, 7680, 32, 128]` | 658.5 | 1084.0 | 0.607 | 通过 |
| 7 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k[2, 7680, 32, 192]  v[2, 7680, 32, 128]` | 658.5 | 1084.0 | 0.607 | 通过 |
| 8 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 678.5 | 1116.0 | 0.608 | 通过 |
| 9 | prefill | fp16 | True | `q[2, 7680, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 678.5 | 1120.0 | 0.606 | 通过 |
| 10 | prefill | bf16 | False | `q[2, 8192, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 698.5 | 1156.0 | 0.604 | 通过 |
| 11 | prefill | bf16 | True | `q[2, 8192, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 698.5 | 1160.0 | 0.602 | 通过 |

### 2.2 `fa3_varlen`（flash_attn_npu_3.flash_attn_varlen_func，12/12 通过）

| # | 形态 | dtype | causal | 输入形状 | 候选 (MB) | FIA (MB) | 候选/FIA | 结果 |
|---|---|---|---|---|---|---|---|---|
| 0 | decode | fp16 | True | `q[8192, 32, 64]  k[14336, 1, 64]  v[14336, 1, 512]  qv[8192, 32, 512]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 618.5 | 824.0 | 0.751 | 通过 |
| 1 | decode | bf16 | True | `q[8192, 32, 64]  k[14336, 1, 64]  v[14336, 1, 512]  qv[8192, 32, 512]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 618.5 | 824.0 | 0.751 | 通过 |
| 2 | decode | fp16 | False | `q[8192, 32, 64]  k[14336, 1, 64]  v[14336, 1, 512]  qv[8192, 32, 512]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 618.5 | 820.0 | 0.754 | 通过 |
| 3 | decode | bf16 | False | `q[8192, 32, 64]  k[16384, 1, 64]  v[16384, 1, 512]  qv[8192, 32, 512]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 620.5 | 822.0 | 0.755 | 通过 |
| 4 | decode | fp16 | True | `q[8192, 32, 64]  k[16384, 1, 64]  v[16384, 1, 512]  qv[8192, 32, 512]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 620.5 | 826.0 | 0.751 | 通过 |
| 5 | decode | bf16 | True | `q[8192, 32, 64]  k[16384, 1, 64]  v[16384, 1, 512]  qv[8192, 32, 512]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 620.5 | 826.0 | 0.751 | 通过 |
| 6 | prefill | bf16 | False | `q[14336, 32, 192]  k[14336, 32, 192]  v[14336, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 618.5 | 1012.0 | 0.611 | 通过 |
| 7 | prefill | fp16 | False | `q[14336, 32, 192]  k[14336, 32, 192]  v[14336, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 618.5 | 1012.0 | 0.611 | 通过 |
| 8 | prefill | bf16 | True | `q[14336, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 658.5 | 1080.0 | 0.610 | 通过 |
| 9 | prefill | fp16 | False | `q[14336, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 658.5 | 1076.0 | 0.612 | 通过 |
| 10 | prefill | bf16 | True | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 698.5 | 1160.0 | 0.602 | 通过 |
| 11 | prefill | fp16 | True | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 698.5 | 1160.0 | 0.602 | 通过 |

### 2.3 `fa3_kvcache`（flash_attn_npu_3.flash_attn_with_kvcache，12/12 通过）

| # | 形态 | dtype | causal | 输入形状 | 候选 (MB) | FIA (MB) | 候选/FIA | 结果 |
|---|---|---|---|---|---|---|---|---|
| 0 | decode | fp16 | True | `q[2, 4096, 32, 64]  k_cache[2, 7680, 1, 64]  v_cache[2, 7680, 1, 512]  qv[2, 4096, 32, 512]` | 620.5 | 826.0 | 0.751 | 通过 |
| 1 | decode | fp16 | True | `q[2, 4096, 32, 64]  k_cache[2, 8192, 1, 64]  v_cache[2, 8192, 1, 512]  qv[2, 4096, 32, 512]` | 620.5 | 826.0 | 0.751 | 通过 |
| 2 | decode | bf16 | False | `q[2, 4096, 32, 64]  k_cache[2, 8192, 1, 64]  v_cache[2, 8192, 1, 512]  qv[2, 4096, 32, 512]` | 620.5 | 822.0 | 0.755 | 通过 |
| 3 | decode | fp16 | False | `q[2, 4096, 32, 64]  k_cache[2, 8192, 1, 64]  v_cache[2, 8192, 1, 512]  qv[2, 4096, 32, 512]` | 620.5 | 822.0 | 0.755 | 通过 |
| 4 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k_cache[2, 7680, 32, 192]  v_cache[2, 7680, 32, 128]` | 658.5 | 1084.0 | 0.607 | 通过 |
| 5 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k_cache[2, 8192, 32, 192]  v_cache[2, 8192, 32, 128]` | 678.5 | 1116.0 | 0.608 | 通过 |
| 6 | prefill | bf16 | True | `q[2, 8192, 32, 192]  k_cache[2, 8192, 32, 192]  v_cache[2, 8192, 32, 128]` | 698.5 | 1160.0 | 0.602 | 通过 |
| 7 | prefill | bf16 | True | `q[2, 8192, 32, 192]  k_cache[2, 8192, 32, 192]  v_cache[2, 8192, 32, 128]` | 698.5 | 1160.0 | 0.602 | 通过 |
| 8 | prefill | fp16 | True | `q[2, 8192, 32, 192]  k_cache[2, 8192, 32, 192]  v_cache[2, 8192, 32, 128]` | 698.5 | 1160.0 | 0.602 | 通过 |
| 9 | prefill | fp16 | False | `q[2, 8192, 32, 192]  k_cache[2, 8192, 32, 192]  v_cache[2, 8192, 32, 128]` | 698.5 | 1156.0 | 0.604 | 通过 |
| 10 | prefill | bf16 | True | `q[2, 8192, 32, 192]  k_cache[2, 8192, 32, 192]  v_cache[2, 8192, 32, 128]` | 698.5 | 1160.0 | 0.602 | 通过 |
| 11 | prefill | fp16 | True | `q[2, 8192, 32, 192]  k_cache[2, 8192, 32, 192]  v_cache[2, 8192, 32, 128]` | 698.5 | 1160.0 | 0.602 | 通过 |

### 2.4 `fa4_func`（flash_attn_npu_4.flash_attn_func，12/12 通过）

| # | 形态 | dtype | causal | 输入形状 | 候选 (MB) | FIA (MB) | 候选/FIA | 结果 |
|---|---|---|---|---|---|---|---|---|
| 0 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k[2, 7680, 32, 192]  v[2, 7680, 32, 128]` | 658.0 | 1080.0 | 0.609 | 通过 |
| 1 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k[2, 7680, 32, 192]  v[2, 7680, 32, 128]` | 658.0 | 1080.0 | 0.609 | 通过 |
| 2 | prefill | fp16 | True | `q[2, 7680, 32, 192]  k[2, 7680, 32, 192]  v[2, 7680, 32, 128]` | 662.0 | 1088.0 | 0.608 | 通过 |
| 3 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k[2, 7680, 32, 192]  v[2, 7680, 32, 128]` | 662.0 | 1084.0 | 0.611 | 通过 |
| 4 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 682.0 | 1116.0 | 0.611 | 通过 |
| 5 | prefill | fp16 | True | `q[2, 7680, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 682.0 | 1120.0 | 0.609 | 通过 |
| 6 | prefill | bf16 | False | `q[2, 7680, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 682.0 | 1116.0 | 0.611 | 通过 |
| 7 | prefill | fp16 | True | `q[2, 8192, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 8 | prefill | bf16 | True | `q[2, 8192, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 9 | prefill | fp16 | False | `q[2, 8192, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 702.0 | 1156.0 | 0.607 | 通过 |
| 10 | prefill | fp16 | True | `q[2, 8192, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 11 | prefill | fp16 | True | `q[2, 8192, 32, 192]  k[2, 8192, 32, 192]  v[2, 8192, 32, 128]` | 702.0 | 1160.0 | 0.605 | 通过 |

### 2.5 `fa4_varlen`（flash_attn_npu_4.flash_attn_varlen_func，12/12 通过）

| # | 形态 | dtype | causal | 输入形状 | 候选 (MB) | FIA (MB) | 候选/FIA | 结果 |
|---|---|---|---|---|---|---|---|---|
| 0 | prefill | fp16 | False | `q[14336, 32, 192]  k[14336, 32, 192]  v[14336, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 618.0 | 1008.0 | 0.613 | 通过 |
| 1 | prefill | bf16 | False | `q[14336, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 658.0 | 1072.0 | 0.614 | 通过 |
| 2 | prefill | fp16 | True | `q[14336, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 662.0 | 1080.0 | 0.613 | 通过 |
| 3 | prefill | fp16 | True | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 4 | prefill | bf16 | False | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 702.0 | 1156.0 | 0.607 | 通过 |
| 5 | prefill | bf16 | True | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 6 | prefill | bf16 | True | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 7 | prefill | fp16 | True | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 8 | prefill | fp16 | True | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 9 | prefill | fp16 | True | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 10 | prefill | bf16 | True | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 702.0 | 1160.0 | 0.605 | 通过 |
| 11 | prefill | bf16 | False | `q[16384, 32, 192]  k[16384, 32, 192]  v[16384, 32, 128]  cu_seqlens_q[2]  cu_seqlens_k[2]` | 702.0 | 1156.0 | 0.607 | 通过 |

## 3. 说明

- 标杆分发按**节点名**进行：节点文件 `nodes_memory_cann.yaml` 里标杆节点名为 `fia_mla`，适配器据此走 FIA 分支（`FIA_NODE_NAME="fia_mla"`）。
- 候选与标杆同为 NPU device 0，故用 `--single_process` 串行执行，避免同卡并发干扰测量。
- `fa4_*` 为 prefill-only（沿用现有用例集合口径）；decode 覆盖见 `single_benchmark_cpu` / `single_benchmark_cann`。
- 运行期反复出现 `perf_compare_tool.py:69 current chip model Unknown is not in the chip model to l2 cache size dict` 警告；因所有用例比值远低于 1.05 主判据、未依赖 L2 豁免，不影响结论。
- 生成器 `docs/test/atk/tools/gen_atk_mla.py` 已同步为 FIA 标杆（重跑不会回退成 GPU 标杆）。

---

数据来源（各接口最新一份 ATK 报告）：
- `fa3_func`：[all_fa3_func_mem_reports_2026-11-26-05-10-24.xlsx](interfaces/fa3_func/atk_output/all_fa3_func_mem_2026-11-26-13-10-22-161051/report/all_fa3_func_mem_reports_2026-11-26-05-10-24.xlsx)
- `fa3_varlen`：[all_fa3_varlen_mem_reports_2026-11-26-05-15-03.xlsx](interfaces/fa3_varlen/atk_output/all_fa3_varlen_mem_2026-11-26-13-15-03-235532/report/all_fa3_varlen_mem_reports_2026-11-26-05-15-03.xlsx)
- `fa3_kvcache`：[all_fa3_kvcache_mem_reports_2026-11-26-04-59-23.xlsx](interfaces/fa3_kvcache/atk_output/all_fa3_kvcache_mem_2026-11-26-12-59-20-756118/report/all_fa3_kvcache_mem_reports_2026-11-26-04-59-23.xlsx)
- `fa4_func`：[all_fa4_func_mem_reports_2026-11-26-05-18-28.xlsx](interfaces/fa4_func/atk_output/all_fa4_func_mem_2026-11-26-13-18-27-953376/report/all_fa4_func_mem_reports_2026-11-26-05-18-28.xlsx)
- `fa4_varlen`：[all_fa4_varlen_mem_reports_2026-11-26-05-22-19.xlsx](interfaces/fa4_varlen/atk_output/all_fa4_varlen_mem_2026-11-26-13-22-18-938666/report/all_fa4_varlen_mem_reports_2026-11-26-05-22-19.xlsx)
