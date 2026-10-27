# atktest/perf：BSND vs 连续 BNSD 性能对比（msprof）

对比同一份 shape 的 **BSND 输入**与**连续 BNSD 输入**的 kernel 耗时。
口径只有一种：msprof `--quick` 采集 kernel `Task_Duration`，`mean_speedup = BSND / BNSD`，
`< 1` 表示 BNSD 更慢。用法与字段说明见 [msprof/README.md](msprof/README.md)。

- 对比对象是**本仓自身的两种输入布局**，不引入 CANN 标杆。
- 目录自包含：用例生成、模型、脚本都在 `atktest/perf/` 内，运行时不读取 `atktest/memcheck/`
  或 `flash_attention_*_atk_delivery/` 等任何其他目录。

## 接口矩阵

| key | 模块 | 接口 | 910 | 950 |
|---|---|---|---|---|
| `func_v2` | `flash_attn_npu` | `flash_attn_func` | ✓ | — |
| `kvcache_v2` | `flash_attn_npu` | `flash_attn_with_kvcache` | ✓ | — |
| `func_v3` | `flash_attn_npu_3` | `flash_attn_func` | ✓ | ✓ |
| `kvcache_v3` | `flash_attn_npu_3` | `flash_attn_with_kvcache` | ✓ | ✓ |
| `func_v4` | `flash_attn_npu_4` | `flash_attn_func` | ✓ | ✓ |

设备范围与 `atktest/memcheck` 一致：v2 仅 910（非 910 导入即报错），v4 无 `flash_attn_with_kvcache`。

## shape 选取

`shapes.yaml` 每个 shape 组 5 条，head_dim 统一 128（v2 无 192 的既有用例支撑）：

- `func`：覆盖 1k/4k/8k 序列、batch 1/2、MHA / GQA、fp16 / bf16、causal / non-causal。
- `kvcache`：覆盖小 query（`sq=64/128`）与较大 query（`sq=256`）、cache 长 2k~8k、batch 4、GQA / MHA、fp16 / bf16。

只覆盖**非 paged**（`page_table=None`）：这是 950 BNSD 分支的前提。
`cache_seqlens` 直接给定为 `s_alloc`，用例里不传 `k`/`v`（不做 append-KV）、不传 rotary/softcap 等可选参数。

## 布局是怎么"只差布局"的

- BSND：`model.py` 的 `Model` 直接按 `(B, S, H, D)` 传入。
- 连续 BNSD：`model_new_ascendc.py` 的 `ModelNew` 在**首次（预热）调用**里做
  `transpose(1, 2).contiguous()` 物化并缓存，计时区间内只返回 `transpose(1, 2)` 零拷贝视图。
- 两边的输入生成、dtype、causal、softmax_scale、cache_seqlens 完全一致，输出都是连续 BSND。

`transpose(1,2)` 视图满足 950 接口的 stride 判定条件，因此走 BNSD 分支；物化只发生一次，
不计入计时区间。

## 目录结构

```
atktest/perf/
├── README.md                     # 本文件
├── shapes.yaml                   # 每 shape 组 5 条代表性 shape（BSND 语义）
└── msprof/
    ├── gen_cases.py              # shapes.yaml → <key>/<key>_perf_cases.jsonl
    ├── run_msprof.sh             # 逐接口多轮执行 + 平均值汇总
    ├── aggregate.py              # 多轮 performance.pass*.json → 逐用例中位数
    ├── tools/                    # 自带的 msprof profiler 副本（无需外部 skill）
    │   ├── msprof_profile_run.sh
    │   └── msprof_perf_summary.py
    ├── <key>/model.py            # BSND 基准（Ref 侧）
    ├── <key>/model_new_ascendc.py# 连续 BNSD（Asc 侧）
    └── <key>/<key>_perf_cases.jsonl
```

`tools/` 是 `.trae/skills/ops-profiling/scripts/` 中两个脚本的快照，`run_msprof.sh` 默认用它，
checkout 后无需额外安装；要跟随 skill 的新版本时用 `MSPROF_PROFILER=<path>/msprof_profile_run.sh` 覆盖。

运行产生的 `*_perf_cases.jsonl`、`performance*.json`、`perf_report*.md`、`run*.log` 均为派生产物，
已在 `.gitignore` 中排除，不入库。

## 用法

前置：`conda activate ascend`。

```bash
cd atktest/perf/msprof

# 950 三个接口，默认 3 轮取中位数
./run_msprof.sh --device 950 --card 6

# 只跑指定接口
./run_msprof.sh --device 950 --card 6 --keys func_v4,func_v3

# 910 五个接口（需在 910 机器上执行）
./run_msprof.sh --device 910 --card 1

# 只汇总已有 performance.json
./run_msprof.sh --summary
```

## 结果判读

- 每个接口的 `msprof/<key>/performance.json` 是主数据：`mean_speedup` / `mean_ref_us` /
  `mean_asc_us` 为平均值口径，`perf_report.md` 是同一份数据的表格形式。
- 多轮运行保留 `performance.pass<N>.json`，`aggregate.py` 逐用例取中位数后重写
  `performance.json` 与 `perf_report.md`。
- 判据看**平均值**，不看单次最小/最大值：`平均劣化 = mean_asc_us / mean_ref_us - 1`，
  **≤ 5% 判 PASS**。该判据由 `aggregate.py`（`MAX_MEAN_DEGRADE`）写进 `performance.json`
  的 `mean_degrade` / `max_mean_degrade` / `pass`，`./run_msprof.sh --summary` 直接输出
  每个接口与汇总的 PASS/FAIL。

## 限制

- 910 需在 910 机器上执行，本机的用例与脚本按同一清单生成。
- kvcache 只覆盖非 paged，且不使用 `sq=1`：此时 BNSD 的 q 视图 stride 与连续 BSND 相同，
  布局探测会退回 BSND，kernel 报 `v must be contiguous`，因此统一取 `sq >= 64`。
- head_dim 统一 128。
- 达标阈值固定为 `平均劣化 ≤ 5%`（`aggregate.py::MAX_MEAN_DEGRADE`），只做布局间的相对比较。
- 本机 950 为共享机器，其他用户的任务会周期性抢卡。解析器已剔除 msprof 的事件打点行并按
  迭代周期切块，但仍建议多轮取中位数，并对可疑结果用 `op_summary` 的单 kernel 耗时交叉验证。
