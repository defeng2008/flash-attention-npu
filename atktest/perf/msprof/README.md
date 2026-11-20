# msprof：BSND vs 连续 BNSD 的 kernel 级对比

用 msprof（`--quick`，只取 kernel `Task_Duration`）比较同一份 shape 的两种输入布局：
`model.py` 为 **BSND** 连续输入，`model_new_ascendc.py` 为 **连续 BNSD** 输入。
两者除布局外完全一致（同 shape、同 dtype、同 causal、同 softmax_scale）。

**判定只看平均值**：`mean_speedup = mean(BSND 耗时 / BNSD 耗时)`，`< 1` 表示 BNSD 更慢；
对应的劣化百分比为 `(BNSD 均值 / BSND 均值 - 1) × 100%`。
**达标判据：平均劣化 ≤ 5%**（`aggregate.py::MAX_MEAN_DEGRADE`），`run_msprof.sh --summary`
按该判据逐接口打印 PASS/FAIL 并给出总判定；`perf_report.md` 同步带 `判定` 行。

## 设备与接口矩阵

| 设备 | 接口 key |
|---|---|
| 910 | `func_v2` `kvcache_v2` `func_v3` `kvcache_v3` `func_v4` |
| 950 | `func_v3` `kvcache_v3` `func_v4` |

v2 仅 910（非 910 导入即报错），v4 无 `flash_attn_with_kvcache`，950 不支持 v2。
本机只有 950，910 需在实际 910 机器上执行（用例与脚本按同一清单生成）。

## 目录结构

```
msprof/
├── gen_cases.py                 # shapes.yaml → <key>/<key>_perf_cases.jsonl
├── run_msprof.sh                # 逐接口多轮执行 + 平均口径汇总
├── aggregate.py                 # 多轮 performance.pass*.json → 逐用例中位数
├── tools/                       # msprof profiler 脚本副本（默认使用，无需外部 skill）
├── <key>/model.py               # BSND 基准（msprof 的 Ref 侧）
├── <key>/model_new_ascendc.py   # 连续 BNSD（msprof 的 Asc 侧）
├── <key>/<key>_perf_cases.jsonl # 用例条数与 shape/dtype（派生物，不入库）
├── <key>/performance.pass*.json # 每轮原始结果（派生物，不入库）
└── <key>/performance.json       # 多轮中位数合并后的结果（派生物，不入库）
```

用例 shape 与 `../shapes.yaml` 一致：`func` 5 条、`kvcache` 5 条，head_dim 统一 128。
jsonl 只负责枚举用例；真实输入由 `model.py::get_input_groups()` 按 case 序号生成，
`get_init_inputs()` 提供该 case 的 `causal` 开关。

profiler 默认取本目录的 `tools/msprof_profile_run.sh`（`.trae/skills/ops-profiling/scripts/`
的快照），checkout 后无需安装 skill；要换成别的版本时用 `MSPROF_PROFILER=<path>` 覆盖。

## 用法

```bash
conda activate ascend
source ~/miniconda3/Ascend/cann/set_env.sh

cd atktest/perf/msprof
./run_msprof.sh --device 950 --card 6          # 950 三个接口，默认 3 轮
./run_msprof.sh --device 950 --parallel        # 三接口各占一张卡并行（默认 5 6 7）
./run_msprof.sh --device 950 --parallel --cards "5 6 7"
./run_msprof.sh --device 910 --card 1          # 需在 910 机器上执行
./run_msprof.sh --device 950 --keys func_v3    # 单接口
./run_msprof.sh --device 950 --passes 1        # 只跑 1 轮（快，但共享机器上不稳）
./run_msprof.sh --summary                      # 只汇总已有 performance.json
```

每个接口每轮约 1.5~2 分钟（5 用例 × 两种实现 × 预热 3 + 计时 5），默认 3 轮取中位数。
`--parallel` 时每个接口的输出写进各自的 `<key>/run.log`，父进程只打印开始/结束行。
注意本机 0 卡处于 Alarm 状态，默认并行卡位是 5/6/7。

## 口径说明

- 只有 kernel 时间进判据：主机侧开销（约 40%）和跨进程时钟漂移（±30%）都不计入，
  因此该口径可以判 5% 级别的差异。
- 两种布局走的是**同一个 kernel 的不同模板实例**：BSND 为 `FAInfer...Format1...`，
  连续 BNSD 为 `FAInfer...Format2...`（非因果为 `FAInferDnI...`），block 数相同。
- 共享机器上单轮 `--quick` 结果可能被其他进程的尖峰污染。早期版本存在一个已修的系统性
  伪影：`task_time.csv` 里混有约 100 行 `kernel_name=N/A` 的事件打点行（`DAVID_EVENT_RECORD`
  / `DAVID_EVENT_WAIT` 等，单条可达 1.6s 的首次 JIT 记录），旧解析要求「总行数能被迭代数
  整除」，失败后回退为按行位置分片，导致计时窗口可能多框进整次迭代。实测该项把 func_v4
  因果用例压到 5/6≈0.83，即虚报 −17%。现在解析改为「剔除事件行 + 从尾部识别迭代周期再
  切块」，同一份原始采集回放后 c1/c2/c4 的加速比由 0.859/0.836/0.852 修正为
  0.998/0.997/0.979。仍默认跑 3 轮、逐用例取中位数（`aggregate.py`），并结合 `op_summary`
  的单 kernel 耗时交叉验证。
- BNSD 存储只在首次（预热）调用物化一次并缓存，计时区间内仅 `transpose(1, 2)` 零拷贝视图。
- 每个接口的 `performance.json` 即主数据，`mean_speedup` / `mean_ref_us` /
  `mean_asc_us` 为平均值口径；`perf_report.md` 是同一份数据的表格形式。
- kvcache 不覆盖 paged、不使用 `sq=1`（此时 BNSD 视图 stride 与 BSND 相同，布局探测会退回
  BSND 并报 `v must be contiguous`）。
