# 性能评估结果

- **Operator**: perf_bnsd_vs_cann
- **Device**: npu:4 (source=cli)
- **Warmup**: 3
- **Repeats**: 1
- **Seed**: 0
- **Timing method**: msprof.op_summary.Task_Duration

## 性能对比

| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |
| ---- | ----- | ----- | ------------- | -------- | -------------- |
| 0 | [1, 64, 4096, 128] | float16 | 4451.03 | 3669.64 | 0.824 |

## 全量汇总

| 指标 | 值 |
| ---- | -- |
| 用例数 | 1 |
| 平均加速比（>1 表示自定义算子更快） | 0.824 |
| 标杆几何平均耗时 (us) | 3669.64 |
| 自定义算子几何平均耗时 (us) | 4451.03 |
| 自定义算子更优（比值>1） | 0 |
| 标杆更优（比值<1） | 1 |

### 按数据类型汇总

| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |
| ----- | ------ | ------------------- | ------------- | -------- |
| float16 | 1 | 0.824 | 0 | 1 |

## 简短分析

- 平均加速比 0.824 小于 1，标杆路径整体更优。
- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。

## 深度瓶颈分析

如需进一步分析性能瓶颈（各流水线利用率、核间负载均衡、主 Bound 判定），可运行：
```bash
python3 ${SKILL_PATH}/scripts/msprof_perf_summary.py ./PROF_GROUP_* perf_bnsd_vs_cann
```

或参考 `ops-profiling/references/optimization_quickref.md` 获取优化建议。
