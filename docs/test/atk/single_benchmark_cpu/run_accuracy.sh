#!/usr/bin/env bash
# 汇总执行 single_benchmark_cpu —— npu 候选 vs CPU fp32 标杆精度对比
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for iface in fa3_func fa3_varlen fa3_kvcache fa4_func fa4_varlen; do
  bash "${HERE}/interfaces/${iface}/run_accuracy.sh"
done
