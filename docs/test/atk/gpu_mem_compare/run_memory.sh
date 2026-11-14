#!/usr/bin/env bash
# 汇总执行 gpu_mem_compare —— npu(FA3/FA4) vs GPU(daoailab) 峰值显存对比（需要 GPU 主机）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for iface in fa3_func fa3_varlen fa3_kvcache fa4_func fa4_varlen; do
  bash "${HERE}/interfaces/${iface}/run_memory.sh"
done
