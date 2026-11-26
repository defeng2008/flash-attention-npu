#!/usr/bin/env bash
# 汇总执行 gpu_mem_compare —— npu(FA3/FA4) vs CANN 9.2 FIA(npu) 峰值显存对比（候选与标杆同为 NPU）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for iface in fa3_func fa3_varlen fa3_kvcache fa4_func fa4_varlen; do
  script="${HERE}/interfaces/${iface}/run_memory.sh"
  # 子脚本若被传输/编辑工具转成 CRLF，先就地修正，否则 bash 会报
  # "/usr/bin/env: 'bash\r': No such file or directory"
  if grep -q $'\r' "${script}" 2>/dev/null; then sed -i 's/\r$//' "${script}"; fi
  bash "${script}"
done
