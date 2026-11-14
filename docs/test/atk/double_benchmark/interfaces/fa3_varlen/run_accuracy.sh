#!/usr/bin/env bash
# double_benchmark —— npu 候选 vs GPU(daoailab) + CPU 双标杆精度对比
#
# 流程（yaml 是唯一真值，用例全部由 atk case 生成、不再手写）：
#   1) atk case -f <stem>.yaml -p .  ->  result/<stem>/json/all_<stem>.json
#   2) cp 到 <cat>/cases/all_<stem>.json（交付里可直接打开查看）
#   3) atk task -c cases/all_<stem>.json ... 执行并产出精度/显存报告
#   4) 最后补跑用户场景 anchor：cases/user_scenarios_fa3_varlen.json（报告与上一步分开）
#
# 环境约定：conda 环境 cann（已装 atk + atk_private_precision）/ CANN set_env.sh / TMPDIR
set -euo pipefail

export CONDA_ENV="${ATK_CONDA_ENV:-cann}"
export CANN_ROOT="${CANN_ROOT:-/home/w00811046/cann/cann-9.2.0-clean/ascend-toolkit}"
export TMPDIR="${TMPDIR:-/home/w00811046/tmp}"
ATK="${ATK_BIN:-atk}"

# 加载 CANN 运行环境（torch_npu 需要 libhccl.so 等）
if [ -f "${CANN_ROOT}/set_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${CANN_ROOT}/set_env.sh"
else
  echo "[warn] 未找到 ${CANN_ROOT}/set_env.sh，请用 CANN_ROOT 指定含 set_env.sh 的 CANN 目录" >&2
fi
# 已激活环境里有 atk 就直接用，否则经 conda run 进入 CONDA_ENV
if command -v "${ATK}" >/dev/null 2>&1; then
  ATK_CMD=("${ATK}")
else
  ATK_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" "${ATK}")
fi

INTERFACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERFACE_NAME="$(basename "${INTERFACE_DIR}")"
DELIVERY_DIR="$(cd "${INTERFACE_DIR}/../../.." && pwd)"
CASE_DIR="${DELIVERY_DIR}/double_benchmark/cases"
NODES_FILE="nodes_accuracy_gpu.yaml"
SEED="${ATK_SEED:-20261113}"
if [ "multi" = "multi" ]; then _SP=""; else _SP="--single_process"; fi

mkdir -p "${CASE_DIR}"
cd "${INTERFACE_DIR}"

# yaml -> cases/all_<stem>.json
gen_case() {
  local yaml="$1" stem="$2"
  "${ATK_CMD[@]}" case -f "${yaml}" -p . -s "${SEED}" --log warning
  cp "result/${stem}/json/all_${stem}.json" "${CASE_DIR}/all_${stem}.json"
  echo "[gen] ${yaml} -> ${CASE_DIR}/all_${stem}.json"
}

# cases json -> 精度/显存报告
run_case() {
  "${ATK_CMD[@]}" task \
    -c "$1" \
    -n "${NODES_FILE}" \
    -p . \
    --task accuracy \
    --bm_device cpu \
    ${_SP} \
    --log warning \
    --save_data input \
    --save_data output
}

STEMS=()
for yaml in "${INTERFACE_NAME}"*.yaml; do
  [ -e "${yaml}" ] || continue
  stem="$(basename "${yaml}" .yaml)"
  gen_case "${yaml}" "${stem}"
  STEMS+=("${stem}")
done
if [ "${#STEMS[@]}" -eq 0 ]; then
  echo "[error] ${INTERFACE_DIR} 下没有 ${INTERFACE_NAME}*.yaml" >&2
  exit 1
fi
for stem in "${STEMS[@]}"; do
  run_case "${CASE_DIR}/all_${stem}.json"
done

# 用户场景 anchor，单独一个 json，报告与上面的生成用例分开
run_case "${CASE_DIR}/user_scenarios_fa3_varlen.json"
echo "[done] ${INTERFACE_NAME} (double_benchmark) -> ${CASE_DIR}"
