#!/usr/bin/env bash
# BNSD 精度测试 runner：单标杆 (npu vs cpu) / 双标杆 (npu vs cpu vs gpu)。
#
#   ./run_accuracy.sh --device 910 --mode single
#   ./run_accuracy.sh --device 910 --interface func_v3 --mode single --limit 3
#   ./run_accuracy.sh --device 910 --mode double --card 0
#
# 单/双标杆的唯一区别是节点表：
#   single -> nodes_accuracy_single_bm.yaml (npu + cpu)，单进程可跑
#   double -> nodes_accuracy_double_bm.yaml (npu + cpu + gpu)，gpu 是远端主机，必须多进程
set -uo pipefail

ACCURACY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATK="${ATK_BIN:-$(command -v atk || true)}"
MATRIX="${ACCURACY_DIR}/atk_accuracy_matrix.yaml"

DEVICE=""
MODE="single"
CARD="0"
IFACE_ARG=""
LIMIT=""
ATK_LOG="${ATK_LOG:-warning}"

usage() {
  cat <<'EOF'
用法: ./run_accuracy.sh --device {910|950} [选项]

  --device     必填，目标设备代际: 910 / 950
  --mode       single（单标杆 npu vs cpu，默认）/ double（双标杆 npu vs cpu vs gpu）
  --card       物理卡号，默认 0
  --interface  接口 key，逗号分隔；缺省为该设备支持的全部接口
  --limit      每个接口只跑前 N 条用例，用于冒烟
  --atk        atk 可执行文件路径（也可用环境变量 ATK_BIN）
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --device)    DEVICE="${2:-}"; shift 2 ;;
    --mode)      MODE="${2:-}"; shift 2 ;;
    --card)      CARD="${2:-}"; shift 2 ;;
    --interface) IFACE_ARG="${2:-}"; shift 2 ;;
    --limit)     LIMIT="${2:-}"; shift 2 ;;
    --atk)       ATK="${2:-}"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

case "${DEVICE}" in
  910|950) ;;
  *) echo "必须用 --device 指定 910 或 950" >&2; usage; exit 2 ;;
esac

case "${MODE}" in
  single)
    NODES_BASE="${ACCURACY_DIR}/nodes_accuracy_single_bm.yaml"
    SINGLE_PROCESS=("--single_process")
    ;;
  double)
    NODES_BASE="${ACCURACY_DIR}/nodes_accuracy_double_bm.yaml"
    # 远端 GPU 标杆节点无法在单进程模式下调度。
    SINGLE_PROCESS=()
    ;;
  *) echo "--mode 只能是 single 或 double" >&2; usage; exit 2 ;;
esac

CANN_SET_ENV="${CANN_SET_ENV:-${HOME}/miniconda3/Ascend/cann/set_env.sh}"
if [ -f "${CANN_SET_ENV}" ]; then
  # set_env.sh 会引用未定义的 PYTHONPATH 等变量，在 set -u 下会直接退出，故临时关闭
  set +u
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV}"
  set -u
fi

if [ -z "${ATK}" ] || ! command -v "${ATK}" >/dev/null 2>&1; then
  echo "找不到 atk（请先 conda activate ascend，或用 --atk / ATK_BIN 指定）" >&2
  exit 2
fi
if [ ! -f "${MATRIX}" ]; then
  echo "缺少接口矩阵: ${MATRIX}" >&2
  exit 2
fi
if [ ! -f "${NODES_BASE}" ]; then
  echo "缺少节点表: ${NODES_BASE}" >&2
  exit 2
fi

# 1. 按目标设备派生用例（不改写精度标准，比较器由各 plugin 目录自带）
echo "=== 生成用例 (device=${DEVICE}) ==="
FILTER_ARGS=(--device "${DEVICE}")
[ -n "${LIMIT}" ] && FILTER_ARGS+=(--limit "${LIMIT}")
python3 "${ACCURACY_DIR}/cases/filter_cases.py" "${FILTER_ARGS[@]}" || exit 1

# 2. 待跑接口列表
if [ -n "${IFACE_ARG}" ]; then
  IFS=',' read -r -a INTERFACES <<< "${IFACE_ARG}"
else
  mapfile -t INTERFACES < <(python3 - "${MATRIX}" "${DEVICE}" <<'PY'
import sys
import yaml

matrix, device = sys.argv[1], sys.argv[2]
with open(matrix, encoding="utf-8") as handle:
    entries = yaml.safe_load(handle)["interfaces"]
for entry in entries:
    if str(device) in {str(item) for item in entry.get("devices", ())}:
        print(entry["key"])
PY
  )
fi

if [ "${#INTERFACES[@]}" -eq 0 ]; then
  echo "设备 ${DEVICE} 没有可用接口" >&2
  exit 2
fi

# 3. 卡号：非 0 时生成一份覆盖 npu 节点 devices 的节点文件（只替换第一处）
NODES_FILE="${NODES_BASE}"
if [ "${CARD}" != "0" ]; then
  NODES_FILE="${NODES_BASE%.yaml}.card${CARD}.yaml"
  python3 - "${NODES_BASE}" "${NODES_FILE}" "${CARD}" <<'PY'
import sys

src, dst, card = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, encoding="utf-8") as handle:
    text = handle.read()
with open(dst, "w", encoding="utf-8") as handle:
    handle.write(text.replace("devices: [0]", f"devices: [{card}]", 1))
PY
fi

resolve_paths() {
  python3 - "${MATRIX}" "$1" <<'PY'
import pathlib
import sys
import yaml

matrix, key = sys.argv[1], sys.argv[2]
with open(matrix, encoding="utf-8") as handle:
    entries = {entry["key"]: entry for entry in yaml.safe_load(handle)["interfaces"]}
base = pathlib.Path(matrix).resolve().parent
entry = entries[key]
print((base / entry["plugin_dir"]).resolve())
print((base / entry["case_source"]).resolve())
PY
}

# 4. 逐接口串行执行
declare -a SUMMARY=()
overall=0

for key in "${INTERFACES[@]}"; do
  case_file="${ACCURACY_DIR}/cases/${DEVICE}/${key}.json"
  if [ ! -f "${case_file}" ]; then
    echo "[${key}] 跳过：该接口不支持 ${DEVICE}（无 ${case_file}）"
    SUMMARY+=("${key}	SKIP	-")
    continue
  fi

  mapfile -t RESOLVED < <(resolve_paths "${key}")
  PLUGIN_DIR="${RESOLVED[0]:-}"
  if [ ! -d "${PLUGIN_DIR}" ]; then
    echo "[${key}] 跳过：plugin 目录不存在 ${PLUGIN_DIR}" >&2
    SUMMARY+=("${key}	SKIP	-")
    continue
  fi

  RUN_DIR="${ACCURACY_DIR}/reports/${DEVICE}/${key}/${MODE}"
  mkdir -p "${RUN_DIR}" "${PLUGIN_DIR}/atk_output"
  marker="$(mktemp)"
  touch "${marker}"

  cmd=(
    "${ATK}" task
    -c "${case_file}"
    -n "${NODES_FILE}"
    -p .
    --task accuracy
    --bm_device cpu
    --log "${ATK_LOG}"
    --save_data input
    --save_data output
  )
  cmd+=("${SINGLE_PROCESS[@]}")

  echo
  echo "=== [${DEVICE}] ${key} (${MODE}) : ${#cmd[@]} args, card=${CARD} ==="
  ( cd "${PLUGIN_DIR}" && "${cmd[@]}" )
  rc=$?
  [ "${rc}" -ne 0 ] && overall="${rc}"

  run_archived=""
  while IFS= read -r run_dir; do
    mv "${run_dir}" "${RUN_DIR}/"
    run_archived="${RUN_DIR}/$(basename "${run_dir}")"
  done < <(find "${PLUGIN_DIR}/atk_output" -mindepth 1 -maxdepth 1 -type d -newer "${marker}" -print 2>/dev/null)
  rm -f "${marker}"

  fail_lines=0
  if [ -n "${run_archived}" ] && [ -f "${run_archived}/log/atk.log" ]; then
    fail_lines="$(grep -cE '\[ERROR\]' "${run_archived}/log/atk.log" || true)"
  fi
  SUMMARY+=("${key}	exit=${rc}	error_lines=${fail_lines}")
  echo "[${key}] exit=${rc} 归档 -> ${RUN_DIR}"
done

echo
echo "=== 汇总 (device=${DEVICE}, mode=${MODE}) ==="
for line in "${SUMMARY[@]}"; do
  echo "  ${line}"
done
echo
echo "判读：reports/${DEVICE}/<接口>/${MODE}/<run>/report/*.xlsx 为精度报告；"
echo "      reports/${DEVICE}/<接口>/${MODE}/<run>/log/atk.log 里 error_info 形如"
echo "      'max_re=... (..x <= 2.0); avg_re=... (..x <= 1.2); rms_re=... (..x <= 1.2)'。"

# 按 --card 生成的节点文件只在本次运行有效。
if [ "${NODES_FILE}" != "${NODES_BASE}" ]; then
  rm -f "${NODES_FILE}"
fi

exit "${overall}"
