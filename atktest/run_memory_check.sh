#!/usr/bin/env bash
# ATK memory_check runner for flash-attn v2/v3/v4 on Ascend 910 / 950.
#
#   ./run_memory_check.sh --device 950 --card 4
#   ./run_memory_check.sh --device 950 --interface func_v3 --max-task 5
#   ./run_memory_check.sh --device 910 --interface func_v2,kvcache_v3
set -uo pipefail

ATKTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATK="${ATK_BIN:-/home/w00811046/miniconda3/envs/ascend/bin/atk}"
MATRIX="${ATKTEST_DIR}/atk_memory_check_matrix.yaml"
NODES_BASE="${ATKTEST_DIR}/nodes_memory_check.yaml"

DEVICE=""
CARD="0"
IFACE_ARG=""
MAX_TASK=""

usage() {
  cat <<'EOF'
用法: ./run_memory_check.sh --device {910|950} [选项]

  --device      必填，目标设备代际: 910 / 950
  --card        物理卡号，默认 0
  --interface   接口 key，逗号分隔；缺省为该设备支持的全部接口
  --max-task    只跑前 N 条用例（ATK -mt），用于冒烟
  --atk         atk 可执行文件路径（也可用环境变量 ATK_BIN）
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --device)     DEVICE="${2:-}"; shift 2 ;;
    --card)       CARD="${2:-}"; shift 2 ;;
    --interface)  IFACE_ARG="${2:-}"; shift 2 ;;
    --max-task)   MAX_TASK="${2:-}"; shift 2 ;;
    --atk)        ATK="${2:-}"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

case "${DEVICE}" in
  910|950) ;;
  *) echo "必须用 --device 指定 910 或 950" >&2; usage; exit 2 ;;
esac

CANN_SET_ENV="${CANN_SET_ENV:-${HOME}/miniconda3/Ascend/cann/set_env.sh}"
if [ -f "${CANN_SET_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV}"
fi

if ! command -v "${ATK}" >/dev/null 2>&1; then
  echo "找不到 atk: ${ATK}（请先 conda activate ascend 或用 --atk 指定）" >&2
  exit 2
fi
if [ ! -f "${MATRIX}" ]; then
  echo "缺少接口矩阵: ${MATRIX}" >&2
  exit 2
fi

# 1. 按目标设备生成用例
echo "=== 生成用例 (device=${DEVICE}) ==="
python3 "${ATKTEST_DIR}/cases/filter_cases.py" --device "${DEVICE}" || exit 1

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

# 3. 卡号：非 0 时生成一份覆盖 devices 的节点文件
NODES_FILE="${NODES_BASE}"
if [ "${CARD}" != "0" ]; then
  NODES_FILE="${ATKTEST_DIR}/nodes_memory_check.card${CARD}.yaml"
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

# 4. 逐接口串行执行（msSanitizer 不可并发）
declare -a SUMMARY=()
overall=0

for key in "${INTERFACES[@]}"; do
  case_file="${ATKTEST_DIR}/cases/${DEVICE}/${key}.json"
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

  RUN_DIR="${ATKTEST_DIR}/reports/${DEVICE}/${key}"
  mkdir -p "${RUN_DIR}/mss_logs" "${PLUGIN_DIR}/atk_output"
  marker="$(mktemp)"
  touch "${marker}"

  cmd=(
    "${ATK}" task
    -c "${case_file}"
    -n "${NODES_FILE}"
    -p .
    --task memory_check
    -ms
    -msl "${RUN_DIR}/mss_logs/${key}.log"
    --single_process
    --log warning
  )
  if [ -n "${MAX_TASK}" ]; then
    cmd+=(-mt "${MAX_TASK}")
  fi

  echo
  echo "=== [${DEVICE}] ${key} : ${#cmd[@]} args, card=${CARD} ==="
  ( cd "${PLUGIN_DIR}" && "${cmd[@]}" )
  rc=$?
  [ "${rc}" -ne 0 ] && overall="${rc}"

  while IFS= read -r run_dir; do
    mv "${run_dir}" "${RUN_DIR}/"
  done < <(find "${PLUGIN_DIR}/atk_output" -mindepth 1 -maxdepth 1 -type d -newer "${marker}" -print 2>/dev/null)
  rm -f "${marker}"

  fail_logs="$(find "${RUN_DIR}" -name 'mssanitizer.log' 2>/dev/null | wc -l)"
  SUMMARY+=("${key}	exit=${rc}	mss_fail_logs=${fail_logs}")
  echo "[${key}] exit=${rc} 归档 -> ${RUN_DIR}"
done

echo
echo "=== 汇总 (device=${DEVICE}) ==="
for line in "${SUMMARY[@]}"; do
  echo "  ${line}"
done
echo
echo "判读：reports/${DEVICE}/<接口>/ 下没有 mssanitizer.log 即该接口的用例全部通过内存检查。"
echo "原始 sanitizer 日志：reports/${DEVICE}/<接口>/mss_logs/<接口>.log"

exit "${overall}"
