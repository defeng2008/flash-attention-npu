#!/usr/bin/env bash
# 用 cases/gen 下的生成配置重建 cases/source/<key>.json。
#
#   ./gen_cases.sh                        # 重建全部 5 个接口
#   ./gen_cases.sh --interface func_v3    # 只重建一个
#   ./gen_cases.sh --dry-run              # 只生成到 cases/gen/result 并打印条数，不覆盖 cases/source
set -uo pipefail

ATKTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DIR="${ATKTEST_DIR}/cases/gen"
SOURCE_DIR="${ATKTEST_DIR}/cases/source"
MATRIX="${ATKTEST_DIR}/atk_memory_check_matrix.yaml"
ATK="${ATK_BIN:-$(command -v atk || true)}"

SEED=2026
IFACE_ARG=""
DRY_RUN=0

usage() {
  cat <<'EOF'
用法: ./gen_cases.sh [选项]

  --interface  接口 key，逗号分隔；缺省为矩阵中的全部接口
  --seed       生成随机种子，默认 2026
  --dry-run    只生成到 cases/gen/result 并打印条数，不覆盖 cases/source
  --atk        atk 可执行文件路径（也可用环境变量 ATK_BIN）
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --interface) IFACE_ARG="${2:-}"; shift 2 ;;
    --seed)      SEED="${2:-}"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --atk)       ATK="${2:-}"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

CANN_SET_ENV="${CANN_SET_ENV:-${HOME}/miniconda3/Ascend/cann/set_env.sh}"
if [ -f "${CANN_SET_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV}"
fi

if [ -z "${ATK}" ] || ! command -v "${ATK}" >/dev/null 2>&1; then
  echo "找不到 atk（请先 conda activate ascend，或用 --atk / ATK_BIN 指定）" >&2
  exit 2
fi

if [ -n "${IFACE_ARG}" ]; then
  IFS=',' read -r -a INTERFACES <<< "${IFACE_ARG}"
else
  # 接口列表取自矩阵，避免与 atk_memory_check_matrix.yaml 双份维护。
  mapfile -t INTERFACES < <(python3 - "${MATRIX}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    for entry in yaml.safe_load(handle)["interfaces"]:
        print(entry["key"])
PY
  )
fi

[ "${DRY_RUN}" -eq 1 ] && echo "=== dry-run：结果只写到 cases/gen/result，不覆盖 cases/source ==="

overall=0
for key in "${INTERFACES[@]}"; do
  yaml_file="${GEN_DIR}/${key}.yaml"
  if [ ! -f "${yaml_file}" ]; then
    echo "[${key}] 跳过：缺少生成配置 ${yaml_file}" >&2
    overall=1
    continue
  fi

  # result/<key>/json/all_<key>.json 由 atk 覆盖写入，先清掉避免读到上一轮产物。
  rm -rf "${GEN_DIR:?}/result/${key}"
  ( cd "${GEN_DIR}" && "${ATK}" case -f "${key}.yaml" -p . -s "${SEED}" -en 0 )
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "[${key}] 生成失败 exit=${rc}" >&2
    overall=1
    continue
  fi

  generated="${GEN_DIR}/result/${key}/json/all_${key}.json"
  if [ ! -f "${generated}" ]; then
    echo "[${key}] 生成失败：找不到 ${generated}" >&2
    overall=1
    continue
  fi

  count="$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${generated}")"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[${key}] 生成 ${count} 条 -> ${generated}"
  else
    mkdir -p "${SOURCE_DIR}"
    cp "${generated}" "${SOURCE_DIR}/${key}.json"
    echo "[${key}] 生成 ${count} 条 -> cases/source/${key}.json"
  fi
done

# 生成过程的中间产物不入库。
rm -rf "${GEN_DIR}/atk_output" "${GEN_DIR}/__pycache__" "${GEN_DIR}/atk.log"
if [ "${DRY_RUN}" -eq 1 ]; then
  echo "dry-run 产物保留在 cases/gen/result/ 供核对。"
else
  rm -rf "${GEN_DIR:?}/result"
fi

echo
echo "完成。下一步：python3 cases/filter_cases.py --device {910|950}"
echo "注意：cases/source 变更后需重新 filter，910/950 才会用上新用例。"
exit "${overall}"
