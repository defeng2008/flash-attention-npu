#!/usr/bin/env bash
# 用 msprof --quick 采集 kernel 时间，对比同一份 shape 的
# BSND（model.py）与连续 BNSD（model_new_ascendc.py）实现。
#
# 用法:
#   ./run_msprof.sh --device 950 --card 6              # 950 的三个接口（串行）
#   ./run_msprof.sh --device 950 --parallel            # 三接口各占一张卡并行
#   ./run_msprof.sh --device 950 --parallel --cards "5 6 7"
#   ./run_msprof.sh --device 910 --card 1              # 需在 910 机器上执行
#   ./run_msprof.sh --device 950 --card 6 --keys func_v3,kvcache_v3
#   ./run_msprof.sh --device 950 --card 6 --passes 3   # 每接口跑多轮取中位数
#   ./run_msprof.sh --summary                          # 只汇总已有 performance.json
#
# 判定口径：mean_speedup = 平均(BSND耗时 / BNSD耗时)，< 1 表示 BNSD 更慢。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 优先用仓库内自带的 profiler 副本（tools/），保证 checkout 后即可运行；
# 需要跟随 skill 更新时用 MSPROF_PROFILER 覆盖。
PROFILER="${MSPROF_PROFILER:-$HERE/tools/msprof_profile_run.sh}"
if [[ ! -f "$PROFILER" ]]; then
  echo "ERROR: 未找到 $PROFILER，请用 MSPROF_PROFILER=<path>/msprof_profile_run.sh 指定" >&2
  exit 2
fi

DEVICE=950
CARD=6
KEYS=""
WARMUP=3
REPEATS=5
PASSES=3
PARALLEL=0
CARDS="5 6 7"
SUMMARY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --card) CARD="$2"; shift 2 ;;
    --keys) KEYS="${2//,/ }"; shift 2 ;;
    --warm-up) WARMUP="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --passes) PASSES="$2"; shift 2 ;;
    --parallel) PARALLEL=1; shift ;;
    --cards) CARDS="${2//,/ }"; shift 2 ;;
    --summary) SUMMARY=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

case "$DEVICE" in
  910) DEVICE_KEYS="func_v2 kvcache_v2 func_v3 kvcache_v3 func_v4" ;;
  950) DEVICE_KEYS="func_v3 kvcache_v3 func_v4" ;;
  *) echo "--device 只支持 910 / 950" >&2; exit 2 ;;
esac

summarize() {
  MSFROF_DIR="$HERE" KEYS="${KEYS:-$DEVICE_KEYS}" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MSFROF_DIR"])
keys = os.environ["KEYS"].split()
print(f"{'接口':<12}{'有效/总':>8}{'BSND均值us':>12}{'BNSD均值us':>12}{'平均耗时劣化':>14}{'平均加速比':>12}")
rows = []
for key in keys:
    path = root / key / "performance.json"
    if not path.is_file():
        print(f"{key:<12}{'缺失':>8}")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    ref, asc = data["mean_ref_us"], data["mean_asc_us"]
    degrade = (asc - ref) / ref * 100 if ref else float("nan")
    rows.append((key, data["n_cases_valid"], data["n_cases_total"], ref, asc, degrade,
                 data["mean_speedup"]))
    print(f"{key:<12}{data['n_cases_valid']:>4}/{data['n_cases_total']:<3}{ref:>12.2f}{asc:>12.2f}"
          f"{degrade:>13.2f}%{data['mean_speedup']:>12.4f}")
if rows:
    total_ref = sum(r[3] for r in rows)
    total_asc = sum(r[4] for r in rows)
    print(f"{'合计':<12}{'':>8}{total_ref:>12.2f}{total_asc:>12.2f}"
          f"{(total_asc - total_ref) / total_ref * 100:>13.2f}%{total_ref / total_asc:>12.4f}")
PY
}

if [[ "$SUMMARY" == "1" ]]; then
  summarize
  exit 0
fi

[[ -f "$PROFILER" ]] || { echo "未找到 $PROFILER" >&2; exit 1; }

python3 "$HERE/gen_cases.py"

# 单个接口：跑满 PASSES 轮再取中位数；输出全部落到 <key>/run.log，便于并行时不串台。
run_key() {
  local key="$1" card="$2"
  local log="$HERE/$key/run.log"
  rm -f "$HERE/$key"/performance.pass*.json "$HERE/$key"/perf_report.pass*.md
  : > "$log"
  local pass
  for pass in $(seq 1 "$PASSES"); do
    echo "=== [$DEVICE] $key (card $card) pass $pass/$PASSES ===" >> "$log"
    bash "$PROFILER" --quick --output-dir="$HERE/$key" \
      --warm-up="$WARMUP" --repeats="$REPEATS" --device="$card" >> "$log" 2>&1
    mv "$HERE/$key/performance.json" "$HERE/$key/performance.pass$pass.json"
    mv "$HERE/$key/perf_report.md" "$HERE/$key/perf_report.pass$pass.md"
  done
  python3 "$HERE/aggregate.py" "$key" >> "$log" 2>&1
  echo "--- [$DEVICE] $key done (card $card)"
}

RUN_KEYS="${KEYS:-$DEVICE_KEYS}"
KEY_TOTAL=$(wc -w <<< "$RUN_KEYS")

if [[ "$PARALLEL" == "1" ]]; then
  read -r -a CARD_ARR <<< "$CARDS"
  [[ ${#CARD_ARR[@]} -gt 0 ]] || { echo "--cards 为空" >&2; exit 2; }
  idx=0
  pids=()
  for key in $RUN_KEYS; do
    card="${CARD_ARR[$((idx % ${#CARD_ARR[@]}))]}"
    echo "=== [$DEVICE] $key -> card $card (并行 $((idx + 1))/$KEY_TOTAL) ==="
    run_key "$key" "$card" &
    pids+=("$!")
    idx=$((idx + 1))
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  [[ "$fail" == "0" ]] || echo "警告：有接口执行失败，详见各自 run.log" >&2
else
  for key in $RUN_KEYS; do
    echo "=== [$DEVICE] $key (card $CARD) ==="
    run_key "$key" "$CARD"
  done
fi

echo
echo "=== $DEVICE 汇总（平均口径）==="
summarize
