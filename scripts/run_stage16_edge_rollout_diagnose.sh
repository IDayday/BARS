#!/usr/bin/env bash
set -Eeuo pipefail

SRC_ROOT="${SRC_ROOT:-runs_stage16_full12}"
CONFIG="${CONFIG:-configs/d4rl_antmaze_stage16.json}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6}"
MAX_PARALLEL="${MAX_PARALLEL:-7}"
NUM_EDGES="${NUM_EDGES:-256}"
HORIZON="${HORIZON:-50}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-1.0}"
PER_GROUP="${PER_GROUP:-48}"
PACKAGE="${PACKAGE:-1}"
REPORT_DIR="${REPORT_DIR:-reports}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${REPORT_DIR}/stage16_edge_rollout_diagnose_${TS}.log"
mkdir -p "$REPORT_DIR"

IFS=',' read -r -a GPU_ARR <<< "$GPUS_CSV"
if [ "${#GPU_ARR[@]}" -eq 0 ]; then
  GPU_ARR=(0)
fi

if [ ! -d "$SRC_ROOT" ]; then
  echo "[ERROR] SRC_ROOT does not exist: $SRC_ROOT" | tee -a "$LOG_PATH" >&2
  exit 1
fi

mapfile -t RUN_DIRS < <(find "$SRC_ROOT" -mindepth 3 -maxdepth 3 -type d -path "*/full_bars/*" | sort)
if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
  echo "[ERROR] no full_bars run dirs found under $SRC_ROOT" | tee -a "$LOG_PATH" >&2
  exit 1
fi

echo "[INFO] SRC_ROOT=$SRC_ROOT" | tee -a "$LOG_PATH"
echo "[INFO] CONFIG=$CONFIG" | tee -a "$LOG_PATH"
echo "[INFO] GPUS=$GPUS_CSV MAX_PARALLEL=$MAX_PARALLEL NUM_EDGES=$NUM_EDGES" | tee -a "$LOG_PATH"
echo "[INFO] run_count=${#RUN_DIRS[@]}" | tee -a "$LOG_PATH"

active=0
idx=0
fail=0
for run_dir in "${RUN_DIRS[@]}"; do
  env_name="$(basename "$(dirname "$(dirname "$run_dir")")")"
  variant="$(basename "$(dirname "$run_dir")")"
  run_base="$(basename "$run_dir")"
  seed="0"
  if [[ "$run_base" =~ seed([0-9]+) ]]; then
    seed="${BASH_REMATCH[1]}"
  fi
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  idx=$((idx + 1))
  echo "[LAUNCH] gpu=$gpu env=$env_name seed=$seed run_dir=$run_dir" | tee -a "$LOG_PATH"
  (
    set -Eeuo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    export D4RL_SUPPRESS_IMPORT_ERROR=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
    cmd=(python -m bars.cli diagnose \
      --config "$CONFIG" \
      --run-dir "$run_dir" \
      --env "$env_name" \
      --seed "$seed" \
      --variant "$variant" \
      --node-method bars \
      --set diagnostics.edge_rollout_enabled=true \
      --set diagnostics.edge_rollout.enabled=true \
      --set diagnostics.edge_rollout.num_edges="$NUM_EDGES" \
      --set diagnostics.edge_rollout.horizon="$HORIZON" \
      --set diagnostics.edge_rollout.goal_tolerance="$SUCCESS_THRESHOLD" \
      --set diagnostics.edge_rollout.success_threshold="$SUCCESS_THRESHOLD" \
      --set diagnostics.edge_rollout.stratified=true \
      --set diagnostics.edge_rollout.per_group="$PER_GROUP" \
      --set 'diagnostics.edge_rollout.groups=["selected_supported","selected_unlabeled_bridge","selected_hard_neg_proxy","unselected_supported","unselected_hard_neg_proxy"]')
    if [ "$PACKAGE" = "1" ]; then
      cmd+=(--package)
    fi
    "${cmd[@]}"
    echo "[DONE] env=$env_name seed=$seed run_dir=$run_dir"
  ) >> "$LOG_PATH" 2>&1 || {
    echo "[FAILED] env=$env_name seed=$seed run_dir=$run_dir" | tee -a "$LOG_PATH" >&2
    exit 1
  } &
  active=$((active + 1))
  if [ "$active" -ge "$MAX_PARALLEL" ]; then
    if ! wait -n; then
      fail=$((fail + 1))
    fi
    active=$((active - 1))
  fi
done
while [ "$active" -gt 0 ]; do
  if ! wait -n; then
    fail=$((fail + 1))
  fi
  active=$((active - 1))
done

echo "[INFO] diagnostics finished failures=$fail log=$LOG_PATH" | tee -a "$LOG_PATH"
exit "$fail"
