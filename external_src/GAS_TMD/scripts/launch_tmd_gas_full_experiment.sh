#!/usr/bin/env bash
set -euo pipefail

ENVS=(${ENVS:-antmaze-medium-stitch-v0 antmaze-medium-navigate-v0 antmaze-large-stitch-v0 antmaze-large-navigate-v0})
SEEDS=(${SEEDS:-0 1 2})
GPUS=(${GPUS:-0 1 2 3})
MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS[@]}}"

TRAIN_STEPS_TMD="${TRAIN_STEPS_TMD:-1000000}"
TRAIN_STEPS_TMD_LOW="${TRAIN_STEPS_TMD_LOW:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
MAX_CALIBRATION_PAIRS="${MAX_CALIBRATION_PAIRS:-50000}"
ROOT="${ROOT:-artifacts/tmd_gas_full_$(date +%Y%m%d_%H%M%S)}"
GAS_POLICY_ROOT="${GAS_POLICY_ROOT:-}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /root/miniconda3/envs/gcrlo/bin/python ]; then
    PYTHON_BIN=/root/miniconda3/envs/gcrlo/bin/python
  else
    PYTHON_BIN=python
  fi
fi

cd "$(dirname "$0")/../../.."
mkdir -p "$ROOT/logs"
PIDS_TSV="$ROOT/monitor_pids.tsv"
printf "stage\tenv\tseed\tgpu\tpid\tlog\n" > "$PIDS_TSV"

job_idx=0

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 30
  done
}

launch_job() {
  local stage="$1"; shift
  local env_name="$1"; shift
  local seed="$1"; shift
  local gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
  local log="$ROOT/logs/${stage}_${env_name}_seed${seed}.log"
  wait_for_slot
  (
    set -x
    "$@" --env_name "$env_name" --seed "$seed" --gpu "$gpu"
  ) > "$log" 2>&1 &
  local pid=$!
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$stage" "$env_name" "$seed" "$gpu" "$pid" "$log" >> "$PIDS_TSV"
  job_idx=$((job_idx + 1))
}

wait_stage() {
  local stage="$1"
  local failed=0
  for pid in $(jobs -rp); do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "Stage failed: $stage" >&2
    exit 1
  fi
}

run_dir_for() {
  local root="$1"; local group="$2"; local env_name="$3"; local seed="$4"
  find "$root/$group" -maxdepth 1 -type d -name "${env_name}_sd$(printf '%03d' "$seed")_*" | sort | tail -n 1
}

latest_ckpt() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name "params_*.pkl" | sort -V | tail -n 1
}

find_gas_policy_ckpt() {
  local env_name="$1"; local seed="$2"
  if [ -z "$GAS_POLICY_ROOT" ]; then
    return 0
  fi
  find "$GAS_POLICY_ROOT" -type f -name "params_*.pkl" -path "*${env_name}_sd$(printf '%03d' "$seed")_*" | sort -V | tail -n 1
}

echo "[Stage 1] Train official TMD agents."
for env_name in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    launch_job pretrain_tmd "$env_name" "$seed" \
      "$PYTHON_BIN" external_src/GAS_TMD/pretrain_tmd.py \
      --train_steps "$TRAIN_STEPS_TMD" \
      --log_interval "$LOG_INTERVAL" \
      --save_interval "$SAVE_INTERVAL" \
      --run_group tmd_actor_train \
      --save_tmd_dir "$ROOT/exp_tmd"
  done
done
wait_stage pretrain_tmd

echo "[Stage 2] Construct TMD directed graphs."
for env_name in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tmd_dir="$(run_dir_for "$ROOT/exp_tmd" tmd_actor_train "$env_name" "$seed")"
    tmd_ckpt="$(latest_ckpt "$tmd_dir")"
    launch_job construct_graph_tmd "$env_name" "$seed" \
      "$PYTHON_BIN" external_src/GAS_TMD/construct_graph_tmd.py \
      --tmd_path "$tmd_ckpt" \
      --run_group tmd_graph_construct \
      --save_graph_dir "$ROOT/exp_graph_tmd" \
      --max_calibration_pairs "$MAX_CALIBRATION_PAIRS"
  done
done
wait_stage construct_graph_tmd

echo "[Stage 3] Evaluate TMD graph + TMD actor."
for env_name in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tmd_ckpt="$(latest_ckpt "$(run_dir_for "$ROOT/exp_tmd" tmd_actor_train "$env_name" "$seed")")"
    graph_dir="$(run_dir_for "$ROOT/exp_graph_tmd" tmd_graph_construct "$env_name" "$seed")"
    launch_job eval_tmd_actor "$env_name" "$seed" \
      "$PYTHON_BIN" external_src/GAS_TMD/evaluate_gas_tmd.py \
      --mode tmd_graph_tmd_actor \
      --tmd_path "$tmd_ckpt" \
      --keygraph_tmd_path "$graph_dir/keygraph_tmd.pkl" \
      --eval_episodes "$EVAL_EPISODES" \
      --run_group tmd_actor_eval \
      --save_eval_dir "$ROOT/exp_eval_tmd"
  done
done
wait_stage eval_tmd_actor

echo "[Stage 4] Train TMD-conditioned GAS low-level policies."
for env_name in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tmd_ckpt="$(latest_ckpt "$(run_dir_for "$ROOT/exp_tmd" tmd_actor_train "$env_name" "$seed")")"
    graph_dir="$(run_dir_for "$ROOT/exp_graph_tmd" tmd_graph_construct "$env_name" "$seed")"
    launch_job train_tmd_low "$env_name" "$seed" \
      "$PYTHON_BIN" external_src/GAS_TMD/train_policy_tmd_low.py \
      --tmd_path "$tmd_ckpt" \
      --tmd_calibration_path "$graph_dir/tmd_calibration.json" \
      --train_steps "$TRAIN_STEPS_TMD_LOW" \
      --log_interval "$LOG_INTERVAL" \
      --save_interval "$SAVE_INTERVAL" \
      --run_group tmd_full_gas_low_train \
      --save_policy_dir "$ROOT/exp_policy_tmd_low"
  done
done
wait_stage train_tmd_low

echo "[Stage 5] Evaluate TMD graph + TMD-conditioned GAS low-level policy."
for env_name in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tmd_ckpt="$(latest_ckpt "$(run_dir_for "$ROOT/exp_tmd" tmd_actor_train "$env_name" "$seed")")"
    graph_dir="$(run_dir_for "$ROOT/exp_graph_tmd" tmd_graph_construct "$env_name" "$seed")"
    low_ckpt="$(latest_ckpt "$(run_dir_for "$ROOT/exp_policy_tmd_low" tmd_full_gas_low_train "$env_name" "$seed")")"
    launch_job eval_tmd_low "$env_name" "$seed" \
      "$PYTHON_BIN" external_src/GAS_TMD/evaluate_gas_tmd.py \
      --mode tmd_full_gas_low \
      --tmd_path "$tmd_ckpt" \
      --tmd_low_policy_path "$low_ckpt" \
      --keygraph_tmd_path "$graph_dir/keygraph_tmd.pkl" \
      --eval_episodes "$EVAL_EPISODES" \
      --run_group tmd_full_gas_low_eval \
      --save_eval_dir "$ROOT/exp_eval_tmd"
  done
done
wait_stage eval_tmd_low

if [ -n "$GAS_POLICY_ROOT" ]; then
  echo "[Stage 6] Evaluate TMD graph + existing GAS policy."
  for env_name in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      gas_ckpt="$(find_gas_policy_ckpt "$env_name" "$seed" || true)"
      if [ -z "$gas_ckpt" ]; then
        echo "Skipping GAS policy eval for $env_name seed $seed; no checkpoint under $GAS_POLICY_ROOT"
        continue
      fi
      tmd_ckpt="$(latest_ckpt "$(run_dir_for "$ROOT/exp_tmd" tmd_actor_train "$env_name" "$seed")")"
      graph_dir="$(run_dir_for "$ROOT/exp_graph_tmd" tmd_graph_construct "$env_name" "$seed")"
      launch_job eval_gas_policy "$env_name" "$seed" \
        "$PYTHON_BIN" external_src/GAS_TMD/evaluate_gas_tmd.py \
        --mode tmd_graph_gas_policy \
        --tmd_path "$tmd_ckpt" \
        --gas_policy_path "$gas_ckpt" \
        --keygraph_tmd_path "$graph_dir/keygraph_tmd.pkl" \
        --eval_episodes "$EVAL_EPISODES" \
        --run_group tmd_graph_gas_policy_eval \
        --save_eval_dir "$ROOT/exp_eval_tmd"
    done
  done
  wait_stage eval_gas_policy
fi

echo "Full TMD-GAS experiment complete: $ROOT"
echo "PID/log manifest: $PIDS_TSV"
