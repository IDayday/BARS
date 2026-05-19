#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

ENVS="${ENVS:-antmaze-medium-stitch-v0,antmaze-medium-navigate-v0}"
SEEDS="${SEEDS:-0}"
GPUS="${GPUS:-0}"
VARIANTS="${VARIANTS:-gas_shortest,gas_reachability_budget,gas_boundary_budget}"
BUDGETS="${BUDGETS:-2.0,3.0}"
FALLBACK_MODES="${FALLBACK_MODES:-progress_stall_v2,none}"
EPISODES="${EPISODES:-50}"
TASK_IDS="${TASK_IDS:-all}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/stage22}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
STAGE22_ROOT="${STAGE22_ROOT:-runs_stage22_eval}"
QUICK="${QUICK:-1}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
LOG_ROOT="${LOG_ROOT:-runs_stage22_eval_logs}"
MAX_PARALLEL_EVAL="${MAX_PARALLEL_EVAL:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export BARS_USE_TENSORBOARD="${BARS_USE_TENSORBOARD:-1}"
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-runs_stage22_tensorboard}"

IFS=',' read -r -a ENV_ARR <<< "$ENVS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a VARIANT_ARR <<< "$VARIANTS"
IFS=',' read -r -a BUDGET_ARR <<< "$BUDGETS"
IFS=',' read -r -a FALLBACK_ARR <<< "$FALLBACK_MODES"

mkdir -p "$LOG_ROOT"
FAILED="$LOG_ROOT/failed_jobs.csv"
if [[ ! -f "$FAILED" ]]; then
  echo "env,seed,variant,budget,fallback,stage,log,error_class" > "$FAILED"
fi

classify_error() {
  local log="$1"
  if grep -Eiq "ModuleNotFoundError|ImportError|No module named" "$log"; then echo "dependency/import";
  elif grep -Eiq "CUDA|cudnn|XLA|GPU|jaxlib|torch" "$log"; then echo "CUDA/JAX";
  elif grep -Eiq "MuJoCo|EGL|GLFW|mujoco" "$log"; then echo "MuJoCo/EGL";
  elif grep -Eiq "ogbench|d4rl|dataset|No registered env|reset.*goal" "$log"; then echo "env/dataset";
  elif grep -Eiq "checkpoint|params_|keygraph|artifact|pickle" "$log"; then echo "checkpoint/artifact";
  elif grep -Eiq "planner|budget_infeasible|no_start_connection" "$log"; then echo "planner";
  elif grep -Eiq "actor|sample_actions|skill" "$log"; then echo "policy adapter";
  else echo "unknown"; fi
}

run_eval_job() {
  local env="$1"
  local seed="$2"
  local variant="$3"
  local budget="$4"
  local fallback="$5"
  local gpu="$6"
  local run_dir="$LOG_ROOT/$env/seed$seed/$variant/budget$budget/fallback_$fallback"
  local out_dir="$STAGE22_ROOT/$env/seed$seed/$variant/budget$budget/fallback_$fallback"
  mkdir -p "$run_dir" "$out_dir"
  local log="$run_dir/evaluate.log"
  if [[ -f "$out_dir/eval.csv" ]]; then
    local count
    count="$(($(wc -l < "$out_dir/eval.csv") - 1))"
    if [[ "$count" -ge "$EPISODES" ]]; then
      echo "[stage22_eval] skip existing $env seed$seed $variant budget=$budget fallback=$fallback"
      echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"budget\":\"$budget\",\"fallback\":\"$fallback\",\"stage\":\"eval\",\"skipped\":true}" > "$run_dir/status.json"
      return 0
    fi
  fi
  echo "{\"status\":\"running\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"budget\":\"$budget\",\"fallback\":\"$fallback\",\"gpu\":\"$gpu\",\"stage\":\"eval\"}" > "$run_dir/status.json"
  python -m bars.gas_bars.evaluate \
    --env "$env" \
    --seed "$seed" \
    --task-ids "$TASK_IDS" \
    --episodes "$EPISODES" \
    --variant "$variant" \
    --budget "$budget" \
    --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
    --artifact-root "$ARTIFACT_ROOT" \
    --stage22-root "$STAGE22_ROOT" \
    --gas-repo-path "$GAS_REPO_PATH" \
    --gpu "$gpu" \
    --eval-on-cpu "$EVAL_ON_CPU" \
    --fallback-mode "$fallback" \
    --recompute-paths-per-episode 1 \
    --prefer-pretrained "$PREFER_PRETRAINED" \
    --train-if-missing "$TRAIN_IF_MISSING" \
    --quick "$QUICK" \
    > "$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"budget\":\"$budget\",\"fallback\":\"$fallback\",\"gpu\":\"$gpu\",\"stage\":\"eval\"}" > "$run_dir/status.json"
  else
    local err
    err="$(classify_error "$log")"
    echo "$env,$seed,$variant,$budget,$fallback,eval,$log,$err" >> "$FAILED"
    echo "{\"status\":\"failed\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"budget\":\"$budget\",\"fallback\":\"$fallback\",\"gpu\":\"$gpu\",\"stage\":\"eval\",\"error_class\":\"$err\"}" > "$run_dir/status.json"
    echo "[stage22_eval] FAILED env=$env seed=$seed variant=$variant budget=$budget fallback=$fallback class=$err; continuing"
  fi
  return "$rc"
}

active_jobs=0
overall_rc=0
idx=0
for env in "${ENV_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    for variant in "${VARIANT_ARR[@]}"; do
      for budget in "${BUDGET_ARR[@]}"; do
        for fallback in "${FALLBACK_ARR[@]}"; do
          gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
          idx=$((idx + 1))
          if [[ "$MAX_PARALLEL_EVAL" -le 1 ]]; then
            run_eval_job "$env" "$seed" "$variant" "$budget" "$fallback" "$gpu" || overall_rc=1
          else
            run_eval_job "$env" "$seed" "$variant" "$budget" "$fallback" "$gpu" &
            active_jobs=$((active_jobs + 1))
            if [[ "$active_jobs" -ge "$MAX_PARALLEL_EVAL" ]]; then
              if ! wait -n; then
                overall_rc=1
              fi
              active_jobs=$((active_jobs - 1))
            fi
          fi
        done
      done
    done
  done
done

while [[ "$active_jobs" -gt 0 ]]; do
  if ! wait -n; then
    overall_rc=1
  fi
  active_jobs=$((active_jobs - 1))
done

exit "$overall_rc"
