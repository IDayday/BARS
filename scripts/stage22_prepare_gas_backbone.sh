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
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/gas}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
QUICK="${QUICK:-1}"
FULL="${FULL:-0}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
LOG_ROOT="${LOG_ROOT:-runs_stage22_prepare}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export BARS_USE_TENSORBOARD="${BARS_USE_TENSORBOARD:-1}"
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-runs_stage22_tensorboard}"

IFS=',' read -r -a ENV_ARR <<< "$ENVS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

mkdir -p "$LOG_ROOT"
FAILED="$LOG_ROOT/failed_jobs.csv"
if [[ ! -f "$FAILED" ]]; then
  echo "env,seed,stage,log,error_class" > "$FAILED"
fi

classify_error() {
  local log="$1"
  if grep -Eiq "ModuleNotFoundError|ImportError|No module named" "$log"; then echo "dependency/import";
  elif grep -Eiq "CUDA|cudnn|XLA|GPU|jaxlib" "$log"; then echo "CUDA/JAX";
  elif grep -Eiq "MuJoCo|EGL|GLFW|mujoco" "$log"; then echo "MuJoCo/EGL";
  elif grep -Eiq "ogbench|d4rl|dataset|No registered env|Environment" "$log"; then echo "env/dataset";
  elif grep -Eiq "checkpoint|params_|keygraph|artifact|pickle" "$log"; then echo "checkpoint/artifact";
  else echo "unknown"; fi
}

idx=0
for env in "${ENV_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
    idx=$((idx + 1))
    run_dir="$LOG_ROOT/$env/seed$seed"
    mkdir -p "$run_dir"
    log="$run_dir/prepare.log"
    echo "{\"status\":\"running\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"prepare\"}" > "$run_dir/status.json"
    echo "[stage22_prepare] env=$env seed=$seed gpu=$gpu log=$log"
    python -m bars.external.gas_prepare \
      --env "$env" \
      --seed "$seed" \
      --artifact-root "$ARTIFACT_ROOT" \
      --gas-repo-path "$GAS_REPO_PATH" \
      --gpu "$gpu" \
      --quick "$QUICK" \
      --prefer-pretrained "$PREFER_PRETRAINED" \
      --train-if-missing "$TRAIN_IF_MISSING" \
      --export-embeddings 1 \
      > "$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"prepare\"}" > "$run_dir/status.json"
    else
      err="$(classify_error "$log")"
      echo "$env,$seed,prepare,$log,$err" >> "$FAILED"
      echo "{\"status\":\"failed\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"prepare\",\"error_class\":\"$err\"}" > "$run_dir/status.json"
      echo "[stage22_prepare] FAILED env=$env seed=$seed class=$err; continuing"
    fi
  done
done
