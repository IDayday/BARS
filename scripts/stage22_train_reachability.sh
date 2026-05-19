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
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/stage22}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
QUICK="${QUICK:-1}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
LOG_ROOT="${LOG_ROOT:-runs_stage22_reachability}"
FORCE_REACHABILITY="${FORCE_REACHABILITY:-0}"
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
  elif grep -Eiq "CUDA|cudnn|XLA|GPU|jaxlib|torch" "$log"; then echo "CUDA/JAX";
  elif grep -Eiq "ogbench|d4rl|dataset|No registered env" "$log"; then echo "env/dataset";
  elif grep -Eiq "checkpoint|params_|keygraph|artifact|pickle" "$log"; then echo "checkpoint/artifact";
  else echo "unknown"; fi
}

idx=0
for env in "${ENV_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
    idx=$((idx + 1))
    run_dir="$LOG_ROOT/$env/seed$seed"
    out="$ARTIFACT_ROOT/$env/seed$seed"
    mkdir -p "$run_dir" "$out"
    log="$run_dir/reachability.log"
    if [[ "$FORCE_REACHABILITY" != "1" && -f "$out/model.pt" && -f "$out/edge_scores.csv" ]]; then
      echo "[stage22_reach] skip existing $env seed$seed"
      echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"stage\":\"reachability\",\"skipped\":true}" > "$run_dir/status.json"
      continue
    fi
    echo "{\"status\":\"running\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"reachability\"}" > "$run_dir/status.json"
    python -m bars.gas_bars.score_edges \
      --env "$env" \
      --seed "$seed" \
      --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
      --gas-repo-path "$GAS_REPO_PATH" \
      --out "$out" \
      --quick "$QUICK" \
      --gpu "$gpu" \
      --device cuda \
      --prefer-pretrained "$PREFER_PRETRAINED" \
      --train-if-missing "$TRAIN_IF_MISSING" \
      > "$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"reachability\"}" > "$run_dir/status.json"
    else
      err="$(classify_error "$log")"
      echo "$env,$seed,reachability,$log,$err" >> "$FAILED"
      echo "{\"status\":\"failed\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"reachability\",\"error_class\":\"$err\"}" > "$run_dir/status.json"
      echo "[stage22_reach] FAILED env=$env seed=$seed class=$err; continuing"
    fi
  done
done
