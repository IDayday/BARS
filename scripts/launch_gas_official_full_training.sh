#!/usr/bin/env bash
set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gcrlo/bin/python}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/gas_official_full_20260521}"
LOG_ROOT="${LOG_ROOT:-runs_gas_official_full_20260521}"
ENVS="${ENVS:-antmaze-medium-navigate-v0,antmaze-large-navigate-v0,antmaze-giant-navigate-v0,antmaze-medium-stitch-v0,antmaze-large-stitch-v0,antmaze-giant-stitch-v0,antmaze-medium-explore-v0,antmaze-large-explore-v0,scene-play-v0,kitchen-partial-v0}"
SEEDS="${SEEDS:-42,43,44,45,46}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-49}"
EVAL_VIDEO_EPISODES="${EVAL_VIDEO_EPISODES:-1}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"
SCHEDULER_SLEEP="${SCHEDULER_SLEEP:-60}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export BARS_USE_TENSORBOARD="${BARS_USE_TENSORBOARD:-0}"
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$LOG_ROOT/tensorboard}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"

PY_SITE_PACKAGES="$("$PYTHON_BIN" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
if [[ -d "$PY_SITE_PACKAGES/nvidia" ]]; then
  JAX_CUDA_LIBS="$(find "$PY_SITE_PACKAGES/nvidia" -type d -path '*/lib' | paste -sd: -)"
else
  JAX_CUDA_LIBS=""
fi
export PYTHONPATH="$(pwd):$(pwd)/$GAS_REPO_PATH:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${JAX_CUDA_LIBS:+$JAX_CUDA_LIBS:}/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export PATH="$PY_SITE_PACKAGES/nvidia/cuda_nvcc/bin:$PATH"

IFS=',' read -r -a ENV_ARR <<< "$ENVS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

mkdir -p "$LOG_ROOT"
MANIFEST="$LOG_ROOT/job_manifest.tsv"
PIDS="$LOG_ROOT/pids.tsv"
QUEUE="$LOG_ROOT/queue.tsv"
echo -e "env\tseed\tgpu\tartifact_root\tlog_dir" > "$MANIFEST"
echo -e "env\tseed\tgpu\tpid\tlog\tstatus" > "$PIDS"
: > "$QUEUE"

for env_name in "${ENV_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    echo -e "$env_name\t$seed" >> "$QUEUE"
  done
done

declare -A GPU_PIDS=()

gpu_free() {
  local gpu="$1"
  local pid="${GPU_PIDS[$gpu]:-}"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if jobs -pr | grep -qx "$pid"; then
    return 1
  fi
  wait "$pid" || true
  unset "GPU_PIDS[$gpu]"
  return 0
}

launch_one() {
  local env_name="$1"
  local seed="$2"
  local gpu="$3"
  local log_dir="$LOG_ROOT/$env_name/seed$seed"
  mkdir -p "$log_dir"
  local log="$log_dir/run.log"
  local job="$log_dir/job.sh"
  echo -e "$env_name\t$seed\t$gpu\t$ARTIFACT_ROOT\t$log_dir" >> "$MANIFEST"
  cat > "$job" <<EOF
#!/usr/bin/env bash
set +e
cd "$(pwd)"
export WANDB_MODE="$WANDB_MODE"
export WANDB_DISABLED="$WANDB_DISABLED"
export BARS_USE_TENSORBOARD="$BARS_USE_TENSORBOARD"
export TENSORBOARD_LOGDIR="$TENSORBOARD_LOGDIR/$env_name/seed$seed"
export XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE"
export MUJOCO_GL="$MUJOCO_GL"
export D4RL_SUPPRESS_IMPORT_ERROR="$D4RL_SUPPRESS_IMPORT_ERROR"
export PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export PATH="$PATH"
echo '{"status":"running","stage":"train","env":"$env_name","seed":$seed,"gpu":"$gpu","started_at":"'\$(date -Is)'"}' > "$log_dir/status.json"
echo "[gas_official] train env=$env_name seed=$seed gpu=$gpu started \$(date -Is)"
"$PYTHON_BIN" -m bars.external.gas_prepare \\
  --env "$env_name" \\
  --seed "$seed" \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --gas-repo-path "$GAS_REPO_PATH" \\
  --gpu "$gpu" \\
  --quick 0 \\
  --prefer-pretrained 0 \\
  --train-if-missing 1 \\
  --export-embeddings 1
train_rc=\$?
if [[ "\$train_rc" -eq 0 && "$EVAL_AFTER_TRAIN" == "1" ]]; then
  echo '{"status":"running","stage":"eval","env":"$env_name","seed":$seed,"gpu":"$gpu","started_eval_at":"'\$(date -Is)'"}' > "$log_dir/status.json"
  echo "[gas_official] eval env=$env_name seed=$seed gpu=$gpu started \$(date -Is)"
  "$PYTHON_BIN" scripts/gas_official_eval.py \\
    --env "$env_name" \\
    --seed "$seed" \\
    --gpu "$gpu" \\
    --artifact-root "$ARTIFACT_ROOT" \\
    --gas-repo-path "$GAS_REPO_PATH" \\
    --eval-episodes "$EVAL_EPISODES" \\
    --eval-video-episodes "$EVAL_VIDEO_EPISODES" \\
    --eval-on-cpu "$EVAL_ON_CPU"
  eval_rc=\$?
else
  eval_rc=0
fi
if [[ "\$train_rc" -eq 0 && "\$eval_rc" -eq 0 ]]; then
  status=completed
  rc=0
else
  status=failed
  rc=\$(( train_rc != 0 ? train_rc : eval_rc ))
fi
echo '{"status":"'\$status'","env":"$env_name","seed":$seed,"gpu":"$gpu","train_rc":"'\$train_rc'","eval_rc":"'\$eval_rc'","finished_at":"'\$(date -Is)'"}' > "$log_dir/status.json"
echo "[gas_official] env=$env_name seed=$seed gpu=$gpu finished train_rc=\$train_rc eval_rc=\$eval_rc \$(date -Is)"
exit "\$rc"
EOF
  chmod +x "$job"
  nohup bash "$job" > "$log" 2>&1 < /dev/null &
  local pid=$!
  GPU_PIDS[$gpu]="$pid"
  echo -e "$env_name\t$seed\t$gpu\t$pid\t$log\tlaunched" >> "$PIDS"
  echo "[gas_official] launched env=$env_name seed=$seed gpu=$gpu pid=$pid log=$log"
}

while IFS=$'\t' read -r env_name seed; do
  launched=0
  while [[ "$launched" -eq 0 ]]; do
    for gpu in "${GPU_ARR[@]}"; do
      if gpu_free "$gpu"; then
        launch_one "$env_name" "$seed" "$gpu"
        launched=1
        break
      fi
    done
    if [[ "$launched" -eq 0 ]]; then
      sleep "$SCHEDULER_SLEEP"
    fi
  done
done < "$QUEUE"

for gpu in "${!GPU_PIDS[@]}"; do
  pid="${GPU_PIDS[$gpu]}"
  if kill -0 "$pid" 2>/dev/null; then
    wait "$pid" || true
  fi
done

echo "[gas_official] all queued jobs finished $(date -Is)"
