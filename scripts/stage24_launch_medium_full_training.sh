#!/usr/bin/env bash
set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

ENVS="${ENVS:-antmaze-medium-navigate-v0,antmaze-medium-stitch-v0}"
SEEDS="${SEEDS:-0,1,2}"
GPUS="${GPUS:-0,1,2,3,4,5}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/gas}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
LOG_ROOT="${LOG_ROOT:-runs_stage24_train_medium_full_bg}"
QUICK="${QUICK:-0}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-0}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

IFS=',' read -r -a ENV_ARR <<< "$ENVS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

mkdir -p "$LOG_ROOT"
MANIFEST="$LOG_ROOT/job_manifest.tsv"
PIDS="$LOG_ROOT/pids.tsv"
echo -e "env\tseed\tgpu\tpid\tlog\tstatus" > "$PIDS"
echo -e "env\tseed\tgpu\tlog_root\tartifact_root" > "$MANIFEST"

idx=0
for env in "${ENV_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
    idx=$((idx + 1))
    log_dir="$LOG_ROOT/$env/seed$seed"
    mkdir -p "$log_dir"
    echo -e "$env\t$seed\t$gpu\t$log_dir\t$ARTIFACT_ROOT" >> "$MANIFEST"
    job="$log_dir/job.sh"
    cat > "$job" <<EOF
#!/usr/bin/env bash
set +e
cd "$(pwd)"
echo "{\"status\":\"running\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"gas_prepare_full\",\"started_at\":\"\$(date -Is)\"}" > "$log_dir/status.json"
echo "[stage24_train] env=$env seed=$seed gpu=$gpu started \$(date -Is)"
PATH=/root/anaconda3/envs/gcrlo/bin:\$PATH \\
PYTHONPATH="$(pwd)" \\
WANDB_MODE="$WANDB_MODE" \\
WANDB_DISABLED="$WANDB_DISABLED" \\
XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \\
python -m bars.external.gas_prepare \\
  --env "$env" \\
  --seed "$seed" \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --gas-repo-path "$GAS_REPO_PATH" \\
  --gpu "$gpu" \\
  --quick "$QUICK" \\
  --prefer-pretrained "$PREFER_PRETRAINED" \\
  --train-if-missing "$TRAIN_IF_MISSING" \\
  --export-embeddings 1
rc=\$?
if [[ \$rc -eq 0 ]]; then
  echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"gas_prepare_full\",\"finished_at\":\"\$(date -Is)\",\"rc\":\$rc}" > "$log_dir/status.json"
else
  echo "{\"status\":\"failed\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"gas_prepare_full\",\"finished_at\":\"\$(date -Is)\",\"rc\":\$rc}" > "$log_dir/status.json"
fi
echo "[stage24_train] env=$env seed=$seed gpu=$gpu finished rc=\$rc \$(date -Is)"
exit \$rc
EOF
    chmod +x "$job"
    echo "{\"status\":\"launching\",\"env\":\"$env\",\"seed\":$seed,\"gpu\":\"$gpu\",\"stage\":\"gas_prepare_full\",\"launched_at\":\"$(date -Is)\"}" > "$log_dir/status.json"
    nohup bash "$job" > "$log_dir/prepare.log" 2>&1 < /dev/null &
    pid=$!
    disown "$pid" 2>/dev/null || true
    echo -e "$env\t$seed\t$gpu\t$pid\t$log_dir/prepare.log\tlaunched" >> "$PIDS"
    echo "[stage24_launch] env=$env seed=$seed gpu=$gpu pid=$pid log=$log_dir/prepare.log"
  done
done

echo "[stage24_launch] manifest=$MANIFEST"
echo "[stage24_launch] pids=$PIDS"
