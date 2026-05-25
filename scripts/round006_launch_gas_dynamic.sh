#!/usr/bin/env bash
set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

if [[ -f scripts/stage24_env_mirrors.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/stage24_env_mirrors.sh
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export D4RL_SUPPRESS_IMPORT_ERROR=1
export BARS_OGBENCH_OFFLINE=0
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-/root/remote/datasets/ogbench}"
export BARS_DOWNLOAD_PARALLEL_CHUNKS="${BARS_DOWNLOAD_PARALLEL_CHUNKS:-1}"
export ROUND006_GPU_SLOTS_PER_GPU="${ROUND006_GPU_SLOTS_PER_GPU:-2}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /root/anaconda3/envs/gcrlo/bin/python ]]; then
    export PYTHON=/root/anaconda3/envs/gcrlo/bin/python
  else
    export PYTHON=python
  fi
fi

ROUND="${ROUND:-006}"
RUN_ROOT="${RUN_ROOT:-runs_round006_gas_dynamic}"
OUT_ROOT="${OUT_ROOT:-artifacts/gas_selftrain_round006}"
SEEDS="${SEEDS:-42,43,44,45,46}"
GPUS="${GPUS:-0,1,2,3,4,5}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DOWNLOAD_POLL_SECONDS="${DOWNLOAD_POLL_SECONDS:-30}"
ENVS="${ENVS:-}"

mkdir -p "$RUN_ROOT/_orchestrator" reports commands "rounds/round_${ROUND}"

cmd=(
  "$PYTHON" scripts/round006_gas_dynamic_orchestrator.py
  --round "$ROUND"
  --dataset-dir "$OGBENCH_DATASET_DIR"
  --run-root "$RUN_ROOT"
  --out-root "$OUT_ROOT"
  --seeds "$SEEDS"
  --gpus "$GPUS"
  --gpu-slots-per-gpu "$ROUND006_GPU_SLOTS_PER_GPU"
  --poll-seconds "$POLL_SECONDS"
  --download-poll-seconds "$DOWNLOAD_POLL_SECONDS"
)

if [[ -n "$ENVS" ]]; then
  cmd+=(--envs "$ENVS")
fi

printf '%q ' "${cmd[@]}" > "commands/round_${ROUND}_gas_dynamic_active_command.txt"
printf '\n' >> "commands/round_${ROUND}_gas_dynamic_active_command.txt"

setsid "${cmd[@]}" < /dev/null >> "$RUN_ROOT/_orchestrator/orchestrator.log" 2>&1 &
pid=$!
echo "$pid" > "$RUN_ROOT/_orchestrator/orchestrator.pid"
cat > "reports/round_${ROUND}_gas_dynamic_process.md" <<EOF
# Round ${ROUND} GAS Dynamic Process

- Started: $(date -Is)
- PID: ${pid}
- Log: ${RUN_ROOT}/_orchestrator/orchestrator.log
- Events: ${RUN_ROOT}/_orchestrator/events.jsonl
- Dataset root: ${OGBENCH_DATASET_DIR}
- Run root: ${RUN_ROOT}
- Artifact root: ${OUT_ROOT}
- Seeds: ${SEEDS}
- GPUs: ${GPUS}
- GPU slots per GPU: ${ROUND006_GPU_SLOTS_PER_GPU}
- Python: ${PYTHON}
- Proxy env HTTP_PROXY set: $([[ -n "${HTTP_PROXY:-}" ]] && echo yes || echo no)
- Proxy env HTTPS_PROXY set: $([[ -n "${HTTPS_PROXY:-}" ]] && echo yes || echo no)
EOF

echo "$pid"
