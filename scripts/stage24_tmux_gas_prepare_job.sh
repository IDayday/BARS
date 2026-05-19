#!/usr/bin/env bash
set -uo pipefail

ENV_NAME="$1"
SEED="$2"
GPU="$3"
ARTIFACT_ROOT="${4:-artifacts/gas_stage24_full}"
LOG_ROOT="${5:-runs_stage24_tmux_medium}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"

RUN_ID="${ENV_NAME//[^A-Za-z0-9_]/_}_seed${SEED}_gpu${GPU}"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/$RUN_ID.log"
STATUS="$LOG_ROOT/$RUN_ID.status.json"

echo "{\"status\":\"running\",\"env\":\"$ENV_NAME\",\"seed\":$SEED,\"gpu\":\"$GPU\",\"started_at\":\"$(date -Is)\"}" > "$STATUS"
echo "[stage24_tmux] env=$ENV_NAME seed=$SEED gpu=$GPU started $(date -Is)" > "$LOG"

PATH=/root/anaconda3/envs/gcrlo/bin:$PATH \
PYTHONPATH=/root/remote/BARS \
WANDB_MODE=disabled \
WANDB_DISABLED=true \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m bars.external.gas_prepare \
  --env "$ENV_NAME" \
  --seed "$SEED" \
  --artifact-root "$ARTIFACT_ROOT" \
  --gas-repo-path "$GAS_REPO_PATH" \
  --gpu "$GPU" \
  --quick 0 \
  --prefer-pretrained 0 \
  --train-if-missing 1 \
  --export-embeddings 1 \
  >> "$LOG" 2>&1
RC=$?

if [[ "$RC" -eq 0 ]]; then
  ST="completed"
else
  ST="failed"
fi
echo "{\"status\":\"$ST\",\"env\":\"$ENV_NAME\",\"seed\":$SEED,\"gpu\":\"$GPU\",\"finished_at\":\"$(date -Is)\",\"rc\":$RC}" > "$STATUS"
echo "[stage24_tmux] env=$ENV_NAME seed=$SEED gpu=$GPU finished rc=$RC $(date -Is)" >> "$LOG"
exit "$RC"
