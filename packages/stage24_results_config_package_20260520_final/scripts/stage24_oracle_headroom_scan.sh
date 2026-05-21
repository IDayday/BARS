#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

CONFIG="${CONFIG:-configs/stage24_oracle_scan.json}"
ENVS="${ENVS:-scene-play-v0}"
SEEDS="${SEEDS:-0}"
GPUS="${GPUS:-0}"
GPU_FIRST="${GPUS%%,*}"
WAIT="${WAIT:-1}"
EDGE_EXEC_PILOT="${EDGE_EXEC_PILOT:-1}"
TOP_K_BRIDGE="${TOP_K_BRIDGE:-4}"
MAX_SOURCES="${MAX_SOURCES:-200}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/stage24}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
REPORTS_ROOT="${REPORTS_ROOT:-reports}"
ORACLE_REPORTS_ROOT="${ORACLE_REPORTS_ROOT:-reports/stage24_oracle_scan_tmp}"
PREPARE_LOG_ROOT="${PREPARE_LOG_ROOT:-runs_stage24_prepare_oracle}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
QUICK="${QUICK:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export BARS_USE_TENSORBOARD="${BARS_USE_TENSORBOARD:-1}"
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-runs_stage24_tensorboard}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p "$REPORTS_ROOT" "$ORACLE_REPORTS_ROOT" "$ARTIFACT_ROOT"
overall=0

echo "[stage24_oracle] prepare/load official GAS artifacts"
if [[ "$SKIP_PREPARE" != "1" ]]; then
  ARTIFACT_ROOT="$GAS_ARTIFACT_ROOT" ENVS="$ENVS" SEEDS="$SEEDS" GPUS="$GPUS" QUICK="$QUICK" PREFER_PRETRAINED="$PREFER_PRETRAINED" TRAIN_IF_MISSING="$TRAIN_IF_MISSING" LOG_ROOT="$PREPARE_LOG_ROOT" bash scripts/stage22_prepare_gas_backbone.sh || overall=1
else
  echo "[stage24_oracle] skip prepare (SKIP_PREPARE=1)"
fi

echo "[stage24_oracle] build bridge graphs G0/G1/G2/G3"
python scripts/stage23_build_bridge_graphs.py \
  --envs "$ENVS" --seeds "$SEEDS" \
  --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --reports-root "$ORACLE_REPORTS_ROOT" \
  --top-k-bridge "$TOP_K_BRIDGE" \
  --max-sources "$MAX_SOURCES" || overall=1

echo "[stage24_oracle] run edge execution pilots"
python scripts/stage23_edge_execution.py \
  --envs "$ENVS" --seeds "$SEEDS" --graph-id G3 \
  --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --gas-repo-path "$GAS_REPO_PATH" \
  --reports-root "$ORACLE_REPORTS_ROOT" \
  --gpu "$GPU_FIRST" \
  --pilot "$EDGE_EXEC_PILOT" \
  --prefer-pretrained "$PREFER_PRETRAINED" \
  --train-if-missing "$TRAIN_IF_MISSING" || overall=1

echo "[stage24_oracle] build oracle graph from rollout-success bridge labels"
python scripts/stage23_oracle_bridge.py \
  --envs "$ENVS" --seeds "$SEEDS" --graph-id G3 \
  --artifact-root "$ARTIFACT_ROOT" \
  --reports-root "$ORACLE_REPORTS_ROOT" || overall=1

python scripts/stage24_analyze.py \
  --reports-root "$REPORTS_ROOT" \
  --oracle-reports-root "$ORACLE_REPORTS_ROOT" \
  --oracle-artifact-root "$ARTIFACT_ROOT" || true

if [[ "$WAIT" == "1" ]]; then
  echo "[stage24_oracle] reports written to $REPORTS_ROOT/stage24_oracle_headroom.csv"
fi
exit "$overall"
