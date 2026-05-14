#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${GAS_REPO:-external_src/GAS}"
STAGE="${STAGE:-construct_graph}"
ENV_NAME="${ENV_NAME:-antmaze-giant-stitch-v0}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
if [ ! -d "$REPO" ]; then
  echo "Missing GAS repo at $REPO. Run scripts/setup_routeb_backbones.sh first." >&2
  exit 2
fi
cd "$REPO"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export CUDA_VISIBLE_DEVICES="$GPU"
case "$STAGE" in
  construct_graph)
    : "${TDR_PATH:?Set TDR_PATH to official GAS params/TDR checkpoint path}"
    python construct_graph.py --env_name "$ENV_NAME" --seed "$SEED" --gpu "$GPU" --tdr_path "$TDR_PATH" --save_graph_dir "${SAVE_GRAPH_DIR:-exp_graph}" --te_threshold "${TE_THRESHOLD:-0.99}"
    ;;
  *)
    echo "Unsupported GAS stage=$STAGE. Use official GAS scripts directly for pretrain/evaluate." >&2
    exit 3
    ;;
esac
