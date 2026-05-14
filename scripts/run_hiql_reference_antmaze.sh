#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${HIQL_REPO:-external_src/HIQL}"
ENV_NAME="${ENV_NAME:-antmaze-medium-play-v2}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
ALGO="${ALGO:-hiql}"
RUN_GROUP="${RUN_GROUP:-HIQL_REF}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ ! -d "$REPO" ]; then
  echo "Missing HIQL repo at $REPO. Run scripts/setup_routeb_backbones.sh first." >&2
  exit 2
fi
cd "$REPO"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="$ROOT_DIR/external_runtime${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" main.py \
  --run_group "$RUN_GROUP" \
  --wandb.offline=true \
  --seed "$SEED" \
  --env_name "$ENV_NAME" \
  --pretrain_steps "${PRETRAIN_STEPS:-1000002}" \
  --eval_interval "${EVAL_INTERVAL:-100000}" \
  --save_interval "${SAVE_INTERVAL:-250000}" \
  --p_currgoal "${P_CURRGOAL:-0.2}" \
  --p_trajgoal "${P_TRAJGOAL:-0.5}" \
  --p_randomgoal "${P_RANDOMGOAL:-0.3}" \
  --discount "${DISCOUNT:-0.99}" \
  --temperature "${TEMPERATURE:-1}" \
  --high_temperature "${HIGH_TEMPERATURE:-1}" \
  --pretrain_expectile "${PRETRAIN_EXPECTILE:-0.7}" \
  --geom_sample "${GEOM_SAMPLE:-1}" \
  --use_layer_norm "${USE_LAYER_NORM:-1}" \
  --value_hidden_dim "${VALUE_HIDDEN_DIM:-512}" \
  --value_num_layers "${VALUE_NUM_LAYERS:-3}" \
  --batch_size "${BATCH_SIZE:-1024}" \
  --use_rep "${USE_REP:-1}" \
  --policy_train_rep "${POLICY_TRAIN_REP:-0}" \
  --rep_dim "${REP_DIM:-10}" \
  --rep_type "${REP_TYPE:-concat}" \
  --algo_name "$ALGO" \
  --use_waypoints "${USE_WAYPOINTS:-1}" \
  --way_steps "${WAY_STEPS:-25}" \
  --high_p_randomgoal "${HIGH_P_RANDOMGOAL:-0.3}"
