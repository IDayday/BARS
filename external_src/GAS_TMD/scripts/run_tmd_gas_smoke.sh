#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-antmaze-medium-stitch-v0}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
ROOT="${ROOT:-artifacts/tmd_gas_smoke_$(date +%Y%m%d_%H%M%S)}"
MAX_DATASET_STATES="${MAX_DATASET_STATES:-4096}"
SMOKE_LATENT_DIM="${SMOKE_LATENT_DIM:-128}"
SMOKE_HIDDEN_DIMS="${SMOKE_HIDDEN_DIMS:-(128,128)}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-64}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /root/miniconda3/envs/gcrlo/bin/python ]; then
    PYTHON_BIN=/root/miniconda3/envs/gcrlo/bin/python
  else
    PYTHON_BIN=python
  fi
fi

cd "$(dirname "$0")/../../.."
mkdir -p "$ROOT/logs"

"$PYTHON_BIN" external_src/GAS_TMD/pretrain_tmd.py \
  --env_name "$ENV_NAME" \
  --seed "$SEED" \
  --gpu "$GPU" \
  --train_steps 1000 \
  --log_interval 100 \
  --save_interval 1000 \
  --run_group smoke_tmd_actor \
  --save_tmd_dir "$ROOT/exp_tmd" \
  --agent_config.latent_dim="$SMOKE_LATENT_DIM" \
  --agent_config.batch_size="$SMOKE_BATCH_SIZE" \
  --agent_config.actor_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  --agent_config.value_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  > "$ROOT/logs/pretrain_tmd.log" 2>&1

TMD_RUN_DIR="$(find "$ROOT/exp_tmd/smoke_tmd_actor" -maxdepth 1 -type d -name "${ENV_NAME}_sd$(printf '%03d' "$SEED")_*" | sort | tail -n 1)"
TMD_CKPT="$TMD_RUN_DIR/params_1000.pkl"

"$PYTHON_BIN" external_src/GAS_TMD/construct_graph_tmd.py \
  --env_name "$ENV_NAME" \
  --seed "$SEED" \
  --gpu "$GPU" \
  --tmd_path "$TMD_CKPT" \
  --save_graph_dir "$ROOT/exp_graph_tmd" \
  --run_group smoke_tmd_graph \
  --max_calibration_pairs 2048 \
  --max_dataset_states "$MAX_DATASET_STATES" \
  --topk 32 \
  --agent_config.latent_dim="$SMOKE_LATENT_DIM" \
  --agent_config.batch_size="$SMOKE_BATCH_SIZE" \
  --agent_config.actor_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  --agent_config.value_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  > "$ROOT/logs/construct_graph_tmd.log" 2>&1

GRAPH_RUN_DIR="$(find "$ROOT/exp_graph_tmd/smoke_tmd_graph" -maxdepth 1 -type d -name "${ENV_NAME}_sd$(printf '%03d' "$SEED")_*" | sort | tail -n 1)"
KEYGRAPH="$GRAPH_RUN_DIR/keygraph_tmd.pkl"

"$PYTHON_BIN" external_src/GAS_TMD/evaluate_gas_tmd.py \
  --mode tmd_graph_tmd_actor \
  --env_name "$ENV_NAME" \
  --seed "$SEED" \
  --gpu "$GPU" \
  --tmd_path "$TMD_CKPT" \
  --keygraph_tmd_path "$KEYGRAPH" \
  --eval_episodes 2 \
  --eval_tasks 1 \
  --eval_max_steps 100 \
  --run_group smoke_tmd_actor_eval \
  --save_eval_dir "$ROOT/exp_eval_tmd" \
  --tmd_agent_config.latent_dim="$SMOKE_LATENT_DIM" \
  --tmd_agent_config.batch_size="$SMOKE_BATCH_SIZE" \
  --tmd_agent_config.actor_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  --tmd_agent_config.value_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  > "$ROOT/logs/evaluate_tmd_actor.log" 2>&1

"$PYTHON_BIN" external_src/GAS_TMD/train_policy_tmd_low.py \
  --env_name "$ENV_NAME" \
  --seed "$SEED" \
  --gpu "$GPU" \
  --tmd_path "$TMD_CKPT" \
  --tmd_calibration_path "$GRAPH_RUN_DIR/tmd_calibration.json" \
  --train_steps 1000 \
  --log_interval 100 \
  --save_interval 1000 \
  --run_group smoke_tmd_full_gas_low \
  --save_policy_dir "$ROOT/exp_policy_tmd_low" \
  --max_dataset_states "$MAX_DATASET_STATES" \
  --tmd_agent_config.latent_dim="$SMOKE_LATENT_DIM" \
  --tmd_agent_config.batch_size="$SMOKE_BATCH_SIZE" \
  --tmd_agent_config.actor_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  --tmd_agent_config.value_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  --agent_config.tmd_latent_dim="$SMOKE_LATENT_DIM" \
  --agent_config.skill_dim="$((SMOKE_LATENT_DIM + 1))" \
  --agent_config.batch_size="$SMOKE_BATCH_SIZE" \
  --agent_config.actor_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  --agent_config.value_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  > "$ROOT/logs/train_policy_tmd_low.log" 2>&1

LOW_RUN_DIR="$(find "$ROOT/exp_policy_tmd_low/smoke_tmd_full_gas_low" -maxdepth 1 -type d -name "${ENV_NAME}_sd$(printf '%03d' "$SEED")_*" | sort | tail -n 1)"
LOW_CKPT="$LOW_RUN_DIR/params_1000.pkl"

"$PYTHON_BIN" external_src/GAS_TMD/evaluate_gas_tmd.py \
  --mode tmd_full_gas_low \
  --env_name "$ENV_NAME" \
  --seed "$SEED" \
  --gpu "$GPU" \
  --tmd_path "$TMD_CKPT" \
  --tmd_low_policy_path "$LOW_CKPT" \
  --keygraph_tmd_path "$KEYGRAPH" \
  --eval_episodes 2 \
  --eval_tasks 1 \
  --eval_max_steps 100 \
  --run_group smoke_tmd_full_gas_low_eval \
  --save_eval_dir "$ROOT/exp_eval_tmd" \
  --tmd_agent_config.latent_dim="$SMOKE_LATENT_DIM" \
  --tmd_agent_config.batch_size="$SMOKE_BATCH_SIZE" \
  --tmd_agent_config.actor_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  --tmd_agent_config.value_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  --tmd_low_agent_config.tmd_latent_dim="$SMOKE_LATENT_DIM" \
  --tmd_low_agent_config.skill_dim="$((SMOKE_LATENT_DIM + 1))" \
  --tmd_low_agent_config.actor_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  --tmd_low_agent_config.value_hidden_dims="$SMOKE_HIDDEN_DIMS" \
  > "$ROOT/logs/evaluate_tmd_full_gas_low.log" 2>&1

echo "Smoke complete: $ROOT"
