#!/usr/bin/env bash
set -euo pipefail

ENVS="${ENVS:-antmaze-giant-stitch-v0,antmaze-large-explore-v0,scene-play-v0}"
SEEDS="${SEEDS:-0}"
TASK_IDS="${TASK_IDS:-1,2,3,4,5}"
EVAL_EPISODES_PER_TASK="${EVAL_EPISODES_PER_TASK:-50}"
USE_OFFICIAL_ARTIFACTS="${USE_OFFICIAL_ARTIFACTS:-1}"
FULL_BUDGET_ONLY="${FULL_BUDGET_ONLY:-1}"
ROUND="${ROUND:-003}"
GPUS="${GPUS:-0}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
RUN_ROOT="${RUN_ROOT:-runs_round003_official_gas_eval}"
REPORTS_ROOT="${REPORTS_ROOT:-reports}"
FORCE="${FORCE:-0}"
OFFICIAL_PROTOCOL="${OFFICIAL_PROTOCOL:-1}"

for arg in "$@"; do
  case "$arg" in
    ENVS=*) ENVS="${arg#ENVS=}" ;;
    SEEDS=*) SEEDS="${arg#SEEDS=}" ;;
    TASK_IDS=*) TASK_IDS="${arg#TASK_IDS=}" ;;
    EVAL_EPISODES_PER_TASK=*) EVAL_EPISODES_PER_TASK="${arg#EVAL_EPISODES_PER_TASK=}" ;;
    USE_OFFICIAL_ARTIFACTS=*) USE_OFFICIAL_ARTIFACTS="${arg#USE_OFFICIAL_ARTIFACTS=}" ;;
    FULL_BUDGET_ONLY=*) FULL_BUDGET_ONLY="${arg#FULL_BUDGET_ONLY=}" ;;
    ROUND=*) ROUND="${arg#ROUND=}" ;;
    GPUS=*) GPUS="${arg#GPUS=}" ;;
    GAS_ARTIFACT_ROOT=*) GAS_ARTIFACT_ROOT="${arg#GAS_ARTIFACT_ROOT=}" ;;
    GAS_REPO_PATH=*) GAS_REPO_PATH="${arg#GAS_REPO_PATH=}" ;;
    RUN_ROOT=*) RUN_ROOT="${arg#RUN_ROOT=}" ;;
    REPORTS_ROOT=*) REPORTS_ROOT="${arg#REPORTS_ROOT=}" ;;
    FORCE=*) FORCE="${arg#FORCE=}" ;;
    OFFICIAL_PROTOCOL=*) OFFICIAL_PROTOCOL="${arg#OFFICIAL_PROTOCOL=}" ;;
  esac
done

if [[ -f scripts/stage24_env_mirrors.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/stage24_env_mirrors.sh
fi

export PYTHONPATH="${PYTHONPATH:-$PWD}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export D4RL_SUPPRESS_IMPORT_ERROR=1
export BARS_OGBENCH_OFFLINE="${BARS_OGBENCH_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

python scripts/round003_official_gas_eval_driver.py \
  --envs "$ENVS" \
  --seeds "$SEEDS" \
  --task-ids "$TASK_IDS" \
  --eval-episodes-per-task "$EVAL_EPISODES_PER_TASK" \
  --use-official-artifacts "$USE_OFFICIAL_ARTIFACTS" \
  --full-budget-only "$FULL_BUDGET_ONLY" \
  --round "$ROUND" \
  --gpus "$GPUS" \
  --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
  --gas-repo-path "$GAS_REPO_PATH" \
  --run-root "$RUN_ROOT" \
  --reports-root "$REPORTS_ROOT" \
  --force "$FORCE" \
  --official-protocol "$OFFICIAL_PROTOCOL"
