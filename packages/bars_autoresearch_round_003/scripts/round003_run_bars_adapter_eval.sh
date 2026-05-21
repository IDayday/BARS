#!/usr/bin/env bash
set -euo pipefail

ENVS="${ENVS:-}"
SEEDS="${SEEDS:-0}"
TASK_IDS="${TASK_IDS:-1,2,3,4,5}"
EVAL_EPISODES_PER_TASK="${EVAL_EPISODES_PER_TASK:-50}"
VARIANT="${VARIANT:-gas_shortest_official_control}"
FALLBACK_MODE="${FALLBACK_MODE:-none}"
USE_OFFICIAL_ARTIFACTS="${USE_OFFICIAL_ARTIFACTS:-1}"
ROUND="${ROUND:-003}"
GPUS="${GPUS:-0}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/round003_adapter}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
RUN_ROOT="${RUN_ROOT:-runs_round003_bars_adapter_eval}"
REPORTS_ROOT="${REPORTS_ROOT:-reports}"
BASELINE_CERTIFICATION="${BASELINE_CERTIFICATION:-reports/round_003_baseline_certification.json}"
FORCE="${FORCE:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"

for arg in "$@"; do
  case "$arg" in
    ENVS=*) ENVS="${arg#ENVS=}" ;;
    SEEDS=*) SEEDS="${arg#SEEDS=}" ;;
    TASK_IDS=*) TASK_IDS="${arg#TASK_IDS=}" ;;
    EVAL_EPISODES_PER_TASK=*) EVAL_EPISODES_PER_TASK="${arg#EVAL_EPISODES_PER_TASK=}" ;;
    VARIANT=*) VARIANT="${arg#VARIANT=}" ;;
    FALLBACK_MODE=*) FALLBACK_MODE="${arg#FALLBACK_MODE=}" ;;
    USE_OFFICIAL_ARTIFACTS=*) USE_OFFICIAL_ARTIFACTS="${arg#USE_OFFICIAL_ARTIFACTS=}" ;;
    ROUND=*) ROUND="${arg#ROUND=}" ;;
    GPUS=*) GPUS="${arg#GPUS=}" ;;
    GAS_ARTIFACT_ROOT=*) GAS_ARTIFACT_ROOT="${arg#GAS_ARTIFACT_ROOT=}" ;;
    ARTIFACT_ROOT=*) ARTIFACT_ROOT="${arg#ARTIFACT_ROOT=}" ;;
    GAS_REPO_PATH=*) GAS_REPO_PATH="${arg#GAS_REPO_PATH=}" ;;
    RUN_ROOT=*) RUN_ROOT="${arg#RUN_ROOT=}" ;;
    REPORTS_ROOT=*) REPORTS_ROOT="${arg#REPORTS_ROOT=}" ;;
    BASELINE_CERTIFICATION=*) BASELINE_CERTIFICATION="${arg#BASELINE_CERTIFICATION=}" ;;
    FORCE=*) FORCE="${arg#FORCE=}" ;;
    EVAL_ON_CPU=*) EVAL_ON_CPU="${arg#EVAL_ON_CPU=}" ;;
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

python scripts/round003_adapter_eval_driver.py \
  --envs "$ENVS" \
  --seeds "$SEEDS" \
  --task-ids "$TASK_IDS" \
  --eval-episodes-per-task "$EVAL_EPISODES_PER_TASK" \
  --variant "$VARIANT" \
  --fallback-mode "$FALLBACK_MODE" \
  --use-official-artifacts "$USE_OFFICIAL_ARTIFACTS" \
  --round "$ROUND" \
  --gpus "$GPUS" \
  --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --gas-repo-path "$GAS_REPO_PATH" \
  --run-root "$RUN_ROOT" \
  --reports-root "$REPORTS_ROOT" \
  --baseline-certification "$BASELINE_CERTIFICATION" \
  --force "$FORCE" \
  --eval-on-cpu "$EVAL_ON_CPU"
