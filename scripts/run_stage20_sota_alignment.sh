#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${MODE:-protocol}"
GPUS="${GPUS:-0}"
LOG_ROOT="${LOG_ROOT:-runs_stage20}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
EPISODES="${EPISODES:-100}"
SEEDS="${SEEDS:-0,1,2}"
ENVS="${ENVS:-antmaze-medium-play-v2,antmaze-medium-diverse-v2}"
LARGE_ENVS="${LARGE_ENVS:-antmaze-large-play-v2,antmaze-large-diverse-v2}"
WARMSTART_ROOT="${WARMSTART_ROOT:-runs_stage16_full12}"
DEFAULT_MEM_MB="${DEFAULT_MEM_MB:-6000}"

export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python scripts/make_stage20_sota_sweeps.py \
  --out-dir configs/sweeps \
  --envs "$ENVS" \
  --large-envs "$LARGE_ENVS" \
  --seeds "$SEEDS" \
  --episodes "$EPISODES" \
  --warmstart-root "$WARMSTART_ROOT" \
  --default-mem-mb "$DEFAULT_MEM_MB"

case "$MODE" in
  protocol)
    SWEEP="configs/sweeps/d4rl_stage20_protocol_fix_medium100.json"
    ;;
  budget)
    SWEEP="configs/sweeps/d4rl_stage20_budget_sweep.json"
    ;;
  gas)
    : "${GAS_ARTIFACT_ROOT:?Set GAS_ARTIFACT_ROOT, e.g. /path/to/gas_artifacts}"
    : "${GAS_REPO_PATH:?Set GAS_REPO_PATH, e.g. external_src/GAS}"
    : "${GAS_POLICY_CKPT_ROOT:?Set GAS_POLICY_CKPT_ROOT, e.g. /path/to/gas_policy_ckpts}"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    SWEEP="configs/sweeps/d4rl_stage20_routeb_gas_same_backbone.json"
    ;;
  hiql)
    : "${HIQL_REPO_PATH:?Set HIQL_REPO_PATH, e.g. external_src/HIQL}"
    : "${HIQL_POLICY_CKPT_ROOT:?Set HIQL_POLICY_CKPT_ROOT, e.g. /path/to/hiql_ckpts}"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    SWEEP="configs/sweeps/d4rl_stage20_routeb_hiql_policy.json"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use protocol|budget|gas|hiql" >&2
    exit 2
    ;;
esac

python -m bars.sched.jobctl launch \
  --sweep "$SWEEP" \
  --log-root "$LOG_ROOT" \
  --gpus "$GPUS" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds "${POLL_SECONDS:-10}" \
  --launch-burst "${LAUNCH_BURST:-1}" \
  ${BACKGROUND:+--background} \
  ${DRY_RUN:+--dry-run}
