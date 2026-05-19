#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

MODE="${MODE:-pilot}"
WAIT="${WAIT:-1}"
NO_WAIT="${NO_WAIT:-0}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
GPUS="${GPUS:-0}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/stage22}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export BARS_USE_TENSORBOARD="${BARS_USE_TENSORBOARD:-1}"
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-runs_stage22_tensorboard}"

case "$MODE" in
  pilot)
    ENVS="${ENVS:-antmaze-medium-stitch-v0,antmaze-medium-navigate-v0}"
    SEEDS="${SEEDS:-0}"
    EPISODES="${EPISODES:-50}"
    QUICK="${QUICK:-1}"
    VARIANTS="${VARIANTS:-gas_shortest,gas_reachability_budget,gas_boundary_budget}"
    BUDGETS="${BUDGETS:-2.0,3.0}"
    FALLBACK_MODES="${FALLBACK_MODES:-none,progress_stall_v2}"
    ;;
  confirm)
    ENVS="${ENVS:-antmaze-medium-stitch-v0,antmaze-medium-navigate-v0,antmaze-large-stitch-v0,antmaze-large-navigate-v0}"
    SEEDS="${SEEDS:-0,1,2}"
    EPISODES="${EPISODES:-100}"
    QUICK="${QUICK:-0}"
    VARIANTS="${VARIANTS:-gas_shortest,gas_reachability_budget,gas_boundary_budget}"
    BUDGETS="${BUDGETS:-1.2,1.6,2.0,2.3,3.0,5.0}"
    FALLBACK_MODES="${FALLBACK_MODES:-none,progress_stall_v2}"
    ;;
  full)
    ENVS="${ENVS:-antmaze-medium-stitch-v0,antmaze-medium-navigate-v0,antmaze-large-stitch-v0,antmaze-large-navigate-v0,antmaze-medium-play-v2,antmaze-medium-diverse-v2,antmaze-large-play-v2,antmaze-large-diverse-v2}"
    SEEDS="${SEEDS:-0,1,2}"
    EPISODES="${EPISODES:-100}"
    QUICK="${QUICK:-0}"
    VARIANTS="${VARIANTS:-gas_shortest,gas_reachability_soft,gas_reachability_budget,gas_boundary_soft,gas_boundary_budget}"
    BUDGETS="${BUDGETS:-1.2,1.6,2.0,2.3,3.0,5.0}"
    FALLBACK_MODES="${FALLBACK_MODES:-none,progress_stall_v2,no_path}"
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

export ENVS SEEDS GPUS EPISODES QUICK PREFER_PRETRAINED TRAIN_IF_MISSING
export GAS_ARTIFACT_ROOT ARTIFACT_ROOT GAS_REPO_PATH
export ARTIFACT_ROOT

echo "[stage22_pipeline] MODE=$MODE ENVS=$ENVS SEEDS=$SEEDS EPISODES=$EPISODES QUICK=$QUICK"

ARTIFACT_ROOT="$GAS_ARTIFACT_ROOT" LOG_ROOT="runs_stage22_prepare" bash scripts/stage22_prepare_gas_backbone.sh
prep_rc=$?

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/stage22}" GAS_ARTIFACT_ROOT="$GAS_ARTIFACT_ROOT" LOG_ROOT="runs_stage22_reachability" bash scripts/stage22_train_reachability.sh
reach_rc=$?

export VARIANTS BUDGETS FALLBACK_MODES
STAGE22_ROOT="${STAGE22_ROOT:-runs_stage22_eval}" LOG_ROOT="runs_stage22_eval_logs" bash scripts/stage22_eval_variants.sh
eval_rc=$?

mkdir -p reports
python scripts/stage22_monitor.py --roots runs_stage22_prepare,runs_stage22_reachability,runs_stage22_eval_logs,runs_stage22_eval --summary-md reports/stage22_live_summary.md
python scripts/analyze_stage22.py --eval-root runs_stage22_eval --artifact-root artifacts/stage22 --out reports

if [[ "$NO_WAIT" == "1" || "$WAIT" == "0" ]]; then
  echo "[stage22_pipeline] --no-wait requested via WAIT=0/NO_WAIT=1; pipeline stages were launched sequentially and current summaries are written."
else
  python scripts/stage22_monitor.py --roots runs_stage22_prepare,runs_stage22_reachability,runs_stage22_eval_logs,runs_stage22_eval --wait --summary-md reports/stage22_live_summary.md
fi

if [[ $prep_rc -ne 0 || $reach_rc -ne 0 || $eval_rc -ne 0 ]]; then
  echo "[stage22_pipeline] One or more stages reported a shell-level failure: prepare=$prep_rc reach=$reach_rc eval=$eval_rc" >&2
  exit 1
fi
