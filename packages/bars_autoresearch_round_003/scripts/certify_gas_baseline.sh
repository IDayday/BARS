#!/usr/bin/env bash
set -euo pipefail

ENVS="${ENVS:-antmaze-medium-stitch-v0,antmaze-medium-navigate-v0}"
SEEDS="${SEEDS:-0}"
ROUND="${ROUND:-002}"
REPORTS_ROOT="${REPORTS_ROOT:-reports}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
MAX_GAP_PP="${MAX_GAP_PP:-2.0}"

for arg in "$@"; do
  case "$arg" in
    ENVS=*) ENVS="${arg#ENVS=}" ;;
    SEEDS=*) SEEDS="${arg#SEEDS=}" ;;
    ROUND=*) ROUND="${arg#ROUND=}" ;;
    REPORTS_ROOT=*) REPORTS_ROOT="${arg#REPORTS_ROOT=}" ;;
    GAS_ARTIFACT_ROOT=*) GAS_ARTIFACT_ROOT="${arg#GAS_ARTIFACT_ROOT=}" ;;
    MAX_GAP_PP=*) MAX_GAP_PP="${arg#MAX_GAP_PP=}" ;;
    USE_OFFICIAL_ARTIFACTS=*|FULL_BUDGET_ONLY=*|EPISODES_PER_GOAL=*) ;;
  esac
done

ROUND_NUM="$(printf '%03d' "${ROUND#0}")"

python scripts/run_official_gas_eval.py \
  --envs "$ENVS" \
  --seeds "$SEEDS" \
  --round "$ROUND_NUM" \
  --reports-root "$REPORTS_ROOT" \
  --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
  --official-out "$REPORTS_ROOT/round_${ROUND_NUM}_gas_official_eval.csv" \
  --adapter-out "$REPORTS_ROOT/round_${ROUND_NUM}_bars_adapter_eval.csv"

python scripts/compare_official_vs_bars_adapter.py \
  --official "$REPORTS_ROOT/round_${ROUND_NUM}_gas_official_eval.csv" \
  --adapter "$REPORTS_ROOT/round_${ROUND_NUM}_bars_adapter_eval.csv" \
  --out "$REPORTS_ROOT/round_${ROUND_NUM}_official_vs_adapter.csv" \
  --md-out "$REPORTS_ROOT/round_${ROUND_NUM}_adapter_mismatch_report.md" \
  --max-gap-pp "$MAX_GAP_PP"
