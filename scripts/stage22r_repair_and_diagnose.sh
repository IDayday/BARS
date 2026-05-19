#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

ENVS="${ENVS:-antmaze-medium-stitch-v0,antmaze-medium-navigate-v0}"
SEEDS="${SEEDS:-0}"
WAIT_FOR_SWEEP="${WAIT_FOR_SWEEP:-1}"

if [[ "$WAIT_FOR_SWEEP" == "1" ]]; then
  python scripts/stage22_monitor.py \
    --roots runs_stage22_prepare,runs_stage22_reachability,runs_stage22_eval_logs,runs_stage22_eval \
    --summary-md reports/stage22_live_summary.md \
    --wait || true
fi

python scripts/analyze_stage22.py \
  --eval-root runs_stage22_eval \
  --artifact-root artifacts/stage22 \
  --out reports

cp reports/stage22_summary.md reports/stage22_finalized_summary.md
cp reports/stage22_grouped.csv reports/stage22_finalized_grouped.csv
cp reports/stage22_variant_comparison.csv reports/stage22_finalized_variant_comparison.csv

python -m bars.gas_bars.boundary_feasibility \
  --envs "$ENVS" --seeds "$SEEDS" \
  --artifact-root artifacts/stage22 \
  --eval-root runs_stage22_eval \
  --out reports

python -m bars.gas_bars.selected_edge_diagnostics \
  --envs "$ENVS" --seeds "$SEEDS" \
  --artifact-root artifacts/stage22 \
  --eval-root runs_stage22_eval \
  --out reports

python -m bars.gas_bars.risk_calibration \
  --envs "$ENVS" --seeds "$SEEDS" \
  --artifact-root artifacts/stage22 \
  --eval-root runs_stage22_eval \
  --out reports/stage22r_recommended_budgets.json \
  --report reports/stage22r_risk_calibration.md

python -m bars.gas_bars.failure_slice \
  --envs "$ENVS" --seeds "$SEEDS" \
  --eval-root runs_stage22_eval \
  --artifact-root artifacts/stage22 \
  --out reports

python scripts/analyze_stage22r_decisions.py \
  --reports-dir reports \
  --out reports/stage22r_decisions.md

cp reports/stage22r_decisions.md reports/stage22_finalized_decisions.md
