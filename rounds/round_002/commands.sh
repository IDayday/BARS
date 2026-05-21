#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export D4RL_SUPPRESS_IMPORT_ERROR=1
python -m compileall bars scripts
python scripts/build_baseline_registry.py --round 002
bash scripts/certify_gas_baseline.sh ENVS=antmaze-medium-stitch-v0,antmaze-medium-navigate-v0 SEEDS=0 ROUND=002 USE_OFFICIAL_ARTIFACTS=1 FULL_BUDGET_ONLY=1
python scripts/reclassify_prior_evidence.py --baseline-cards reports/round_002_baseline_cards.jsonl --stage-reports reports --official-vs-adapter reports/round_002_official_vs_adapter.csv --out reports/round_002_prior_evidence_reclassification.csv --md-out reports/round_002_prior_evidence_reclassification.md
python scripts/analyze_baseline_certification.py --baseline-registry reports/round_002_baseline_registry.csv --baseline-cards reports/round_002_baseline_cards.jsonl --official-eval reports/round_002_gas_official_eval.csv --official-vs-adapter reports/round_002_official_vs_adapter.csv --prior-reclass reports/round_002_prior_evidence_reclassification.csv --gate-out reports/round_002_gate_status.json --md-out reports/round_002_baseline_certification.md
