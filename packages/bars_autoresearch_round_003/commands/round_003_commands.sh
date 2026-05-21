#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD
export WANDB_MODE=disabled
export WANDB_DISABLED=true
python scripts/round003_audit_official_gas_artifacts.py
bash scripts/round003_run_official_gas_eval.sh ENVS=antmaze-giant-stitch-v0,antmaze-large-explore-v0,scene-play-v0 SEEDS=0 TASK_IDS=1,2,3,4,5 EVAL_EPISODES_PER_TASK=50 USE_OFFICIAL_ARTIFACTS=1 FULL_BUDGET_ONLY=1 ROUND=003
python scripts/round003_analyze_certification.py
bash scripts/round003_run_bars_adapter_eval.sh ENVS=$(python - <<'PY'
import json
x=json.load(open('reports/round_003_baseline_certification.json'))
print(','.join(x.get('certified_envs', [])))
PY
) SEEDS=0 TASK_IDS=1,2,3,4,5 EVAL_EPISODES_PER_TASK=50 VARIANT=gas_shortest_official_control FALLBACK_MODE=none USE_OFFICIAL_ARTIFACTS=1 ROUND=003
python scripts/round003_compare_official_vs_adapter.py
python scripts/round003_finalize.py
