#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$PWD}:$PWD"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false

MAX_ROUNDS="${MAX_ROUNDS:-8}"
MAX_WALL_HOURS_PER_ROUND="${MAX_WALL_HOURS_PER_ROUND:-12}"
MAX_GPU_HOURS_PER_ROUND="${MAX_GPU_HOURS_PER_ROUND:-72}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-12}"

python scripts/autoresearch_init.py \
  --max-rounds "$MAX_ROUNDS" \
  --max-wall-hours-per-round "$MAX_WALL_HOURS_PER_ROUND" \
  --max-gpu-hours-per-round "$MAX_GPU_HOURS_PER_ROUND" \
  --max-parallel-jobs "$MAX_PARALLEL_JOBS"

while true; do
  python scripts/autoresearch_run_round.py \
    --max-wall-hours "$MAX_WALL_HOURS_PER_ROUND" \
    --max-gpu-hours "$MAX_GPU_HOURS_PER_ROUND" \
    --max-parallel-jobs "$MAX_PARALLEL_JOBS"
  python scripts/autoresearch_analyze_round.py
  python scripts/autoresearch_plan_next.py
  bash scripts/autoresearch_package_round.sh

  GLOBAL_DECISION=$(python - <<'PY'
import json
from pathlib import Path
p = Path('research_state/bars_research_state.json')
state = json.loads(p.read_text())
print(state.get('global_decision', 'CONTINUE'))
PY
)
  if [[ "$GLOBAL_DECISION" != "CONTINUE" ]]; then
    echo "Autoresearch loop stopped with decision: $GLOBAL_DECISION"
    break
  fi
done
