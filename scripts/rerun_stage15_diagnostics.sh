#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${1:-runs_stage1_diag_v2}"
GPUS="${GPUS:-0,1,2,3}"
PACK="${PACK:-0}"

if [ ! -d "$LOG_ROOT" ]; then
  echo "[ERROR] log root not found: $LOG_ROOT" >&2
  exit 1
fi

export D4RL_SUPPRESS_IMPORT_ERROR=${D4RL_SUPPRESS_IMPORT_ERROR:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

run_count=0
for cfg in $(find "$LOG_ROOT" -mindepth 4 -maxdepth 4 -name config.json | sort); do
  run_dir="$(dirname "$cfg")"
  if [ ! -f "$run_dir/cache/embeddings.npy" ] || [ ! -f "$run_dir/cache/graph.npz" ]; then
    echo "[SKIP] missing cache: $run_dir"
    continue
  fi
  echo "[DIAG] $run_dir"
  args=(
    python -m bars.cli diagnose
    --config "$cfg"
    --run-dir "$run_dir"
    --clear-diagnostics
    --rebuild-boundary
    --set boundary.enabled=true
    --set boundary.method=support_modes
    --set boundary.load_if_exists=false
    --set boundary.support_segments=200000
    --set boundary.direction_fallback=true
    --set diagnostics.enabled=true
    --set diagnostics.balanced_edge_diag=true
    --set diagnostics.edge_support_segments=200000
    --set diagnostics.num_path_pairs=128
    --set diagnostics.path_min_graph_edges=1
    --set diagnostics.include_trivial_path_pairs=false
    --set diagnostics.max_sampling_attempts=10000
    --set 'diagnostics.lambda_risk_values=[0.0,0.1,0.3,1.0,3.0]'
    --set 'diagnostics.planner_variants=["shortest","reachability"]'
  )
  if [ "$PACK" = "1" ]; then
    args+=(--package)
  fi
  "${args[@]}"
  run_count=$((run_count + 1))
done

echo "[INFO] diagnostics-only runs processed: $run_count"
python scripts/collect_csv.py --log-root "$LOG_ROOT"
python scripts/analyze_bars_results.py --log-root "$LOG_ROOT" --stage stage1 --force-collect --out reports/stage1_5_diagnostics.md

echo "[INFO] wrote reports/stage1_5_diagnostics.md"
