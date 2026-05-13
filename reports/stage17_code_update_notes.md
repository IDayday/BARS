# BARS Stage 17 Code Update Notes

This update is built on the uploaded current BARS code package and targets the next experiment round: edge rollout validation, PU reachability retraining, and quick online evaluation once diagnostics are credible.

## Why code changes were needed

The uploaded package already included Stage 1.6 graph-build optimization, support-mode boundary diagnostics, balanced edge diagnostics, and path risk-cost sweeps. However, the next round requires two missing capabilities:

1. **Selective warmstart**: The previous warmstart copied `reachability.pt` unconditionally. This makes a PU-retraining ablation invalid because the run silently loads the old reachability checkpoint instead of retraining.
2. **Stratified edge rollout diagnostics**: The previous edge rollout diagnostic only compared selected vs. unselected edges. It did not separately evaluate selected supported edges, selected unlabeled bridge candidates, and selected hard-negative proxies, which are the key groups for validating cross-trajectory stitching.

## Implemented changes

### 1. Selective warmstart artifacts

`bars/experiments/pipeline.py` now supports:

```json
"experiment": {
  "warmstart_root": "runs_stage16_full12",
  "warmstart_source_variant": "full_bars",
  "warmstart_artifacts": ["tdr", "policy", "embeddings"]
}
```

Valid artifact names:

```text
tdr, policy, reachability, embeddings, graph, boundary, all
```

This enables PU retraining with TDR/policy/embeddings reused but reachability excluded.

### 2. Stratified edge rollout diagnostics

`bars/eval/edge_rollout_diag.py` now supports support-aware edge groups:

```text
selected_supported
selected_unlabeled_bridge
selected_hard_neg_proxy
unselected_supported
unselected_hard_neg_proxy
```

It logs group-wise counts, success rates, mean p_exec, and final distances in addition to overall AUC/AUPRC.

### 3. Diagnostics-only edge rollout helper

Added:

```text
scripts/run_stage16_edge_rollout_diagnose.sh
```

This script runs edge rollout diagnostics on existing `runs_stage16_full12` run directories without retraining or rebuilding graphs. It can use multiple GPUs concurrently.

### 4. New sweeps for the next round

Added:

```text
configs/sweeps/d4rl_stage16_pu_retrain4.json
configs/sweeps/d4rl_stage16_pu_retrain12.json
configs/sweeps/d4rl_stage16_edge_rollout_full12_diagnose.json
configs/sweeps/d4rl_stage16_online_quick_loaded.json
```

The `pu_retrain` sweeps warmstart TDR/policy/embeddings but force reachability retraining. The online quick sweep warmstarts full graph artifacts from `runs_stage16_full12` and evaluates shortest/reachability/full_bars.

### 5. Analysis support

`scripts/analyze_bars_results.py` now includes the new edge-rollout group metrics in the generated `stage*_edge_rollout_summary.csv`.

## Validation performed

Commands run in an isolated copy:

```bash
python -m compileall -q bars scripts
python -m bars.cli run --config configs/toy_smoke.json --run-dir /tmp/bars_mod_smoke --set boundary.method=support_modes --set boundary.support_segments=256 --set diagnostics.edge_rollout_enabled=false
python scripts/collect_csv.py --log-root /tmp/bars_mod_smoke
python scripts/analyze_bars_results.py --log-root /tmp/bars_mod_smoke --stage stage1 --out /tmp/bars_mod_report.md --force-collect
```

All commands completed successfully.

## Recommended next experiment order

1. Commit/push current code.
2. Run diagnostics-only edge rollout on `runs_stage16_full12`.
3. If edge rollout supports reachability, run `d4rl_stage16_pu_retrain4.json`.
4. If PU retrain improves balanced/rollout diagnostics, expand to `d4rl_stage16_pu_retrain12.json`.
5. Only then run quick online eval.
