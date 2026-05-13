# BARS Stage 1.6 Verified Code Audit

This package was audited after the Stage 1.6 implementation pass.

## Verified implemented modules

- `support_modes` portal-boundary overlap in `bars/graph/boundary.py`.
- Edge rollout diagnostics in `bars/eval/edge_rollout_diag.py`, including reset availability and selected/unselected success metrics.
- Optional FAISS/sklearn ANN abstraction in `bars/graph/ann.py`.
- Graph/pipeline profile logging in `bars/common/profile.py` and `logs/profile.csv`.
- PU-style reachability training in `bars/training/reach_train.py`.
- Balanced support edge diagnostics in `bars/graph/diagnostics.py` and `bars/graph/support.py`.
- Nontrivial path sampling, lambda-risk sweep, and diagnostics-only reruns in CLI/pipeline.
- Stage 1.6 sweeps for sanity4, full12, edge rollout quick, and conditional online quick eval.

## Fixes applied during audit

- Patched `bars/common/checkpoint.py` to load checkpoints with `weights_only=False` when available, fixing PyTorch 2.6+ checkpoint reloads that include NumPy normalizer metadata.
- Patched `bars/eval/edge_rollout_diag.py` to log `selected_edge_success_rate`, `unselected_edge_success_rate`, selected/unselected counts, and reset-unavailable counts.
- Patched `bars/experiments/pipeline.py` so `bars.cli diagnose` can optionally run edge rollout diagnostics by loading the cached policy checkpoint.
- Patched `scripts/analyze_bars_results.py` to summarize edge rollout diagnostics.
- Added `direction_fallback_weight` support to `BoundaryIndex` for optional blending of support-mode overlap with direction fallback.
- Updated Stage 1.6 sweep configs and added `d4rl_stage16_online_quick.json`.

## Local validation

The following commands were run locally in the audit environment:

```bash
python -m compileall -q bars scripts
python -m bars.cli run --config configs/toy_smoke.json --run-dir /tmp/bars_final_smoke \
  --set boundary.method=support_modes \
  --set boundary.support_segments=512 \
  --set boundary.support_k=16 \
  --set boundary.num_modes=3
python -m bars.cli diagnose --config configs/toy_smoke.json --run-dir /tmp/bars_final_smoke \
  --clear-diagnostics --rebuild-boundary \
  --set boundary.method=support_modes \
  --set boundary.support_segments=256 \
  --package
python scripts/collect_csv.py --log-root /tmp/bars_final_smoke
python scripts/analyze_bars_results.py --log-root /tmp/bars_final_smoke --stage stage1 \
  --out /tmp/bars_final_smoke_report.md --force-collect
python -m bars.sched.jobctl launch --sweep configs/sweeps/d4rl_stage16_sanity4.json \
  --log-root /tmp/bars_dry --gpus 0,1,2,3,4,5,6 --max-jobs-per-gpu 1 --dry-run
```

The toy run and diagnostics-only rerun completed, and analysis generated a report. D4RL was not run in this audit environment.
