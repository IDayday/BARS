# BARS Stage 1.5 Code Update Notes

This code update focuses on graph-build efficiency and diagnostics reliability before expanding to Stage 2/3.

## Main changes

1. Graph construction now supports landmark spectral bottleneck discovery.
   - `graph.spectral_solver=landmark`
   - `graph.spectral_landmarks=2048`
   - The full support pool is still scored; only the expensive Laplacian eigensolve is performed on landmarks.

2. Diagnostics are now support-aware.
   - `balanced_edge_diag` maps sampled same-trajectory segments onto graph edges.
   - Cross-trajectory selected edges are reported as `cross_traj_selected_rate`, not as true false positives.
   - Path diagnostics can exclude trivial start-node == goal-node pairs and perform lambda risk sweeps.

3. Reachability training now treats cross-trajectory pairs as down-weighted unlabeled/hard-negative pressure, not dominant negatives.
   - Same-trajectory short future pairs are positives.
   - Same-trajectory far-future pairs are horizon negatives.
   - Cross-trajectory random pairs have lower weight.
   - Latent-near cross-trajectory hard negatives have configurable weight.

4. Boundary scoring has a support/mode option.
   - `boundary.method=support_modes` samples dataset segments, maps them to graph edges, and builds departure/arrival mode histograms.
   - Direction boundary is kept as a fallback/baseline.

5. Diagnostics-only rerun is available.
   - `python -m bars.cli diagnose --config <run/config.json> --run-dir <run_dir> --clear-diagnostics --rebuild-boundary ...`
   - `scripts/rerun_stage15_diagnostics.sh runs_stage1_diag_v2` runs this for an existing Stage 1 v2 log root.

## Recommended next command

```bash
cd ~/remote/project/BARS
export D4RL_SUPPRESS_IMPORT_ERROR=1
bash scripts/rerun_stage15_diagnostics.sh runs_stage1_diag_v2
```

Then inspect:

```bash
sed -n '1,260p' reports/stage1_5_diagnostics.md
```
