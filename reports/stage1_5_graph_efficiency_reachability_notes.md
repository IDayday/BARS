# BARS Stage 1.5: Graph Efficiency and Reachability Diagnostics Notes

This code snapshot adds Stage 1.5 infrastructure for two bottlenecks observed in Stage 1 v2:

1. graph construction dominated runtime;
2. edge diagnostics were dominated by cross-trajectory candidate edges while exact same-trajectory graph-edge positives were sparse.

## Main changes

### Graph efficiency

- Conservative CPU BLAS/thread caps are set by default in `bars/__init__.py` and propagated by `bars/sched/jobctl.py` to avoid multi-job CPU oversubscription.
- `bars/graph/nodes.py` supports landmark spectral bottleneck scoring:
  - solve spectral partitioning on a landmark subset;
  - assign the full support pool to landmark labels;
  - keep scoring and node selection over the full support pool.
- Config options:
  - `graph.spectral_solver`: `landmark` / `auto` / `full`
  - `graph.spectral_landmarks`
  - `graph.spectral_full_max_nodes`
  - `graph.spectral_eig_tol`
  - `graph.spectral_eig_maxiter`

This is intended to preserve graph quality while avoiding full eigensolve on the whole support set for every run.

### Reachability diagnostics

- `bars/graph/support.py` implements trajectory-support sampling: sample many same-trajectory segments, project endpoints to graph nodes, and count which graph edges receive direct dataset support.
- `bars/graph/diagnostics.py` reports:
  - `cross_traj_selected_rate` instead of interpreting all cross-trajectory edges as false positives;
  - `balanced_edge_diag` using support-mapped positives and support-missing hard-negative proxies;
  - score quantiles for supported, hard-negative-proxy, and unlabeled-bridge edges.
- Cross-trajectory supported edges are treated as `unlabeled_bridge_edges`, not negatives.

### Path diagnostics

- Path diagnostics support:
  - `path_min_graph_edges`;
  - `include_trivial_path_pairs=false`;
  - `max_sampling_attempts`;
  - `lambda_risk_values` sweep.
- This prevents zero-edge trivial pairs from dominating path-level mechanism diagnostics.

### Diagnostics-only mode

- `python -m bars.cli diagnose --config <config> --run-dir <existing_run_dir>` can rerun diagnostics from cached artifacts without retraining or rebuilding the graph.

## Smoke validation

A toy smoke run was validated with:

```bash
python -m compileall -q bars scripts
python -m bars.cli run --config configs/toy_smoke.json --run-dir /tmp/bars_smoke_final
```

It completed successfully and produced train/graph/diagnostics/eval/summary CSV files and an archive.
