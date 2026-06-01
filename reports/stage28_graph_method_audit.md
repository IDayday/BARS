# Stage28 Graph Method Audit

## Goal

Stage28 stops treating GAS/BARS as a patch target and instead treats the cached BARS run as the strongest available graph-method baseline to diagnose. The purpose is to identify which mechanism actually explains graph offline GCRL failures before proposing a new algorithm.

The audit reuses existing run artifacts whenever possible:

- `cache/embeddings.npy`
- `cache/graph.npz`
- optional `cache/boundary.npz`
- optional `checkpoints/reachability.pt`

No low-level policy or representation retraining is required for the graph-only audit.

## Diagnostic layers

The audit records evidence at five layers:

1. Dataset support: trajectory count, length distribution, temporal support and future-pair sampling.
2. Graph abstraction: coverage distance, endpoint retention, weak/strong connectivity and zero-out-degree nodes.
3. Edge semantics: temporal support rate, cross-trajectory rate, p-exec/risk/cost distributions and low-cost high-confidence cross edges.
4. Path search: path existence, edge count, objective, cross-edge usage, largest-hop ratio and alternative-path proxy.
5. Failure taxonomy: proxy labels that map evidence to next algorithm families.

## Counterfactual graph variants

The Stage28 audit compares the cached BARS graph against support-preserving counterfactuals:

| graph_id | Purpose |
|---|---|
| `base_cached` | The existing BARS/GAS-aligned graph from the run directory. |
| `projection_temporal` | Raw same-trajectory support projected onto the cached node set; tests whether the abstraction preserves dataset paths. |
| `dense_knn` | Denser embedding-neighbor graph with temporal/projection support; tests whether current pruning/topology is the bottleneck. |
| `xy_knn` | XY-space counterfactual for antmaze-style tasks; tests whether learned representation geometry is the bottleneck. |
| `endpoint_aug` | Cached nodes plus trajectory endpoints; tests whether stitch endpoints were removed by abstraction. |
| `bottleneck_aug` | Cached nodes plus endpoint and projection-change states; tests whether bridge/bottleneck states are missing. |

These are diagnostics, not promoted algorithms.

## Failure taxonomy labels

| label | Interpretation | Candidate next algorithm family |
|---|---|---|
| `NO_DATA_PATH_AFTER_NODE_PROJECTION` | Same-trajectory data support is not preserved under current node projection. | Denser support graph or bridge generation; planner changes alone are unlikely to solve it. |
| `BASE_LOST_SUPPORTED_PATH_OR_EDGE_PRUNING` | Projection-temporal graph finds a path but cached BARS does not. | Bridge-preserving abstraction and support-critical edge preservation. |
| `BASE_USES_CROSS_TRAJ_SHORTCUT_FOR_SUPPORTED_PAIR` | Base path uses cross-trajectory shortcuts even for same-trajectory supported pairs. | Conservative connectedness / false-bridge detection. |
| `BASE_SINGLE_HOP_DOMINATED_PATH` | Base path objective is dominated by one long edge. | Long-hop validation, k-diverse paths, recovery planner. |
| `SINGLE_PATH_FRAGILITY_PROXY` | Path exists but removing early edges rarely leaves alternatives. | Path ensemble and execution-time path switching. |
| `GRAPH_PATH_EXISTS_EXECUTION_PROBE_NEEDED` | Graph evidence is not enough to explain failure. | Local edge rollout, execution monitor, goal-interface diagnosis. |
| `NO_GRAPH_PATH_UNRESOLVED` | None of the current proxies isolate the mechanism. | Inspect components and add environment-specific probes. |

## Single-run command

```bash
python scripts/stage28_graph_audit.py \
  --config configs/stage28_graph_method_audit.json \
  --run-dir runs/antmaze-medium-stitch-v0/full_bars/seed44_<stamp> \
  --env antmaze-medium-stitch-v0 \
  --seed 44 \
  --num-pairs 256 \
  --num-cross-pairs 128 \
  --clear
```

The command writes:

```text
<run-dir>/logs/stage28_graph_audit.csv
```

## Matrix command

```bash
LOG_ROOT=runs \
OUT_ROOT=runs_stage28_graph_audit \
VARIANT=full_bars \
ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0,antmaze-giant-navigate-v0,antmaze-giant-stitch-v0 \
SEEDS=44,45,46 \
NUM_PAIRS=256 \
NUM_CROSS_PAIRS=128 \
bash scripts/stage28_run_audit_matrix.sh
```

The matrix command writes per-cell CSVs under `runs_stage28_graph_audit/` and aggregates them into:

```text
runs_stage28_graph_audit/_analysis/stage28_audit_all.csv
runs_stage28_graph_audit/_analysis/stage28_graph_summary.csv
runs_stage28_graph_audit/_analysis/stage28_path_summary.csv
runs_stage28_graph_audit/_analysis/stage28_failure_taxonomy.csv
runs_stage28_graph_audit/_analysis/stage28_recommendations.md
```

## Promotion rule

Stage28 does not promote a new planner by default. It promotes only a research direction after the dominant failure mode is stable across seeds and environments:

- abstraction/path loss -> bridge-preserving graph abstraction;
- cross-trajectory shortcut evidence -> conservative connectedness;
- single-hop or low-diversity evidence -> k-diverse path ensemble and recovery;
- graph path exists but execution remains unknown -> local edge rollout and execution monitor;
- support absent -> bridge generation or model-based stitching rather than edge-cost tuning.
