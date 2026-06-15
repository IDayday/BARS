# Phase 5E Edge Outcome Memory Replanning

Phase 5E tests whether natural-start hierarchical rollout can reuse empirical
online edge outcomes across episodes. The goal is not to add unsupported
shortcuts or train a new policy. It is to persist which support-certified
option edges failed online and feed that evidence back into support-only
replanning.

## Algorithm Change

The hierarchical executor now records edge attempts from online traces. Each
attempt is keyed by:

```text
(segment_source, segment_edge_id, edge_src, edge_dst)
```

An attempt is marked completed if any step in the attempt enters the edge's
destination cluster. Otherwise it is a timeout. The output files are:

- `edge_attempts.csv`
- `edge_attempt_summary.csv`
- `edge_memory_update_summary.csv`
- a persistent memory CSV when `--edge_memory_csv --update_edge_memory` are set

When `--use_edge_memory` is enabled, previous failed edges are converted into
planner penalties through `failure_excess`, `timeouts`, or `attempts`. The
current smoke uses:

```text
cost = base_edge_cost + failure_penalty * failure_count
```

The planner still uses only support-certified graph and support-bank edges.
No kNN, proximity, latent-threshold, or random edges are introduced.

## Commands

```bash
MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5e_edge_memory_bootstrap_antmaze_corebot100k.yaml \
  --device cpu

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5e_edge_memory_replay_antmaze_corebot100k.yaml \
  --device cpu
```

## Results

| run | prior penalized edges | memory edges after run | penalized edges after run | success | steps | completed edges | final goal L2 | failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bootstrap | 0 | 6 | 5 | 0.0 | 104 | 1 | 38.3014 | `start_cluster_not_in_graph` |
| replay | 5 | 15 | 11 | 0.0 | 120 | 3 | 42.0989 | `max_steps_without_success` |

Outputs:

- `results/phase3f/antmaze_large_stitch/edge_memory_bootstrap_corebot100k_H10_B120/`
- `results/phase3f/antmaze_large_stitch/edge_memory_replay_corebot100k_H10_B120/`
- `results/phase3f/edge_memory/antmaze_corebot100k_H10_B120.csv`

## Analysis

The mechanism works: bootstrap produces failure memory, replay reads five prior
penalized edges, and the planner avoids or deprioritizes previously failed
edges. The replay trace completes more edges than bootstrap, but task success
remains zero and final goal distance worsens.

This is a useful negative result. A naive persistent failure penalty does not
solve online execution. It often pushes the planner toward different
support-certified edges that have not yet been attempted online, and those edges
can fail for the same closed-loop drift reasons. In other words, edge memory
adds empirical online evidence, but count-based avoidance alone is too coarse.

## Conclusion

Phase 5E is retained as instrumentation and as a future input to an edge success
model. It is not a complete algorithmic improvement by itself.

Next algorithm work should use this memory as supervised data for a calibrated
closed-loop edge outcome model. The model should condition on online state,
edge source, candidate segment initiation distance, policy action-MSE proxy, and
recent drift, rather than only applying a scalar penalty to edge ids.
