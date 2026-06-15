# Phase 5F Edge Outcome Model

Phase 5F upgrades Phase 5E's persistent edge memory from hard failure-count
avoidance to a smoother edge outcome prior. The goal is to use online evidence
without turning one failed attempt into an overly large path-level cost jump.

## Reviewed Context

- `docs/algorithm_improvement_attempts.md`, especially Phase 5D and Phase 5E.
- Phase 5D showed policy-aware offline action MSE helps choose cleaner support
  segments, but does not solve online execution.
- Phase 5E showed persistent edge memory works mechanically, but naive
  count-based penalties only push the planner toward other untested failing
  edges.

Phase 5F keeps the same support-only constraint: no kNN, proximity,
latent-threshold, or random edges are added.

## Algorithm Change

Phase 5F adds `phase3f/edge_outcome_model.py`.

For every persisted edge memory row, it computes:

```text
posterior_success_prob = (completed + alpha) / (attempts + alpha + beta)
posterior_failure_prob = 1 - posterior_success_prob
edge_outcome_risk_score =
    posterior_failure_prob
  + uncertainty_weight * posterior_uncertainty
  + optional policy_mse_risk
  + optional subgoal_l2_risk
edge_outcome_penalty = outcome_penalty_weight * edge_outcome_risk_score
```

The hierarchical planner can now add this continuous `edge_outcome_penalty` to
edge cost. Phase 5F also keeps a moderate within-episode `failure_penalty` so a
single run does not keep retrying the same failed edge.

Phase 5F also fixes an important Phase 5E metric issue: repeated attempts of the
same edge in one episode are now split when `edge_step` resets. Previously, a
replanned retry of the same edge could be merged into one long attempt.

## Commands

```bash
conda run -n gcrlo pytest -q tests/test_phase3_synthetic.py

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5f_edge_outcome_antmaze_corebot100k.yaml \
  --device cpu
```

## Results

Single AntMaze natural-start smoke, task id 1, seed 0:

| method | success | steps | completed edges | replans | failed edge attempts | final goal L2 | failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Phase 5D policy-aware | 0.0 | 119 | 2 | 9 | 8 | 40.8405 | `max_replans_exceeded` |
| Phase 5E bootstrap | 0.0 | 104 | 1 | 5 | 5 | 38.3014 | `start_cluster_not_in_graph` |
| Phase 5E replay | 0.0 | 120 | 3 | 6 | 5 | 42.0989 | `max_steps_without_success` |
| Phase 5F outcome model | 0.0 | 120 | 3 | 6 | 5 | 41.2900 | `max_steps_without_success` |

Output:

- `results/phase3f/antmaze_large_stitch/edge_outcome_corebot100k_H10_B120/`

## Diagnostics

The outcome model loaded 15 scored memory edges and assigned penalties to all
15. During the smoke run, the planner used both penalized known-risk edges and
unpenalized previously unseen support-bank edges.

The corrected attempt summary shows:

- `bank:5` failed once.
- `bank:15` failed once.
- `bank:11` completed twice.
- several additional support-bank edges failed once.

This reveals the main limitation: online memory only covers edges already
attempted. The planner still has a large untested support-bank action space, so
it can avoid known failures and move into different untested failures.

## Conclusion

Phase 5F is an engineering and metric-semantics improvement. It is slightly
better than Phase 5E replay on this smoke's final goal distance, but it is not a
breakthrough and does not improve task success.

The useful pattern is:

1. Keep within-episode failure avoidance.
2. Use smooth cross-episode outcome priors instead of hard count penalties.
3. Treat unseen edges as the next modeling problem.

The next algorithmic step should estimate risk for unseen support edges before
online attempts. Candidate signals include offline edge support, direct GCBC
action fitting on the edge's segments, current-state initiation distance,
planner usage, and compatibility context.
