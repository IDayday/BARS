# Phase 5G Offline-Informed Edge Prior

Phase 5G addresses the Phase 5F blind spot: online edge memory only scores
edges already attempted online. The planner can still move into previously
unseen support-bank edges that fail for similar closed-loop drift reasons.

## Related Work Checked

- SoRB frames replay-buffer graph search around learned local reachability and
  a local policy, not pure geometry: https://arxiv.org/abs/1906.05253
- HIQL motivates hierarchical goal reaching because nearby subgoals are easier
  to assess than distant goals: https://arxiv.org/abs/2307.11949
- TTGS uses graph search at test time for offline GCRL and explicitly treats
  distance/cost quality as central to long-horizon subgoal execution:
  https://arxiv.org/abs/2510.07257

Phase 5G is not a reimplementation of these methods. It keeps BARS's support
certification constraint and adds an offline risk prior for every support edge
visible to the natural-start planner.

## Algorithm Change

Added `phase3f/offline_edge_prior.py`.

For graph and support-bank edges, Phase 5G builds:

```text
offline_edge_prior_reliability =
    weighted average of:
      certification_reliability
      support_reliability
      diversity_reliability
      horizon_reliability
      policy_reliability
      compatibility_reliability

offline_edge_prior_penalty =
    offline_prior_penalty_weight * (1 - offline_edge_prior_reliability)
```

The certification signal is read from Phase 4G when available:

```text
results/phase4g/antmaze_large_stitch/
  core_plus_bottleneck_budget120_H10__repair_all_budget120_H10/
  planner_direct_repair_edge_certification.csv
```

If an edge lacks certification, it still receives a prior from Phase 2 support
metadata: support count, unique starts/episodes, and median horizon. This is the
critical difference from Phase 5F: unseen support-bank edges no longer have zero
risk by default.

## Commands

```bash
conda run -n gcrlo pytest -q tests/test_phase3_synthetic.py

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5g_offline_prior_antmaze_corebot100k.yaml \
  --device cpu
```

## Results

AntMaze natural-start, task id 1, seeds 0/1/2, 120-step cap:

| method | episodes | success | mean final goal L2 | mean completed edges | mean replans | mean failed edge attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5F outcome-only | 3 | 0.0 | 43.0624 | 1.667 | 7.333 | 6.333 |
| Phase 5G offline prior + outcome | 3 | 0.0 | 42.0333 | 0.333 | 7.667 | 6.667 |

Outputs:

- `results/phase3f/antmaze_large_stitch/edge_outcome_3ep_corebot100k_H10_B120/`
- `results/phase3f/antmaze_large_stitch/offline_prior_3ep_corebot100k_H10_B120/`

Phase 5G scored 2151 planner-visible edge keys:

- median `offline_edge_prior_reliability`: `0.4571`
- median `offline_edge_prior_penalty`: `0.5429`
- lowest-reliability edges are sparse support-bank edges with few starts,
  episodes, and long horizons.

Quick one-episode penalty-weight sweep:

| offline prior penalty weight | final goal L2 | completed edges | replans |
| ---: | ---: | ---: | ---: |
| 1.0 | 39.9252 | 2 | 5 |
| 2.0 | 40.4665 | 2 | 6 |
| 4.0 | 42.5091 | 1 | 8 |

This suggests the offline prior is useful only as a mild bias. Overweighting it
can suppress edge progress and increase replanning.

## Analysis

Phase 5G improves mean final goal distance against the Phase 5F outcome-only
baseline in the small 3-episode smoke, but it still produces zero task success
and fewer completed option edges. This is a real trade-off: the offline prior
pushes the planner away from weak support/unseen edges, but the low-level policy
still struggles to complete the conservative route.

The result supports a direction, not a completed algorithm:

1. Every support edge should have a risk prior before online attempts.
2. That prior should be mild; strong risk penalties make the graph too brittle.
3. Edge completion and final-goal distance can move in opposite directions, so
   both must remain primary metrics.

## Conclusion

Phase 5G is the first Phase 5 step that addresses unseen-edge risk directly. It
is more mature than pure online memory, but it is not sufficient for online task
success.

The next breakthrough likely requires a state-conditioned edge success model:
the risk of edge `e` should depend on the current online observation, distance
to candidate segment initiations, policy action fit, offline support, and
compatibility context. A static edge id prior is too coarse once the agent has
drifted away from the offline initiation manifold.
