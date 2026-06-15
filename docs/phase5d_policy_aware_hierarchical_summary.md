# Phase 5D Policy-Aware Hierarchical Rollout

Phase 5D tests whether policy-aware segment and replan scoring improves the
reset-free hierarchical natural-start executor from Phase 5C.

## Related Work Checked

- SoRB builds replay-buffer graphs but weights graph edges with a learned
  goal-conditioned value/reachability signal, which supports the idea that graph
  edges need policy-aware reachability, not just geometric proximity:
  https://arxiv.org/abs/1906.05253
- HIQL decomposes offline goal-reaching into high-level subgoal prediction and a
  low-level policy, supporting our separation between graph planning and
  low-level option execution:
  https://arxiv.org/abs/2307.11949

Phase 5D is not a reimplementation of either method. It keeps BARS support-only
graph semantics and adds policy-aware diagnostics to support edge execution.

## Algorithm Change

For every candidate real segment of a support edge, Phase 5D scores the segment
termination subgoal with:

```text
score =
    initiation_weight * ||online_obs - segment_initiation||
  + downstream_weight * ||segment_termination - next_edge_initiation_or_goal||
  + policy_mse_weight * MSE(pi(segment_initiation, segment_termination), offline_action) / policy_mse_scale
```

The chosen subgoal is still a real offline segment termination. No kNN,
proximity, or learned latent shortcut edge is added.

Phase 5D also adds failure-penalized replanning. When an edge reaches its
horizon without entering the destination cluster, future replans add a cost
penalty to that exact support edge.

Runtime cluster models can now be cached. Scene kmeans cache was verified with a
second run that reported `cluster_cache_hit: true`.

## Results

| dataset | method | episodes | success | mean completed edges | mean final goal L2 | mean replans | failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AntMaze | Phase 5C `hierarchical_repaired_corebot100k_H10_B120` | 2 | 0.0 | 1.0 | 28.8081 | 9.0 | `max_replans_exceeded` |
| AntMaze | Phase 5D `policy_aware_hierarchical_corebot100k_H10_B120` | 2 | 0.0 | 1.5 | 27.7101 | 9.0 | `max_replans_exceeded` |
| Scene | Phase 5C `hierarchical_repaired_phase4o_s04_H25_B192` | 1 | 0.0 | 0.0 | 5.9754 | 2.0 | `max_steps_without_success` |
| Scene | Phase 5D `policy_aware_hierarchical_phase4o_s04_H25_B192` | 1 | 0.0 | 0.0 | 6.9507 | 2.0 | `max_steps_without_success` |

Outputs:

- `results/phase3f/antmaze_large_stitch/policy_aware_hierarchical_corebot100k_H10_B120/`
- `results/phase3f/scene_play/policy_aware_hierarchical_phase4o_s04_H25_B192/`
- `results/phase3f/scene_play/policy_aware_hierarchical_scene_cache_probe/`

## Diagnostics

The policy-aware score is doing what it was designed to do:

- AntMaze first trace: selected policy action MSE `0.0979`, mean candidate MSE
  `0.4142`.
- Scene first trace: selected policy action MSE `0.00475`, mean candidate MSE
  `0.02347`.

However, lower offline policy MSE did not translate into online task success.
AntMaze improved modestly on completed edges and final goal distance, but still
hit the replan limit. Scene did not improve in this smoke setting.

## Conclusion

Phase 5D is a useful diagnostic and a small AntMaze improvement, not a complete
algorithmic breakthrough. The current bottleneck is no longer just graph edge
selection. It is closed-loop option execution from off-distribution online
states.

Next work should treat policy-aware MSE as one risk feature, not the whole
execution model:

1. Learn a dedicated edge success/reachability model from offline plus online
   failure traces.
2. Penalize edges by empirical online failure counts across episodes, not only
   within a single episode.
3. Train or fine-tune the low-level policy on planner-used and recovery states.
4. Add local recovery when the online state drifts away from the selected
   segment initiation.
5. Run multi-seed natural-start comparisons once the executor is stable.
