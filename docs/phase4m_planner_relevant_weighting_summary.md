# Phase 4M Planner-Relevant Repair Loss Weighting Summary

Phase 4M tests a small planner-aware supervised loss-weighting change on Phase
4E repaired support graphs. It has now been run on Scene H5
`core_plus_bottleneck_budget192_H5`, Scene H10
`core_plus_bottleneck_budget192_H10`, Scene H25
`core_plus_bottleneck_budget192_H25`, and AntMaze H10
`core_plus_bottleneck_budget120_H10`. It does not run an environment rollout.

## Result

Compared with the same augmented graph trained using ordinary clipped
support+bottleneck weights:

| dataset/run | final val MSE ratio | direct repair MSE ratio | planner-used repair MSE ratio | policy support ratio |
| --- | ---: | ---: | ---: | ---: |
| AntMaze H10 B120 | 0.995 | 0.995 | 0.986 | 1.004 |
| Scene H25 B192 | 0.992 | 0.984 | 0.991 | 1.005 |
| Scene H10 B192 | 1.009 | 0.993 | 0.962 | 1.000 |
| Scene H5 B192 | 0.979 | 0.970 | 0.981 | 1.006 |
| mean | 0.994 | 0.985 | 0.980 | 1.003 |

Scene H5 assigns the highest mean weight to the 48 planner-used repair edges
(`1.217` mean loss weight). AntMaze assigns the highest repair-subgroup weight
to the 21 planner-used repair edges (`1.151` mean loss weight). Scene H10 shows
the same planner-used repair MSE improvement pattern but with a small overall
validation-MSE cost.

The follow-up Phase 4N Scene H10 regret sweep tests weaker planner-relevant
weights. `planner_relevant_repair_s02` is the best guarded Scene H10 setting:
it reduces the final validation-MSE ratio from `1.009` under `s04` to `1.006`,
while keeping planner-used repair-edge MSE at `0.960x` of the same augmented
support+bottleneck baseline.

Phase 4P extends the replication to Scene H25. There, `s04` is again the best
candidate: final validation MSE ratio is `0.992`, direct repair-edge MSE ratio
is `0.984`, planner-used repair-edge MSE ratio is `0.991`, and policy-support
score ratio is `1.005`. It is selected by Phase 4O's relaxed guard because the
planner-used improvement is positive but just misses the strict `0.990x`
threshold.

## Analysis

This is now a four-run offline supervised result. Planner relevance improves
direct repair-edge MSE in all four runs and improves planner-used repair-edge
MSE under the selected guard choice in all four runs. The overall validation
metric is more nuanced: it improves on AntMaze H10, Scene H5, and Scene H25,
but Scene H10 regresses under the aggressive `s04` setting. Phase 4N shows that
lowering the Scene H10 planner-relevance strength to `s02` controls this regret
while preserving the main planner-used repair-edge gain. The method is
therefore best treated as a targeted repair-edge objective with an explicit
overall-regret guard, not an unqualified default.

The result should be treated as an offline supervised proxy. It does not prove
closed-loop option execution. The remaining replication gaps are longer
training, additional environments, and eventual env-available rollout.

Artifacts:

- `results/phase4m/scene_play/core_plus_bottleneck_budget192_H5_3000/`
- `results/phase4m_training/scene_play/core_plus_bottleneck_budget192_H5_3000/`
- `results/phase4m/scene_play/core_plus_bottleneck_budget192_H10_3000/`
- `results/phase4m_training/scene_play/core_plus_bottleneck_budget192_H10_3000/`
- `docs/phase4n_planner_relevance_regret_guard_summary.md`
- `docs/phase4p_scene_h25_replication_summary.md`
- `results/phase4m/scene_play/core_plus_bottleneck_budget192_H25_3000/`
- `results/phase4m_training/scene_play/core_plus_bottleneck_budget192_H25_3000/`
- `results/phase4m/antmaze_large_stitch/core_plus_bottleneck_budget120_H10_3000/`
- `results/phase4m_training/antmaze_large_stitch/core_plus_bottleneck_budget120_H10_3000/`
- `results/phase4m/phase4m_replication_summary.csv`
