# Phase 4M Planner-Relevant Repair Loss Weighting Summary

Phase 4M tests a small planner-aware supervised loss-weighting change on Phase
4E repaired support graphs. It has now been run on Scene H5
`core_plus_bottleneck_budget192_H5`, Scene H10
`core_plus_bottleneck_budget192_H10`, and AntMaze H10
`core_plus_bottleneck_budget120_H10`. It does not run an environment rollout.

## Result

Compared with the same augmented graph trained using ordinary clipped
support+bottleneck weights:

| dataset/run | final val MSE ratio | direct repair MSE ratio | planner-used repair MSE ratio | policy support ratio |
| --- | ---: | ---: | ---: | ---: |
| AntMaze H10 B120 | 0.995 | 0.995 | 0.986 | 1.004 |
| Scene H10 B192 | 1.009 | 0.993 | 0.962 | 1.000 |
| Scene H5 B192 | 0.979 | 0.970 | 0.981 | 1.006 |
| mean | 0.994 | 0.986 | 0.977 | 1.003 |

Scene H5 assigns the highest mean weight to the 48 planner-used repair edges
(`1.217` mean loss weight). AntMaze assigns the highest repair-subgroup weight
to the 21 planner-used repair edges (`1.151` mean loss weight). Scene H10 shows
the same planner-used repair MSE improvement pattern but with a small overall
validation-MSE cost.

## Analysis

This is now a three-run offline supervised result. Planner relevance improves
direct repair-edge MSE and planner-used repair-edge MSE in all three runs. The
overall validation metric is more nuanced: it improves on AntMaze H10 and Scene
H5, but Scene H10 regresses by about 0.9%. This suggests the method is doing
what it was designed to do, but the planner-focused weight needs a regret guard
or schedule before becoming a default training recipe.

The result should be treated as an offline supervised proxy. It does not prove
closed-loop option execution. The remaining replication gaps are Scene H25,
longer training, and eventual env-available rollout.

Artifacts:

- `results/phase4m/scene_play/core_plus_bottleneck_budget192_H5_3000/`
- `results/phase4m_training/scene_play/core_plus_bottleneck_budget192_H5_3000/`
- `results/phase4m/scene_play/core_plus_bottleneck_budget192_H10_3000/`
- `results/phase4m_training/scene_play/core_plus_bottleneck_budget192_H10_3000/`
- `results/phase4m/antmaze_large_stitch/core_plus_bottleneck_budget120_H10_3000/`
- `results/phase4m_training/antmaze_large_stitch/core_plus_bottleneck_budget120_H10_3000/`
- `results/phase4m/phase4m_replication_summary.csv`
