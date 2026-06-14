# Phase 4M Planner-Relevant Repair Loss Weighting Summary

Phase 4M tests a small planner-aware supervised loss-weighting change on Scene
H5 `core_plus_bottleneck_budget192_H5`. It uses the Phase 4E repaired support
graph and does not run an environment rollout.

## Result

Compared with the same augmented graph trained using ordinary clipped
support+bottleneck weights:

| metric | augmented support+bottleneck | planner-relevant repair | ratio |
| --- | ---: | ---: | ---: |
| final validation MSE | 0.008423 | 0.008247 | 0.979 |
| direct repair-edge MSE | 0.011174 | 0.010834 | 0.970 |
| planner-used repair-edge MSE | 0.011188 | 0.010975 | 0.981 |
| not-planner-used repair-edge MSE | 0.011172 | 0.010819 | 0.968 |
| direct repair policy support score | 0.816947 | 0.821480 | 1.006 |

The planner-relevant weight table assigns the highest mean weight to the 48
planner-used repair edges (`1.217` mean loss weight), while preserving clipped
weights with max `2.096`.

## Analysis

This is a positive small-scale result. Adding planner relevance did not damage
overall validation MSE; it improved ordinary validation MSE, direct repair-edge
MSE, and the planner-used repair-edge group.

The result should be treated as an offline supervised proxy. It does not prove
closed-loop option execution, and it is only Scene H5 with two seeds. The next
useful step is replication across AntMaze and Scene H10/H25 or longer training,
not a stronger claim from this single run.

Artifacts:

- `results/phase4m/scene_play/core_plus_bottleneck_budget192_H5_3000/`
- `results/phase4m_training/scene_play/core_plus_bottleneck_budget192_H5_3000/`
