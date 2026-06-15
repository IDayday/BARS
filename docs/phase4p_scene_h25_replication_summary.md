# Phase 4P Scene H25 Replication and Planner Scaling

Phase 4P extends the repaired-graph and planner-relevant loss-weighting result
to Scene H25. It also makes the compatibility planner practical at the H25
scale by reusing line-graph indices, edge costs, and pair-coverage lookups
across query evaluations.

No environment rollout is used. These are reset-free graph metrics and offline
supervised GCBC proxy metrics.

## Graph Repair

Scene H25 uses:

- Base graph: `results/phase2/scene_play/core_plus_bottleneck_budget192_H25`
- Repair bank: `results/phase2/scene_play/all_budget192_H25`
- Repair budget: `500` support-certified bank edges

The repaired graph adds 500 support-certified edges covering 186 nodes. The
strict compatibility planner improves substantially:

| graph | method | coverage | mean min pair bridge coverage | incompatible fraction |
| --- | --- | ---: | ---: | ---: |
| base | `compat_threshold` | 0.17 | 0.141649 | 0.000000 |
| repaired | `compat_threshold` | 0.64 | 0.116823 | 0.000000 |

Support shortest-path coverage also rises from `0.17` to `0.65`, but it still
has high pair incompatibility (`0.792` on repaired paths). This repeats the
Scene H5/H10 pattern: support-bank repair recovers coverage, while
compatibility-aware planning is still needed to avoid non-composable paths.

## Planner-Relevant GCBC

Compared with the same augmented graph trained with ordinary clipped
support+bottleneck weights:

| method | final val MSE ratio | direct repair MSE ratio | planner-used repair MSE ratio | policy support ratio |
| --- | ---: | ---: | ---: | ---: |
| `planner_relevant_repair_s02` | 0.994574 | 0.992829 | 1.009024 | 1.001577 |
| `planner_relevant_repair_s04` | 0.992440 | 0.983579 | 0.991416 | 1.004670 |

`s04` is the useful H25 setting. It improves overall validation MSE, direct
repair-edge MSE, and policy-support score, and it slightly improves
planner-used repair-edge MSE. `s02` improves broad repair-edge metrics but
worsens planner-used repair-edge MSE, so it should not be the H25 default.

The Phase 4O selector chooses H25 `s04` as a `relaxed_guard_pass`: it misses the
strict planner-used threshold by a small margin (`0.991416` versus strict
threshold `0.990000`), but all supervised proxy metrics are no worse than the
baseline.

## Interpretation

This strengthens the repeated Scene pattern:

- graph repair is a real structural fix for compressed graph coverage;
- compatibility constraints remain necessary after repair;
- planner-relevant loss weighting is useful, but the best strength is
  dataset/horizon dependent;
- selector guards are needed to avoid manual cherry-picking.

This is not online execution evidence. Closed-loop success still requires env
availability and rollout evaluation.

Artifacts:

- `configs/phase4e_compatibility_graph_repair_scene_H25.yaml`
- `configs/phase4m_planner_relevant_loss_weighting_scene_H25_B192_3000.yaml`
- `results/phase4e/scene_play/core_plus_bottleneck_budget192_H25__repair_all_budget192_H25/`
- `results/phase4m/scene_play/core_plus_bottleneck_budget192_H25_3000/`
- `results/phase4m_training/scene_play/core_plus_bottleneck_budget192_H25_3000/`
