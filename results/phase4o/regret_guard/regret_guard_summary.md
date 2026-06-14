# Phase 4O Planner-Relevance Regret-Guard Selection

This phase turns the Phase 4N manual Scene H10 choice into a reusable
offline supervised model-selection guard. It does not train a new policy,
run an environment rollout, or add unsupported graph edges.

## Guard

- `max_final_val_ratio`: `1.01`
- `max_direct_repair_ratio`: `1.0`
- `max_planner_used_ratio`: `0.99`
- `min_policy_support_ratio`: `1.0`

## Aggregate

- `num_runs`: `3`
- `num_runs_with_guard_pass`: `3`
- `num_runs_fallback_baseline`: `0`
- `mean_selected_final_val_ratio`: `0.9933174106399508`
- `mean_selected_direct_repair_ratio`: `0.9843471938413818`
- `mean_selected_planner_used_repair_ratio`: `0.9759366257134178`

## Selected Runs

| dataset_key | run_name | selected_method | selection_status | final_val_action_mse_ratio_vs_baseline | direct_repair_edge_mse_ratio_vs_baseline | planner_used_repair_edge_mse_ratio_vs_baseline | direct_repair_policy_support_score_ratio_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze_large_stitch | core_plus_bottleneck_budget120_H10_3000 | planner_relevant_repair_s04 | guard_pass | 0.995124 | 0.995081 | 0.986447 | 1.00376 |
| scene_play | core_plus_bottleneck_budget192_H10_3000 | planner_relevant_repair_s02 | guard_pass | 1.00579 | 0.988326 | 0.960478 | 1.00201 |
| scene_play | core_plus_bottleneck_budget192_H5_3000 | planner_relevant_repair_s04 | guard_pass | 0.979038 | 0.969635 | 0.980885 | 1.00555 |

## Interpretation

- A non-baseline method is recommended only when every guard passes.
- If no planner-relevant candidate passes, the selector falls back to
  the same augmented-graph support+bottleneck baseline.
- The selected method is an offline supervised candidate for further
  validation. It is not an execution-success claim.
