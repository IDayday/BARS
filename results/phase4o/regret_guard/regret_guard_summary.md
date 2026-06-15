# Phase 4O Planner-Relevance Regret-Guard Selection

This phase turns the Phase 4N manual Scene H10 choice into a reusable
offline supervised model-selection guard. It does not train a new policy,
run an environment rollout, or add unsupported graph edges.

## Guard

- `max_final_val_ratio`: `1.01`
- `max_direct_repair_ratio`: `1.0`
- `max_planner_used_ratio`: `0.99`
- `min_policy_support_ratio`: `1.0`
- `allow_relaxed_improvement_fallback`: `True`
- `relaxed_max_direct_repair_ratio`: `1.0`
- `relaxed_max_planner_used_ratio`: `1.0`
- `relaxed_min_policy_support_ratio`: `1.0`

## Aggregate

- `num_runs`: `4`
- `num_runs_with_guard_pass`: `3`
- `num_runs_with_relaxed_guard_pass`: `1`
- `num_runs_fallback_baseline`: `0`
- `mean_selected_final_val_ratio`: `0.9930981210288793`
- `mean_selected_direct_repair_ratio`: `0.9841552449214142`
- `mean_selected_planner_used_repair_ratio`: `0.9798063785880092`

## Selected Runs

| dataset_key | run_name | selected_method | selection_status | final_val_action_mse_ratio_vs_baseline | direct_repair_edge_mse_ratio_vs_baseline | planner_used_repair_edge_mse_ratio_vs_baseline | direct_repair_policy_support_score_ratio_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze_large_stitch | core_plus_bottleneck_budget120_H10_3000 | planner_relevant_repair_s04 | guard_pass | 0.995124 | 0.995081 | 0.986447 | 1.00376 |
| scene_play | core_plus_bottleneck_budget192_H10_3000 | planner_relevant_repair_s02 | guard_pass | 1.00579 | 0.988326 | 0.960478 | 1.00201 |
| scene_play | core_plus_bottleneck_budget192_H25_3000 | planner_relevant_repair_s04 | relaxed_guard_pass | 0.99244 | 0.983579 | 0.991416 | 1.00467 |
| scene_play | core_plus_bottleneck_budget192_H5_3000 | planner_relevant_repair_s04 | guard_pass | 0.979038 | 0.969635 | 0.980885 | 1.00555 |

## Interpretation

- A non-baseline method is recommended only when every guard passes.
- If strict guards fail but the relaxed fallback is enabled, the selector
  can choose a candidate that improves direct repair MSE, planner-used
  repair MSE, and policy-support score without exceeding final-val regret.
- If neither strict nor relaxed guards pass, the selector falls back to
  the same augmented-graph support+bottleneck baseline.
- The selected method is an offline supervised candidate for further
  validation. It is not an execution-success claim.
