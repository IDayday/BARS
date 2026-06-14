# Phase 4M Planner-Relevant Weighting Replication Summary

This aggregates reset-free offline supervised Phase 4M runs. It does
not report environment rollout success.

## Aggregate

- `num_runs`: `2`
- `datasets`: `['antmaze-large-stitch-v0', 'scene-play-v0']`
- `mean_final_val_mse_ratio`: `0.987081142066941`
- `mean_direct_repair_mse_ratio`: `0.9823575835256515`
- `mean_planner_used_repair_mse_ratio`: `0.9836659095617538`
- `num_runs_planner_used_improved`: `2`
- `num_runs_direct_repair_improved`: `2`
- `num_runs_final_val_improved`: `2`
- `note`: `Offline supervised replication summary; not rollout success.`

## Runs

| dataset | phase2_run | final_val_action_mse_ratio_vs_baseline | direct_repair_edge_mse_ratio_vs_baseline | planner_used_repair_edge_mse_ratio_vs_baseline | direct_repair_policy_support_score_ratio_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| antmaze-large-stitch-v0 | core_plus_bottleneck_budget120_H10 | 0.995124 | 0.995081 | 0.986447 | 1.00376 |
| scene-play-v0 | core_plus_bottleneck_budget192_H5 | 0.979038 | 0.969635 | 0.980885 | 1.00555 |

## Interpretation

A ratio below 1.0 is better for MSE metrics. A ratio above 1.0 is
better for direct repair policy support score. These are supervised
offline proxies and must not be interpreted as closed-loop execution.
