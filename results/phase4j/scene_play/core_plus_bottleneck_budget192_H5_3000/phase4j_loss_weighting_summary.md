# Phase 4J Mixed/Loss-Weighted GCBC Training Study

This phase keeps transition sampling broad and applies controlled
per-edge loss weights to rare, low-support, or bottleneck edges. It is
offline supervised action fitting only, not rollout success.

## Recommendations

- `core_plus_bottleneck_budget192_H5`: `loss_support_bottleneck_s03`
  - final MSE: `0.008177034353138879`
  - rare-edge mean MSE: `0.009038379811559905`
  - final MSE ratio vs baseline: `1.0163051753898713`
  - rare-edge ratio vs baseline: `0.9395382729240617`

## Baseline Comparisons

| method | final_val_action_mse | rare_edge_mean_mse | final_val_action_mse_ratio_vs_baseline | rare_edge_mean_mse_ratio_vs_baseline |
| --- | --- | --- | --- | --- |
| loss_bottleneck_s03 | 0.0083225 | 0.00919963 | 1.03438 | 0.9563 |
| loss_support_bottleneck_s03 | 0.00817703 | 0.00903838 | 1.01631 | 0.939538 |
| loss_support_s03 | 0.00823283 | 0.00931497 | 1.02324 | 0.96829 |
| uniform_transition_none | 0.00804585 | 0.00962002 | 1 | 1 |

## Interpretation Rules

- A useful loss-weighted method should improve rare-edge mean MSE without
  a large final validation MSE regression.
- This study does not change graph provenance: all examples still come from
  Phase 2 support-certified edge segments.
- These metrics are offline proxies and do not prove option execution.
