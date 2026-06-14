# Phase 4I Stronger GCBC Sampling Study

This phase evaluates whether edge-balanced sampling improves offline
supervised GCBC fitting on rare, low-support, bottleneck, and
long-horizon option edges. It is not rollout success.

## Recommendations

- `core_plus_bottleneck_budget192_H5`: `uniform_transition`
  - final MSE: `0.00804584543220695`
  - rare-edge mean MSE: `0.009620023017721635`
  - final MSE ratio vs baseline: `1.0`
  - rare-edge ratio vs baseline: `1.0`

## Baseline Comparisons

| phase2_run | sampling_mode | final_val_action_mse | rare_edge_mean_mse | final_val_action_mse_ratio_vs_baseline | rare_edge_mean_mse_ratio_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| core_plus_bottleneck_budget192_H5 | bottleneck_support_balanced | 0.0107702 | 0.00965429 | 1.3386 | 1.00356 |
| core_plus_bottleneck_budget192_H5 | bottleneck_weighted | 0.00986258 | 0.0093758 | 1.2258 | 0.974613 |
| core_plus_bottleneck_budget192_H5 | support_balanced | 0.0101093 | 0.00908784 | 1.25646 | 0.944679 |
| core_plus_bottleneck_budget192_H5 | uniform_edge | 0.00917783 | 0.00884973 | 1.14069 | 0.919928 |
| core_plus_bottleneck_budget192_H5 | uniform_transition | 0.00804585 | 0.00962002 | 1 | 1 |

## Interpretation Rules

- Lower final validation MSE means better average held-out action fitting.
- Lower rare-edge mean MSE means better fitting on bottleneck, low-support,
  and long-horizon edge groups.
- A sampling mode is useful only if rare-edge gains do not create a large
  overall validation regression.
- These metrics are offline proxies and do not prove option execution.
