# Phase 4G Direct Repair-Edge Policy Evidence Summary

This is reset-free offline supervised policy evidence. Direct repair
edge MSE evaluates a trained GCBC model on real repair-edge segments;
it is not rollout success.

## Diagnostics

- `num_repair_edges`: `200`
- `num_direct_scored_edges`: `200`
- `note`: `Direct repair policy MSE is offline supervised evidence, not rollout success.`
- `mean_direct_edge_action_mse`: `0.05540404832623096`
- `median_direct_edge_action_mse`: `0.05210425721146028`
- `mean_direct_policy_support_score`: `0.37183183645947393`
- `median_direct_policy_support_score`: `0.352718468083053`
- `mean_policy_score_delta_direct_minus_transfer`: `-0.13437510287322763`
- `spearman_transfer_vs_direct_policy_score`: `0.054745291421243805`
- `transfer_certified_rate`: `0.91`
- `direct_certified_rate`: `0.905`
- `mean_transfer_reliability`: `0.4081701379913887`
- `mean_direct_reliability`: `0.3735233606176564`

## Planner Metrics

### support_shortest_path

- `path_coverage`: `0.642`
- `mean_min_edge_proxy_score`: `0.10513908413942245`
- `mean_uncertified_edge_fraction`: `0.265438198953824`
- `mean_pair_incompatible_fraction`: `0.1565121659949246`
- `mean_repair_edge_fraction`: `0.04091751965583741`
- `mean_repair_certified_fraction`: `0.8522012578616351`
- `mean_base_path_cost`: `50.962350413397495`

### calibrated_compat_penalized

- `path_coverage`: `0.642`
- `mean_min_edge_proxy_score`: `0.25819322164506603`
- `mean_uncertified_edge_fraction`: `0.023218005952380952`
- `mean_pair_incompatible_fraction`: `0.026814268977278385`
- `mean_repair_edge_fraction`: `0.1286641447856401`
- `mean_repair_certified_fraction`: `0.9825581395348837`
- `mean_base_path_cost`: `56.07353992621472`

### compat_threshold

- `path_coverage`: `0.62`
- `mean_min_edge_proxy_score`: `0.1458208914002856`
- `mean_uncertified_edge_fraction`: `0.15560050341603737`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.05918714797747055`
- `mean_repair_certified_fraction`: `0.926923076923077`
- `mean_base_path_cost`: `53.36083744676137`

### calibrated_compat_threshold

- `path_coverage`: `0.62`
- `mean_min_edge_proxy_score`: `0.2761069532567158`
- `mean_uncertified_edge_fraction`: `0.013553706272152875`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.15578178094307127`
- `mean_repair_certified_fraction`: `0.9832089552238806`
- `mean_base_path_cost`: `56.82856212893626`

## Interpretation

Phase 4G replaces the repair-edge policy component with direct GCBC
action-fitting evidence where possible. The result is stronger than
Phase 4F's transfer-only proxy, but still does not prove closed-loop
edge execution.
