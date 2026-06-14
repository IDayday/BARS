# Phase 4G Direct Repair-Edge Policy Evidence Summary

This is reset-free offline supervised policy evidence. Direct repair
edge MSE evaluates a trained GCBC model on real repair-edge segments;
it is not rollout success.

## Diagnostics

- `num_repair_edges`: `500`
- `num_direct_scored_edges`: `500`
- `note`: `Direct repair policy MSE is offline supervised evidence, not rollout success.`
- `mean_direct_edge_action_mse`: `0.038237718896338785`
- `median_direct_edge_action_mse`: `0.023924724622206253`
- `mean_direct_policy_support_score`: `0.5613987912850971`
- `median_direct_policy_support_score`: `0.6190438053660963`
- `mean_policy_score_delta_direct_minus_transfer`: `0.21998491071931187`
- `spearman_transfer_vs_direct_policy_score`: `0.5467250685239224`
- `transfer_certified_rate`: `0.794`
- `direct_certified_rate`: `0.87`
- `mean_transfer_reliability`: `0.3200390060463293`
- `mean_direct_reliability`: `0.3665760088594466`

## Planner Metrics

### support_shortest_path

- `path_coverage`: `0.51`
- `mean_min_edge_proxy_score`: `0.11847457942373496`
- `mean_uncertified_edge_fraction`: `0.3568627450980392`
- `mean_pair_incompatible_fraction`: `0.9084967320261437`
- `mean_repair_edge_fraction`: `0.30196078431372547`
- `mean_repair_certified_fraction`: `0.9547619047619047`
- `mean_base_path_cost`: `12.242883552647953`

### calibrated_compat_penalized

- `path_coverage`: `0.51`
- `mean_min_edge_proxy_score`: `0.17300133454025365`
- `mean_uncertified_edge_fraction`: `0.1964052287581699`
- `mean_pair_incompatible_fraction`: `0.707516339869281`
- `mean_repair_edge_fraction`: `0.3251633986928104`
- `mean_repair_certified_fraction`: `0.9560185185185185`
- `mean_base_path_cost`: `13.15910838568809`

### compat_threshold

- `path_coverage`: `0.48`
- `mean_min_edge_proxy_score`: `0.14841365797825998`
- `mean_uncertified_edge_fraction`: `0.16448412698412698`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.2517729377104377`
- `mean_repair_certified_fraction`: `0.9768518518518517`
- `mean_base_path_cost`: `23.53210792535327`

### calibrated_compat_threshold

- `path_coverage`: `0.48`
- `mean_min_edge_proxy_score`: `0.24361123623951766`
- `mean_uncertified_edge_fraction`: `0.03733540764790765`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.40100484006734005`
- `mean_repair_certified_fraction`: `0.973170731707317`
- `mean_base_path_cost`: `25.71148769458902`

## Interpretation

Phase 4G replaces the repair-edge policy component with direct GCBC
action-fitting evidence where possible. The result is stronger than
Phase 4F's transfer-only proxy, but still does not prove closed-loop
edge execution.
