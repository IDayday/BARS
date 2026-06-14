# Phase 4G Direct Repair-Edge Policy Evidence Summary

This is reset-free offline supervised policy evidence. Direct repair
edge MSE evaluates a trained GCBC model on real repair-edge segments;
it is not rollout success.

## Diagnostics

- `num_repair_edges`: `500`
- `num_direct_scored_edges`: `500`
- `note`: `Direct repair policy MSE is offline supervised evidence, not rollout success.`
- `mean_direct_edge_action_mse`: `0.01523549872890449`
- `median_direct_edge_action_mse`: `0.008105255997470404`
- `mean_direct_policy_support_score`: `0.7808818150058316`
- `median_direct_policy_support_score`: `0.8503518125496998`
- `mean_policy_score_delta_direct_minus_transfer`: `0.43946793444004645`
- `spearman_transfer_vs_direct_policy_score`: `0.5450019188643471`
- `transfer_certified_rate`: `0.794`
- `direct_certified_rate`: `0.89`
- `mean_transfer_reliability`: `0.3200390060463293`
- `mean_direct_reliability`: `0.3959780428806425`

## Planner Metrics

### support_shortest_path

- `path_coverage`: `0.51`
- `mean_min_edge_proxy_score`: `0.12603402263451396`
- `mean_uncertified_edge_fraction`: `0.35196078431372546`
- `mean_pair_incompatible_fraction`: `0.9084967320261437`
- `mean_repair_edge_fraction`: `0.30196078431372547`
- `mean_repair_certified_fraction`: `0.9619047619047618`
- `mean_base_path_cost`: `12.242883552647953`

### calibrated_compat_penalized

- `path_coverage`: `0.51`
- `mean_min_edge_proxy_score`: `0.18098743648843765`
- `mean_uncertified_edge_fraction`: `0.1915032679738562`
- `mean_pair_incompatible_fraction`: `0.7009803921568627`
- `mean_repair_edge_fraction`: `0.3349673202614379`
- `mean_repair_certified_fraction`: `0.9639639639639639`
- `mean_base_path_cost`: `13.158273660618795`

### compat_threshold

- `path_coverage`: `0.48`
- `mean_min_edge_proxy_score`: `0.1492548976060325`
- `mean_uncertified_edge_fraction`: `0.16187996031746033`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.2517729377104377`
- `mean_repair_certified_fraction`: `0.9907407407407407`
- `mean_base_path_cost`: `23.53210792535327`

### calibrated_compat_threshold

- `path_coverage`: `0.48`
- `mean_min_edge_proxy_score`: `0.2470576639282307`
- `mean_uncertified_edge_fraction`: `0.03525207431457431`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.42290839947089953`
- `mean_repair_certified_fraction`: `0.975609756097561`
- `mean_base_path_cost`: `25.832040154798204`

## Interpretation

Phase 4G replaces the repair-edge policy component with direct GCBC
action-fitting evidence where possible. The result is stronger than
Phase 4F's transfer-only proxy, but still does not prove closed-loop
edge execution.
