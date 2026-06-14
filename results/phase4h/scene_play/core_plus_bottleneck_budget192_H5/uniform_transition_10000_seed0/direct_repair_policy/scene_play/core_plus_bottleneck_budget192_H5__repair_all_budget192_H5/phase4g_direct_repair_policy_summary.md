# Phase 4G Direct Repair-Edge Policy Evidence Summary

This is reset-free offline supervised policy evidence. Direct repair
edge MSE evaluates a trained GCBC model on real repair-edge segments;
it is not rollout success.

## Diagnostics

- `num_repair_edges`: `500`
- `num_direct_scored_edges`: `500`
- `note`: `Direct repair policy MSE is offline supervised evidence, not rollout success.`
- `mean_direct_edge_action_mse`: `0.011463576496073543`
- `median_direct_edge_action_mse`: `0.005752590353949927`
- `mean_direct_policy_support_score`: `0.8332988952585879`
- `median_direct_policy_support_score`: `0.8911249055165809`
- `mean_policy_score_delta_direct_minus_transfer`: `0.4918850146928027`
- `spearman_transfer_vs_direct_policy_score`: `0.49820359303897216`
- `transfer_certified_rate`: `0.794`
- `direct_certified_rate`: `0.894`
- `mean_transfer_reliability`: `0.3200390060463293`
- `mean_direct_reliability`: `0.40188976399671994`

## Planner Metrics

### support_shortest_path

- `path_coverage`: `0.51`
- `mean_min_edge_proxy_score`: `0.1269506396619673`
- `mean_uncertified_edge_fraction`: `0.34542483660130713`
- `mean_pair_incompatible_fraction`: `0.9084967320261437`
- `mean_repair_edge_fraction`: `0.30196078431372547`
- `mean_repair_certified_fraction`: `0.9904761904761904`
- `mean_base_path_cost`: `12.242883552647953`

### calibrated_compat_penalized

- `path_coverage`: `0.51`
- `mean_min_edge_proxy_score`: `0.18206866864991042`
- `mean_uncertified_edge_fraction`: `0.1849673202614379`
- `mean_pair_incompatible_fraction`: `0.7009803921568627`
- `mean_repair_edge_fraction`: `0.3349673202614379`
- `mean_repair_certified_fraction`: `0.990990990990991`
- `mean_base_path_cost`: `13.158273660618795`

### compat_threshold

- `path_coverage`: `0.48`
- `mean_min_edge_proxy_score`: `0.14935070662291722`
- `mean_uncertified_edge_fraction`: `0.159796626984127`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.2517729377104377`
- `mean_repair_certified_fraction`: `1.0`
- `mean_base_path_cost`: `23.53210792535327`

### calibrated_compat_threshold

- `path_coverage`: `0.48`
- `mean_min_edge_proxy_score`: `0.24757374342760316`
- `mean_uncertified_edge_fraction`: `0.033168740981240986`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.42290839947089953`
- `mean_repair_certified_fraction`: `1.0`
- `mean_base_path_cost`: `25.832040154798204`

## Interpretation

Phase 4G replaces the repair-edge policy component with direct GCBC
action-fitting evidence where possible. The result is stronger than
Phase 4F's transfer-only proxy, but still does not prove closed-loop
edge execution.
