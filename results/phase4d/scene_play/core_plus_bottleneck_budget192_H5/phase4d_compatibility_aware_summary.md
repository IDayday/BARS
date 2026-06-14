# Phase 4D Compatibility-Aware Planning Summary

This is reset-free offline planning. It does not use environment rollout
and does not claim option execution success.

## Methods

- `support_shortest_path`: Phase 2 support graph shortest path.
- `calibrated_edge_penalized`: Phase 4C calibrated single-edge risk cost.
- `compat_penalized`: support edges with adjacent-edge bridge penalty.
- `calibrated_compat_penalized`: calibrated edge risk plus bridge penalty.
- `compat_threshold`: rejects adjacent edge transitions below the bridge floor.
- `calibrated_compat_threshold`: calibrated edge risk with bridge floor.

## Key Metrics

### support_shortest_path

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.0`
- `mean_pair_incompatible_fraction`: `0.90625`
- `mean_min_edge_proxy_score`: `0.1106689308214164`
- `mean_original_uncertified_edge_fraction`: `0.5718749999999999`
- `mean_base_path_cost`: `10.7958664251808`

### calibrated_edge_penalized

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.005655630665232754`
- `mean_pair_incompatible_fraction`: `0.7583333333333333`
- `mean_min_edge_proxy_score`: `0.16316155592921325`
- `mean_original_uncertified_edge_fraction`: `0.33229166666666665`
- `mean_base_path_cost`: `11.424314372197585`

### compat_penalized

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.03718433746567868`
- `mean_pair_incompatible_fraction`: `0.640625`
- `mean_min_edge_proxy_score`: `0.10021717384496724`
- `mean_original_uncertified_edge_fraction`: `0.5958333333333333`
- `mean_base_path_cost`: `11.96457327595142`

### calibrated_compat_penalized

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.03627854036422941`
- `mean_pair_incompatible_fraction`: `0.7479166666666667`
- `mean_min_edge_proxy_score`: `0.15468461540933037`
- `mean_original_uncertified_edge_fraction`: `0.36354166666666665`
- `mean_base_path_cost`: `11.458585086678584`

### compat_threshold

- `path_coverage`: `0.15`
- `mean_min_pair_termination_bridge_coverage`: `0.11568243745817454`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_min_edge_proxy_score`: `0.1747236610724442`
- `mean_original_uncertified_edge_fraction`: `0.26063492063492066`
- `mean_base_path_cost`: `20.670992427403952`

### calibrated_compat_threshold

- `path_coverage`: `0.15`
- `mean_min_pair_termination_bridge_coverage`: `0.08726954709174264`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_min_edge_proxy_score`: `0.2727783388910937`
- `mean_original_uncertified_edge_fraction`: `0.06277777777777778`
- `mean_base_path_cost`: `22.266591448047166`

## Interpretation

Phase 4D evaluates whether path selection improves when option
composition is treated as a transition-dependent cost. A better
offline planner should preserve useful coverage while raising
path-level adjacent-edge bridge coverage and lowering incompatible
pair exposure. These are graph-layer proxies, not rollout labels.
