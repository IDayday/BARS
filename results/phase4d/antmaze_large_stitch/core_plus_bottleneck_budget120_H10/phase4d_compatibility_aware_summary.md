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

- `path_coverage`: `0.566`
- `mean_min_pair_termination_bridge_coverage`: `0.04297249099177845`
- `mean_pair_incompatible_fraction`: `0.16082443653618028`
- `mean_min_edge_proxy_score`: `0.10492855452564266`
- `mean_original_uncertified_edge_fraction`: `0.3270803270803271`
- `mean_base_path_cost`: `50.55063453182929`

### calibrated_edge_penalized

- `path_coverage`: `0.566`
- `mean_min_pair_termination_bridge_coverage`: `0.09817933513523208`
- `mean_pair_incompatible_fraction`: `0.04985518633205466`
- `mean_min_edge_proxy_score`: `0.25461091126600705`
- `mean_original_uncertified_edge_fraction`: `0.036641618822469885`
- `mean_base_path_cost`: `55.47807524429424`

### compat_penalized

- `path_coverage`: `0.566`
- `mean_min_pair_termination_bridge_coverage`: `0.07019721595507995`
- `mean_pair_incompatible_fraction`: `0.10465907165195422`
- `mean_min_edge_proxy_score`: `0.11685073402547824`
- `mean_original_uncertified_edge_fraction`: `0.26765988148966874`
- `mean_base_path_cost`: `52.0222466818775`

### calibrated_compat_penalized

- `path_coverage`: `0.566`
- `mean_min_pair_termination_bridge_coverage`: `0.11307781049591503`
- `mean_pair_incompatible_fraction`: `0.03318543852351682`
- `mean_min_edge_proxy_score`: `0.24710409110849757`
- `mean_original_uncertified_edge_fraction`: `0.04790427263831519`
- `mean_base_path_cost`: `55.51073441091786`

### compat_threshold

- `path_coverage`: `0.544`
- `mean_min_pair_termination_bridge_coverage`: `0.11465729225448132`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_min_edge_proxy_score`: `0.14783579718341272`
- `mean_original_uncertified_edge_fraction`: `0.21771217712177118`
- `mean_base_path_cost`: `52.79297678323901`

### calibrated_compat_threshold

- `path_coverage`: `0.544`
- `mean_min_pair_termination_bridge_coverage`: `0.14568066310451566`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_min_edge_proxy_score`: `0.2607927936231217`
- `mean_original_uncertified_edge_fraction`: `0.04316318163181632`
- `mean_base_path_cost`: `56.023667894744385`

## Interpretation

Phase 4D evaluates whether path selection improves when option
composition is treated as a transition-dependent cost. A better
offline planner should preserve useful coverage while raising
path-level adjacent-edge bridge coverage and lowering incompatible
pair exposure. These are graph-layer proxies, not rollout labels.
