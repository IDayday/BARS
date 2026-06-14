# Phase 4F Repair-Edge Certification Summary

This is reset-free offline certification. Repair-edge scores are
conservative transfer proxies from support scale, endpoint-neighbor
policy fitting, behavior support, and compatibility context. They are
not rollout success probabilities.

## Diagnostics

- `num_edges`: `200`
- `note`: `Repair-edge certification is a conservative offline transfer proxy, not rollout success.`
- `edge_proxy_score_original_mean`: `nan`
- `edge_proxy_score_original_median`: `nan`
- `calibrated_edge_reliability_score_mean`: `0.4081701379913887`
- `calibrated_edge_reliability_score_median`: `0.4278469094063826`
- `calibrated_certified_edges`: `182`
- `calibrated_certified_rate`: `0.91`
- `original_certified_edges`: `0`
- `original_certified_rate`: `0.0`
- `heldout_label_available_for_repair_edges`: `False`
- `num_base_certification_edges`: `582`
- `num_repair_certification_edges`: `200`

## Planner Metrics

### support_shortest_path

- `path_coverage`: `0.642`
- `mean_min_edge_proxy_score`: `0.10527714700671889`
- `mean_uncertified_edge_fraction`: `0.265438198953824`
- `mean_original_uncertified_edge_fraction`: `0.265438198953824`
- `mean_pair_incompatible_fraction`: `0.1565121659949246`
- `mean_repair_edge_fraction`: `0.04091751965583741`
- `mean_repair_certified_fraction`: `0.8522012578616351`
- `mean_base_path_cost`: `50.962350413397495`

### calibrated_compat_penalized

- `path_coverage`: `0.642`
- `mean_min_edge_proxy_score`: `0.25927703114410494`
- `mean_uncertified_edge_fraction`: `0.023218005952380952`
- `mean_original_uncertified_edge_fraction`: `0.023218005952380952`
- `mean_pair_incompatible_fraction`: `0.02636644094324345`
- `mean_repair_edge_fraction`: `0.1312169312169312`
- `mean_repair_certified_fraction`: `0.9826923076923076`
- `mean_base_path_cost`: `56.120454967809025`

### compat_threshold

- `path_coverage`: `0.62`
- `mean_min_edge_proxy_score`: `0.14576471325797338`
- `mean_uncertified_edge_fraction`: `0.15560050341603737`
- `mean_original_uncertified_edge_fraction`: `0.15560050341603737`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.05918714797747055`
- `mean_repair_certified_fraction`: `0.926923076923077`
- `mean_base_path_cost`: `53.36083744676137`

### calibrated_compat_threshold

- `path_coverage`: `0.62`
- `mean_min_edge_proxy_score`: `0.27675161119821734`
- `mean_uncertified_edge_fraction`: `0.013553706272152875`
- `mean_original_uncertified_edge_fraction`: `0.013553706272152875`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.16055648652422846`
- `mean_repair_certified_fraction`: `0.9836956521739131`
- `mean_base_path_cost`: `56.91714115466668`

## Interpretation

Phase 4F should be read as a planner-facing repair-edge risk estimate.
It narrows the Phase 4E gap where repair edges improved coverage but
were treated as uncertified defaults. It still does not replace
GCBC edge rollout validation.
