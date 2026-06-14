# Phase 4F Repair-Edge Certification Summary

This is reset-free offline certification. Repair-edge scores are
conservative transfer proxies from support scale, endpoint-neighbor
policy fitting, behavior support, and compatibility context. They are
not rollout success probabilities.

## Diagnostics

- `num_edges`: `500`
- `note`: `Repair-edge certification is a conservative offline transfer proxy, not rollout success.`
- `edge_proxy_score_original_mean`: `nan`
- `edge_proxy_score_original_median`: `nan`
- `calibrated_edge_reliability_score_mean`: `0.3200390060463293`
- `calibrated_edge_reliability_score_median`: `0.2767012479074247`
- `calibrated_certified_edges`: `397`
- `calibrated_certified_rate`: `0.794`
- `original_certified_edges`: `0`
- `original_certified_rate`: `0.0`
- `heldout_label_available_for_repair_edges`: `False`
- `num_base_certification_edges`: `1897`
- `num_repair_certification_edges`: `500`

## Planner Metrics

### support_shortest_path

- `path_coverage`: `0.51`
- `mean_min_edge_proxy_score`: `0.1117295215372113`
- `mean_uncertified_edge_fraction`: `0.37973856209150325`
- `mean_original_uncertified_edge_fraction`: `0.37973856209150325`
- `mean_pair_incompatible_fraction`: `0.9084967320261437`
- `mean_repair_edge_fraction`: `0.30196078431372547`
- `mean_repair_certified_fraction`: `0.8619047619047618`
- `mean_base_path_cost`: `12.242883552647953`

### calibrated_compat_penalized

- `path_coverage`: `0.51`
- `mean_min_edge_proxy_score`: `0.16016619645720848`
- `mean_uncertified_edge_fraction`: `0.22745098039215686`
- `mean_original_uncertified_edge_fraction`: `0.22745098039215686`
- `mean_pair_incompatible_fraction`: `0.7124183006535947`
- `mean_repair_edge_fraction`: `0.32810457516339864`
- `mean_repair_certified_fraction`: `0.8935185185185185`
- `mean_base_path_cost`: `13.074823902380208`

### compat_threshold

- `path_coverage`: `0.48`
- `mean_min_edge_proxy_score`: `0.14326612716266623`
- `mean_uncertified_edge_fraction`: `0.1726438492063492`
- `mean_original_uncertified_edge_fraction`: `0.1726438492063492`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.2517729377104377`
- `mean_repair_certified_fraction`: `0.9305555555555556`
- `mean_base_path_cost`: `23.53210792535327`

### calibrated_compat_threshold

- `path_coverage`: `0.48`
- `mean_min_edge_proxy_score`: `0.23031922004028707`
- `mean_uncertified_edge_fraction`: `0.04372745310245311`
- `mean_original_uncertified_edge_fraction`: `0.04372745310245311`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_repair_edge_fraction`: `0.40100484006734005`
- `mean_repair_certified_fraction`: `0.9357723577235773`
- `mean_base_path_cost`: `25.71148769458902`

## Interpretation

Phase 4F should be read as a planner-facing repair-edge risk estimate.
It narrows the Phase 4E gap where repair edges improved coverage but
were treated as uncertified defaults. It still does not replace
GCBC edge rollout validation.
