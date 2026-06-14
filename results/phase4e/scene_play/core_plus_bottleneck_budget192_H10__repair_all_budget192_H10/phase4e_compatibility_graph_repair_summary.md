# Phase 4E Compatibility Graph Repair Summary

This is reset-free offline graph repair. Repair edges are selected only
from a Phase 2 support-certified edge bank; no kNN/proximity/latent
unsupported edges are introduced.

## Repair Summary

- `num_repair_edges`: `500`
- `num_repair_nodes`: `216`
- `num_mapped_repair_edges`: `500`
- `mean_repair_score`: `3.7114188412639413`
- `median_repair_support`: `26.0`
- `mean_repair_median_h`: `6.149`

## Planner Metrics

### base / support_shortest_path

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.0`
- `mean_pair_incompatible_fraction`: `0.90625`
- `mean_base_path_cost`: `16.543601650734253`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / compat_penalized

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.10205200501253134`
- `mean_pair_incompatible_fraction`: `0.6458333333333333`
- `mean_base_path_cost`: `18.491382278372754`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / compat_threshold

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.18281208385386918`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `29.396984543203533`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### repaired / support_shortest_path

- `path_coverage`: `0.55`
- `mean_min_pair_termination_bridge_coverage`: `0.0`
- `mean_pair_incompatible_fraction`: `0.8875757575757575`
- `mean_base_path_cost`: `19.52684233689789`
- `coverage_delta_vs_base_graph`: `0.39`
- `min_pair_coverage_delta_vs_base_graph`: `0.0`
- `pair_incompatible_delta_vs_base_graph`: `-0.018674242424242538`

### repaired / compat_penalized

- `path_coverage`: `0.55`
- `mean_min_pair_termination_bridge_coverage`: `0.03406069835117137`
- `mean_pair_incompatible_fraction`: `0.6848484848484848`
- `mean_base_path_cost`: `21.55237229050936`
- `coverage_delta_vs_base_graph`: `0.39`
- `min_pair_coverage_delta_vs_base_graph`: `-0.06799130666135997`
- `pair_incompatible_delta_vs_base_graph`: `0.03901515151515156`

### repaired / compat_threshold

- `path_coverage`: `0.51`
- `mean_min_pair_termination_bridge_coverage`: `0.11823189242463651`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `33.354142745350984`
- `coverage_delta_vs_base_graph`: `0.35`
- `min_pair_coverage_delta_vs_base_graph`: `-0.06458019142923267`
- `pair_incompatible_delta_vs_base_graph`: `0.0`

## Interpretation

A positive result is improved compatibility or coverage after adding
only support-certified repair edges. These metrics remain offline
graph evidence and do not prove policy execution success.
