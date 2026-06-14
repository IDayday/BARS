# Phase 4E Compatibility Graph Repair Summary

This is reset-free offline graph repair. Repair edges are selected only
from a Phase 2 support-certified edge bank; no kNN/proximity/latent
unsupported edges are introduced.

## Repair Summary

- `num_repair_edges`: `500`
- `num_repair_nodes`: `199`
- `num_mapped_repair_edges`: `500`
- `mean_repair_score`: `3.254986435406438`
- `median_repair_support`: `24.0`
- `mean_repair_median_h`: `3.473`

## Planner Metrics

### base / support_shortest_path

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.0`
- `mean_pair_incompatible_fraction`: `0.90625`
- `mean_base_path_cost`: `10.7958664251808`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / compat_penalized

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.03718433746567868`
- `mean_pair_incompatible_fraction`: `0.640625`
- `mean_base_path_cost`: `11.96457327595142`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / calibrated_compat_penalized

- `path_coverage`: `0.16`
- `mean_min_pair_termination_bridge_coverage`: `0.03627854036422941`
- `mean_pair_incompatible_fraction`: `0.7479166666666667`
- `mean_base_path_cost`: `11.458585086678584`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / compat_threshold

- `path_coverage`: `0.15`
- `mean_min_pair_termination_bridge_coverage`: `0.11568243745817454`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `20.670992427403952`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / calibrated_compat_threshold

- `path_coverage`: `0.15`
- `mean_min_pair_termination_bridge_coverage`: `0.08726954709174264`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `22.266591448047166`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### repaired / support_shortest_path

- `path_coverage`: `0.51`
- `mean_min_pair_termination_bridge_coverage`: `0.007352941176470588`
- `mean_pair_incompatible_fraction`: `0.9084967320261437`
- `mean_base_path_cost`: `12.242883552647953`
- `coverage_delta_vs_base_graph`: `0.35`
- `min_pair_coverage_delta_vs_base_graph`: `0.007352941176470588`
- `pair_incompatible_delta_vs_base_graph`: `0.002246732026143672`

### repaired / compat_penalized

- `path_coverage`: `0.51`
- `mean_min_pair_termination_bridge_coverage`: `0.023153097751165586`
- `mean_pair_incompatible_fraction`: `0.6830065359477123`
- `mean_base_path_cost`: `13.247242126290224`
- `coverage_delta_vs_base_graph`: `0.35`
- `min_pair_coverage_delta_vs_base_graph`: `-0.014031239714513095`
- `pair_incompatible_delta_vs_base_graph`: `0.04238153594771232`

### repaired / calibrated_compat_penalized

- `path_coverage`: `0.51`
- `mean_min_pair_termination_bridge_coverage`: `0.01979432744866127`
- `mean_pair_incompatible_fraction`: `0.7261437908496732`
- `mean_base_path_cost`: `13.292955050694793`
- `coverage_delta_vs_base_graph`: `0.35`
- `min_pair_coverage_delta_vs_base_graph`: `-0.01648421291556814`
- `pair_incompatible_delta_vs_base_graph`: `-0.021772875816993498`

### repaired / compat_threshold

- `path_coverage`: `0.48`
- `mean_min_pair_termination_bridge_coverage`: `0.10688152295465754`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `23.53210792535327`
- `coverage_delta_vs_base_graph`: `0.32999999999999996`
- `min_pair_coverage_delta_vs_base_graph`: `-0.008800914503516999`
- `pair_incompatible_delta_vs_base_graph`: `0.0`

### repaired / calibrated_compat_threshold

- `path_coverage`: `0.48`
- `mean_min_pair_termination_bridge_coverage`: `0.09592800865218137`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `25.310020847107396`
- `coverage_delta_vs_base_graph`: `0.32999999999999996`
- `min_pair_coverage_delta_vs_base_graph`: `0.00865846156043873`
- `pair_incompatible_delta_vs_base_graph`: `0.0`

## Interpretation

A positive result is improved compatibility or coverage after adding
only support-certified repair edges. These metrics remain offline
graph evidence and do not prove policy execution success.
