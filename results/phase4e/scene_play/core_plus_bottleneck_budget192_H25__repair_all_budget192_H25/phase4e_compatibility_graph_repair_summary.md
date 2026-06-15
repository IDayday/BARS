# Phase 4E Compatibility Graph Repair Summary

This is reset-free offline graph repair. Repair edges are selected only
from a Phase 2 support-certified edge bank; no kNN/proximity/latent
unsupported edges are introduced.

## Repair Summary

- `num_repair_edges`: `500`
- `num_repair_nodes`: `186`
- `num_mapped_repair_edges`: `500`
- `mean_repair_score`: `3.7180370912468366`
- `median_repair_support`: `24.0`
- `mean_repair_median_h`: `15.301`

## Planner Metrics

### base / support_shortest_path

- `path_coverage`: `0.17`
- `mean_min_pair_termination_bridge_coverage`: `0.0`
- `mean_pair_incompatible_fraction`: `0.8872549019607843`
- `mean_base_path_cost`: `20.608688463176694`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / compat_penalized

- `path_coverage`: `0.17`
- `mean_min_pair_termination_bridge_coverage`: `0.059259259259259255`
- `mean_pair_incompatible_fraction`: `0.8666666666666667`
- `mean_base_path_cost`: `24.49942903522511`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / compat_threshold

- `path_coverage`: `0.17`
- `mean_min_pair_termination_bridge_coverage`: `0.1416493906497555`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `37.35130972597969`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### repaired / support_shortest_path

- `path_coverage`: `0.65`
- `mean_min_pair_termination_bridge_coverage`: `0.018447360185968056`
- `mean_pair_incompatible_fraction`: `0.791923076923077`
- `mean_base_path_cost`: `32.96110970547937`
- `coverage_delta_vs_base_graph`: `0.48`
- `min_pair_coverage_delta_vs_base_graph`: `0.018447360185968056`
- `pair_incompatible_delta_vs_base_graph`: `-0.09533182503770732`

### repaired / compat_penalized

- `path_coverage`: `0.65`
- `mean_min_pair_termination_bridge_coverage`: `0.04969399503971397`
- `mean_pair_incompatible_fraction`: `0.7473544973544973`
- `mean_base_path_cost`: `35.51401327819563`
- `coverage_delta_vs_base_graph`: `0.48`
- `min_pair_coverage_delta_vs_base_graph`: `-0.009565264219545282`
- `pair_incompatible_delta_vs_base_graph`: `-0.11931216931216937`

### repaired / compat_threshold

- `path_coverage`: `0.64`
- `mean_min_pair_termination_bridge_coverage`: `0.11682294059585083`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `46.16305534675861`
- `coverage_delta_vs_base_graph`: `0.47`
- `min_pair_coverage_delta_vs_base_graph`: `-0.024826450053904686`
- `pair_incompatible_delta_vs_base_graph`: `0.0`

## Interpretation

A positive result is improved compatibility or coverage after adding
only support-certified repair edges. These metrics remain offline
graph evidence and do not prove policy execution success.
