# Phase 4E Compatibility Graph Repair Summary

This is reset-free offline graph repair. Repair edges are selected only
from a Phase 2 support-certified edge bank; no kNN/proximity/latent
unsupported edges are introduced.

## Repair Summary

- `num_repair_edges`: `200`
- `num_repair_nodes`: `100`
- `num_mapped_repair_edges`: `200`
- `mean_repair_score`: `2.7393009851872567`
- `median_repair_support`: `243.5`
- `mean_repair_median_h`: `7.4425`

## Planner Metrics

### base / support_shortest_path

- `path_coverage`: `0.566`
- `mean_min_pair_termination_bridge_coverage`: `0.04297249099177845`
- `mean_pair_incompatible_fraction`: `0.16082443653618028`
- `mean_base_path_cost`: `50.55063453182929`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / compat_penalized

- `path_coverage`: `0.566`
- `mean_min_pair_termination_bridge_coverage`: `0.07019721595507995`
- `mean_pair_incompatible_fraction`: `0.10465907165195422`
- `mean_base_path_cost`: `52.0222466818775`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / calibrated_compat_penalized

- `path_coverage`: `0.566`
- `mean_min_pair_termination_bridge_coverage`: `0.11307781049591503`
- `mean_pair_incompatible_fraction`: `0.03318543852351682`
- `mean_base_path_cost`: `55.51073441091786`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / compat_threshold

- `path_coverage`: `0.544`
- `mean_min_pair_termination_bridge_coverage`: `0.11465729225448132`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `52.79297678323901`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### base / calibrated_compat_threshold

- `path_coverage`: `0.544`
- `mean_min_pair_termination_bridge_coverage`: `0.14568066310451566`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `56.023667894744385`
- `coverage_delta_vs_base_graph`: `nan`
- `min_pair_coverage_delta_vs_base_graph`: `nan`
- `pair_incompatible_delta_vs_base_graph`: `nan`

### repaired / support_shortest_path

- `path_coverage`: `0.642`
- `mean_min_pair_termination_bridge_coverage`: `0.041051721005793775`
- `mean_pair_incompatible_fraction`: `0.1565121659949246`
- `mean_base_path_cost`: `50.962350413397495`
- `coverage_delta_vs_base_graph`: `0.07600000000000007`
- `min_pair_coverage_delta_vs_base_graph`: `-0.0019207699859846786`
- `pair_incompatible_delta_vs_base_graph`: `-0.004312270541255686`

### repaired / compat_penalized

- `path_coverage`: `0.642`
- `mean_min_pair_termination_bridge_coverage`: `0.06832921523167573`
- `mean_pair_incompatible_fraction`: `0.10033485323453974`
- `mean_base_path_cost`: `52.38928361012224`
- `coverage_delta_vs_base_graph`: `0.07600000000000007`
- `min_pair_coverage_delta_vs_base_graph`: `-0.00186800072340422`
- `pair_incompatible_delta_vs_base_graph`: `-0.004324218417414483`

### repaired / calibrated_compat_penalized

- `path_coverage`: `0.642`
- `mean_min_pair_termination_bridge_coverage`: `0.11367149492338868`
- `mean_pair_incompatible_fraction`: `0.03683046316275156`
- `mean_base_path_cost`: `56.03219782210039`
- `coverage_delta_vs_base_graph`: `0.07600000000000007`
- `min_pair_coverage_delta_vs_base_graph`: `0.0005936844274736508`
- `pair_incompatible_delta_vs_base_graph`: `0.003645024639234745`

### repaired / compat_threshold

- `path_coverage`: `0.62`
- `mean_min_pair_termination_bridge_coverage`: `0.11568922975078531`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `53.36083744676137`
- `coverage_delta_vs_base_graph`: `0.07599999999999996`
- `min_pair_coverage_delta_vs_base_graph`: `0.0010319374963039946`
- `pair_incompatible_delta_vs_base_graph`: `0.0`

### repaired / calibrated_compat_threshold

- `path_coverage`: `0.62`
- `mean_min_pair_termination_bridge_coverage`: `0.1487112720280166`
- `mean_pair_incompatible_fraction`: `0.0`
- `mean_base_path_cost`: `56.93541968998181`
- `coverage_delta_vs_base_graph`: `0.07599999999999996`
- `min_pair_coverage_delta_vs_base_graph`: `0.0030306089235009326`
- `pair_incompatible_delta_vs_base_graph`: `0.0`

## Interpretation

A positive result is improved compatibility or coverage after adding
only support-certified repair edges. These metrics remain offline
graph evidence and do not prove policy execution success.
