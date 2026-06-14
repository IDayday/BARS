# Phase 4B Calibrated Risk-Aware Planner Sweep

This is reset-free and offline-only. It sweeps support-only risk-aware
planner parameters and does not claim rollout success.

Total sweep configs: `480`
Pareto configs: `212`

## Baselines

| method | coverage | mean min proxy | uncertified frac | base cost |
| --- | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.566 | 0.060 | 0.924 | 50.551 |
| proxy_penalized | 0.566 | 0.128 | 0.874 | 52.575 |
| proxy_threshold | 0.204 | 0.260 | 0.877 | 58.100 |
| certified_only | 0.000 | nan | nan | nan |

## Recommended Config

- `method`: `floor_proxy_penalized_s0066`
- `planner_method`: `floor_proxy_penalized`
- `path_coverage`: `0.544`
- `mean_min_edge_proxy_score`: `0.21668110609255525`
- `mean_uncertified_edge_fraction`: `0.8257949819757937`
- `mean_base_path_cost`: `55.80885162663986`
- `risk_weight`: `0.0`
- `ood_weight`: `1.0`
- `incompat_weight`: `0.0`
- `uncertified_weight`: `1.0`
- `min_proxy_score`: `0.1`
- `min_heldout_support_lcb`: `0.0`
- `is_pareto`: `True`

Interpretation: the recommended config is selected by an offline
coverage/risk heuristic constrained against the support-shortest-path
baseline. It is not a calibrated execution policy.
