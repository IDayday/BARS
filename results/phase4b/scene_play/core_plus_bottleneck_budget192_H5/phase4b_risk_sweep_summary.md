# Phase 4B Calibrated Risk-Aware Planner Sweep

This is reset-free and offline-only. It sweeps support-only risk-aware
planner parameters and does not claim rollout success.

Total sweep configs: `480`
Pareto configs: `182`

## Baselines

| method | coverage | mean min proxy | uncertified frac | base cost |
| --- | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.160 | 0.065 | 0.984 | 10.796 |
| proxy_penalized | 0.160 | 0.104 | 0.855 | 11.338 |
| proxy_threshold | 0.130 | 0.275 | 0.822 | 16.750 |
| certified_only | 0.010 | 0.351 | 0.000 | 14.884 |

## Recommended Config

- `method`: `floor_proxy_penalized_s0060`
- `planner_method`: `floor_proxy_penalized`
- `path_coverage`: `0.16`
- `mean_min_edge_proxy_score`: `0.09205681787390688`
- `mean_uncertified_edge_fraction`: `0.7333333333333333`
- `mean_base_path_cost`: `11.705876253435967`
- `risk_weight`: `0.0`
- `ood_weight`: `1.0`
- `incompat_weight`: `0.0`
- `uncertified_weight`: `1.0`
- `min_proxy_score`: `0.0`
- `min_heldout_support_lcb`: `0.0`
- `is_pareto`: `True`

Interpretation: the recommended config is selected by an offline
coverage/risk heuristic constrained against the support-shortest-path
baseline. It is not a calibrated execution policy.
