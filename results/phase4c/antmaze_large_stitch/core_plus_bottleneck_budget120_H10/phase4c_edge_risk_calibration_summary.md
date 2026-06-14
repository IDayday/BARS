# Phase 4C Edge Risk Calibration Summary

This is reset-free and offline-only. Calibrated reliability is an
offline planning score, not rollout success probability.

## Score Diagnostics

- `original_certified_rate`: `0.06872852233676977`
- `calibrated_certified_rate`: `0.7457044673539519`
- `edge_proxy_score_original_spearman_heldout_support_rate`: `0.732218571723822`
- `calibrated_edge_reliability_score_spearman_heldout_support_rate`: `0.6877604817407548`
- `edge_proxy_score_original_brier_heldout_support_binary`: `0.4896607643473668`
- `calibrated_edge_reliability_score_brier_heldout_support_binary`: `0.38849103544669594`

## Recommended Planner Config

- `method`: `floor_proxy_penalized_s0085`
- `planner_method`: `floor_proxy_penalized`
- `path_coverage`: `0.566`
- `mean_min_edge_proxy_score`: `0.2731775091559763`
- `mean_uncertified_edge_fraction`: `0.0126801193290555`
- `mean_base_path_cost`: `57.89548130663446`
- `risk_weight`: `0.0`
- `ood_weight`: `1.0`
- `incompat_weight`: `1.0`
- `uncertified_weight`: `1.0`
- `min_proxy_score`: `0.0`
- `min_heldout_support_lcb`: `0.01`
- `is_pareto`: `True`

The heldout-support calibration diagnostics are pseudo-label checks.
They validate repeatability/support alignment, not online execution.
