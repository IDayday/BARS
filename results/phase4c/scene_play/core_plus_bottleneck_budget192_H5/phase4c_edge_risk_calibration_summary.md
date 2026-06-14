# Phase 4C Edge Risk Calibration Summary

This is reset-free and offline-only. Calibrated reliability is an
offline planning score, not rollout success probability.

## Score Diagnostics

- `original_certified_rate`: `0.11017395888244597`
- `calibrated_certified_rate`: `0.4928835002635741`
- `edge_proxy_score_original_spearman_heldout_support_rate`: `0.7539317748983967`
- `calibrated_edge_reliability_score_spearman_heldout_support_rate`: `0.7567472761441709`
- `edge_proxy_score_original_brier_heldout_support_binary`: `0.4044887804294578`
- `calibrated_edge_reliability_score_brier_heldout_support_binary`: `0.3501750855772045`

## Recommended Planner Config

- `method`: `floor_proxy_penalized_s0012`
- `planner_method`: `floor_proxy_penalized`
- `path_coverage`: `0.16`
- `mean_min_edge_proxy_score`: `0.17772694407013773`
- `mean_uncertified_edge_fraction`: `0.15625`
- `mean_base_path_cost`: `12.762131766095305`
- `risk_weight`: `0.0`
- `ood_weight`: `0.0`
- `incompat_weight`: `0.0`
- `uncertified_weight`: `1.0`
- `min_proxy_score`: `0.0`
- `min_heldout_support_lcb`: `0.0`
- `is_pareto`: `True`

The heldout-support calibration diagnostics are pseudo-label checks.
They validate repeatability/support alignment, not online execution.
