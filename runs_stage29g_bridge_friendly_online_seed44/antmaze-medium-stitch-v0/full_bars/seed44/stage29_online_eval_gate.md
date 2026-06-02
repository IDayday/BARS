# Stage29 20ep Online Eval Gate

- Evidence class: controlled 20ep online evaluation with fallback_mode=none.
- Stage29-A offline SCG status: PASS.
- Stage29-B execution evidence status: PASS.
- Support score calibration signal: VALIDATED.
- Boundary validation status: UNVALIDATED_NOT_LOADED.
- Reachability validation status: UNVALIDATED_NOT_LOADED.
- 50ep confirm launched: 0.

## Summary

| planner_id | episodes | success_rate | no_path_rate | path_cross_rate | unsupported_edge_count_mean | subgoal_reach_rate | edge_reach_rate | timeout_rate | stuck_rate | divergence_rate | false_shortcut_proxy_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BARS_BASE | 20 | 0.1 | 0.8 | 1.0 | 2.15 | 0.45454545454545453 | 0.45454545454545453 | 0.1 | 0.1 | 0.1 | 1.0 |
| STAGE29_LEXICOGRAPHIC | 20 | 0.4 | 0.2 | 0.4930555555555555 | 0.0 | 0.459375 | 0.459375 | 0.4 | 0.15 | 0.15 | 0.0 |
| SUPPORT_BUDGET_K1 | 20 | 0.6 | 0.0 | 0.6537012987012987 | 0.0 | 0.5420111832611832 | 0.5420111832611832 | 0.4 | 0.3 | 0.25 | 0.0 |
