# Stage29 20ep Online Eval Gate Analysis

- Gate: controlled 20ep online eval; 50ep confirm is not launched by this analysis.
- Stage29-A offline SCG status: PASS.
- Stage29-B execution evidence status: PASS.
- Support score calibration signal: VALIDATED.
- Boundary validation status: UNVALIDATED_NOT_LOADED.
- Reachability validation status: UNVALIDATED_NOT_LOADED.

## Promotion Gate

| planner_id | gate | stitch_regression_gate | navigate_regression_gate | no_path_gate | false_shortcut_gate | envs_compared |
| --- | --- | --- | --- | --- | --- | --- |
| STAGE29_LEXICOGRAPHIC | BLOCKED_BY_20EP_GATE | 1 | 1 | 1 | 1 | 1 |
| SUPPORT_BUDGET_K1 | BLOCKED_BY_20EP_GATE | 1 | 1 | 1 | 1 | 1 |

## Baseline Comparison

| env | planner_id | success_delta_vs_base | no_path_delta_vs_base | baseline_false_shortcut_proxy_rate | planner_false_shortcut_proxy_rate | ready_for_50ep_confirm |
| --- | --- | --- | --- | --- | --- | --- |
| antmaze-medium-stitch-v0 | STAGE29_LEXICOGRAPHIC | 0.30000000000000004 | -0.6000000000000001 | 1.0 | 0.0 | 1 |
| antmaze-medium-stitch-v0 | SUPPORT_BUDGET_K1 | 0.5 | -0.8 | 1.0 | 0.0 | 1 |
