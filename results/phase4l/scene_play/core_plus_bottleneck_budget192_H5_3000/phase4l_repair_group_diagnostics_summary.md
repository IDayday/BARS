# Phase 4L Direct Repair-Edge Group Diagnostics

Phase 4L compares each loss-weighted checkpoint against the matched
uniform-transition baseline on the same repair edges and seed. The
metric is direct repair-edge supervised action MSE, not rollout success.

## Method Deltas

| method | num_edges | mean_edge_action_mse_delta | mean_edge_action_mse_ratio | fraction_edges_improved | planner_usage_rate |
| --- | --- | --- | --- | --- | --- |
| loss_bottleneck_s03 | 500 | 0.000152484 | 1.02832 | 0.381 | 0.178 |
| loss_support_bottleneck_s03 | 500 | -0.000210621 | 1.00047 | 0.497 | 0.178 |
| loss_support_s03 | 500 | 2.25201e-05 | 1.02464 | 0.456 | 0.178 |

## Top Group Findings

| method | group_type | group_value | num_edges | mean_edge_action_mse_delta | mean_edge_action_mse_ratio | fraction_edges_improved | planner_usage_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss_support_bottleneck_s03 | horizon_group | long_horizon | 54 | -0.000417017 | 0.992832 | 0.490741 | 0.12963 |
| loss_support_bottleneck_s03 | support_group | low_support | 259 | -0.000421409 | 0.996485 | 0.532819 | 0.0810811 |
| loss_support_s03 | horizon_group | long_horizon | 54 | -0.000387332 | 1.02403 | 0.453704 | 0.12963 |
| loss_support_bottleneck_s03 | compatibility_group | high_compatibility | 250 | -0.000277284 | 0.998118 | 0.5 | 0.272 |
| loss_support_bottleneck_s03 | bottleneck_group | high_bottleneck | 240 | -0.000309236 | 0.999162 | 0.5125 | 0.1375 |
| loss_support_bottleneck_s03 | repair_reason | low_compatibility_junction | 500 | -0.000210621 | 1.00047 | 0.497 | 0.178 |
| loss_support_bottleneck_s03 | planner_usage_group | not_planner_used | 411 | -0.00024171 | 0.998776 | 0.508516 | 0 |
| loss_support_bottleneck_s03 | horizon_group | short_horizon | 446 | -0.000185631 | 1.00139 | 0.497758 | 0.183857 |
| loss_support_bottleneck_s03 | compatibility_group | low_compatibility | 250 | -0.000143958 | 1.00282 | 0.494 | 0.084 |
| loss_support_bottleneck_s03 | bottleneck_group | low_bottleneck | 260 | -0.000119592 | 1.00168 | 0.482692 | 0.215385 |
| loss_support_bottleneck_s03 | planner_usage_group | planner_used | 89 | -6.70539e-05 | 1.00829 | 0.44382 | 1 |
| loss_bottleneck_s03 | horizon_group | long_horizon | 54 | -0.000101987 | 1.02292 | 0.407407 | 0.12963 |
| loss_support_s03 | support_group | low_support | 259 | -2.87731e-05 | 1.0347 | 0.472973 | 0.0810811 |
| loss_support_s03 | compatibility_group | low_compatibility | 250 | -4.1449e-06 | 1.03487 | 0.484 | 0.084 |
| loss_support_s03 | bottleneck_group | low_bottleneck | 260 | -2.35115e-08 | 1.02295 | 0.440385 | 0.215385 |
| loss_support_s03 | planner_usage_group | not_planner_used | 411 | 2.94976e-06 | 1.02297 | 0.470803 | 0 |

## Interpretation

Negative MSE deltas mean the candidate checkpoint fits the repair edge
better than the matched uniform-transition baseline. Planner-used groups
are more relevant to the current repaired planner than unused groups,
but this remains an offline supervised proxy.
