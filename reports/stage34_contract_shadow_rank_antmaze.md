# Stage34 Contract-Intervention Deployment Analysis

本表汇总 CAGE-v0.4 shadow/intervention 的 override、干预、committed watchdog、final phase 和局部安全循环代理指标。缺失字段记为 NA。

## Grouped

| committed_lockout_count_mean | committed_lockout_step_count_mean | committed_stale_count_mean | committed_usage_rate_mean | env_name | final_goal_on_rate_mean | final_phase_override_count_mean | final_phase_preserved_count_mean | global_replan_request_count_mean | intervention_count_mean | intervention_rate_mean | local_safe_loop_count | mean_intervention_gain_mean | mean_segment_progress_mean | num_jobs | segment_target_reach_rate_mean | shadow_mean_override_margin_mean | shadow_override_final_phase_rate_mean | shadow_override_on_success_rate_mean | shadow_override_rate_mean | source_cage_rate_mean | source_committed_rate_mean | source_gas_rate_mean | stall_count_mean | success_rate_mean | variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-navigate-v0 | 0.7200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | NA | 0.2215 | 1 | 0.0531 | -0.2698 | 0.0043 | 0.1002 | 0.0820 | NA | NA | NA | 7.2800 | 0.6000 | cage_contract_shadow_rank |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-navigate-v0 | 0.7200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | NA | 0.2229 | 1 | 0.0528 | NA | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA | 0.0000 | 0.6400 | cage_trace_only |
| NA | NA | NA | NA | antmaze-giant-navigate-v0 | NA | NA | NA | NA | NA | NA | 0 | NA | NA | 1 | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0.6800 | gas |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-stitch-v0 | 0.8400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | NA | 0.2949 | 1 | 0.0449 | -0.1736 | 0.0575 | 0.1980 | 0.1924 | NA | NA | NA | 7.9200 | 0.8000 | cage_contract_shadow_rank |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-stitch-v0 | 0.8400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | NA | 0.2935 | 1 | 0.0449 | NA | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA | 0.0000 | 0.8400 | cage_trace_only |
| NA | NA | NA | NA | antmaze-giant-stitch-v0 | NA | NA | NA | NA | NA | NA | 0 | NA | NA | 1 | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0.8000 | gas |

## Job Rows

| committed_lockout_count | committed_lockout_step_count | committed_stale_count | committed_usage_rate | env_name | final_goal_on_rate | final_phase_override_count | final_phase_preserved_count | global_replan_request_count | intervention_count | intervention_rate | job_id | local_safe_loop | mean_intervention_gain | mean_segment_progress | seed | segment_target_reach_rate | selected_source_distribution | shadow_mean_override_margin | shadow_override_final_phase_rate | shadow_override_on_success_rate | shadow_override_rate | source_cage_rate | source_committed_rate | source_gas_rate | stall_count | status | success_rate | variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NA | NA | NA | NA | antmaze-giant-navigate-v0 | NA | NA | NA | NA | NA | NA | antmaze-giant-navigate-v0__seed42__gas | False | NA | NA | 42 | NA | {} | NA | NA | NA | NA | NA | NA | NA | NA | succeeded | 0.6800 | gas |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-navigate-v0 | 0.7200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | antmaze-giant-navigate-v0__seed42__cage_trace_only | False | NA | 0.2229 | 42 | 0.0528 | {} | NA | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA | 0.0000 | succeeded | 0.6400 | cage_trace_only |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-navigate-v0 | 0.7200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | antmaze-giant-navigate-v0__seed42__cage_contract_shadow_rank | False | NA | 0.2215 | 42 | 0.0531 | {} | -0.2698 | 0.0043 | 0.1002 | 0.0820 | NA | NA | NA | 7.2800 | succeeded | 0.6000 | cage_contract_shadow_rank |
| NA | NA | NA | NA | antmaze-giant-stitch-v0 | NA | NA | NA | NA | NA | NA | antmaze-giant-stitch-v0__seed42__gas | False | NA | NA | 42 | NA | {} | NA | NA | NA | NA | NA | NA | NA | NA | succeeded | 0.8000 | gas |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-stitch-v0 | 0.8400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | antmaze-giant-stitch-v0__seed42__cage_trace_only | False | NA | 0.2935 | 42 | 0.0449 | {} | NA | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA | 0.0000 | succeeded | 0.8400 | cage_trace_only |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-stitch-v0 | 0.8400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | antmaze-giant-stitch-v0__seed42__cage_contract_shadow_rank | False | NA | 0.2949 | 42 | 0.0449 | {} | -0.1736 | 0.0575 | 0.1980 | 0.1924 | NA | NA | NA | 7.9200 | succeeded | 0.8000 | cage_contract_shadow_rank |
