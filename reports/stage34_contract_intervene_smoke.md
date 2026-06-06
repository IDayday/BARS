# Stage34 Contract-Intervention Deployment Analysis

本表汇总 CAGE-v0.4 shadow/intervention 的 override、干预、committed watchdog、final phase 和局部安全循环代理指标。缺失字段记为 NA。

## Grouped

| committed_lockout_count_mean | committed_lockout_step_count_mean | committed_stale_count_mean | committed_usage_rate_mean | env_name | final_goal_on_rate_mean | final_phase_override_count_mean | final_phase_preserved_count_mean | global_replan_request_count_mean | intervention_count_mean | intervention_rate_mean | local_safe_loop_count | mean_intervention_gain_mean | mean_segment_progress_mean | num_jobs | segment_target_reach_rate_mean | shadow_mean_override_margin_mean | shadow_override_final_phase_rate_mean | shadow_override_on_success_rate_mean | shadow_override_rate_mean | source_cage_rate_mean | source_committed_rate_mean | source_gas_rate_mean | stall_count_mean | success_rate_mean | variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9.0000 | 171.0000 | 2.0000 | NA | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 0.0030 | 0 | -0.3707 | 0.1703 | 1 | 0.0526 | NA | 0.0000 | NA | 0.0000 | NA | NA | 1.0000 | 25.0000 | 0.0000 | cage_contract_intervene |
| 0.0000 | 0.0000 | 0.0000 | 0.3977 | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | NA | 0.1716 | 1 | 0.4717 | NA | 0.0000 | NA | 0.0000 | 0.2159 | 0.3977 | 0.2955 | 27.0000 | 0.0000 | cage_contract_rank |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-navigate-v0 | 1.0000 | 0.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | 0 | NA | 0.2280 | 1 | 0.0109 | NA | 0.0000 | NA | 0.0000 | NA | NA | NA | 12.0000 | 0.0000 | cage_safe_full |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-navigate-v0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | NA | 0.2203 | 1 | 0.0780 | NA | 0.0000 | NA | 0.0000 | NA | NA | NA | 0.0000 | 0.0000 | cage_trace_only |
| NA | NA | NA | NA | antmaze-giant-navigate-v0 | NA | NA | NA | NA | NA | NA | 0 | NA | NA | 1 | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0.0000 | gas |

## Job Rows

| committed_lockout_count | committed_lockout_step_count | committed_stale_count | committed_usage_rate | env_name | final_goal_on_rate | final_phase_override_count | final_phase_preserved_count | global_replan_request_count | intervention_count | intervention_rate | job_id | local_safe_loop | mean_intervention_gain | mean_segment_progress | seed | segment_target_reach_rate | selected_source_distribution | shadow_mean_override_margin | shadow_override_final_phase_rate | shadow_override_on_success_rate | shadow_override_rate | source_cage_rate | source_committed_rate | source_gas_rate | stall_count | status | success_rate | variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NA | NA | NA | NA | antmaze-giant-navigate-v0 | NA | NA | NA | NA | NA | NA | antmaze-giant-navigate-v0__seed42__gas | False | NA | NA | 42 | NA | {} | NA | NA | NA | NA | NA | NA | NA | NA | succeeded | 0.0000 | gas |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-navigate-v0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | antmaze-giant-navigate-v0__seed42__cage_trace_only | False | NA | 0.2203 | 42 | 0.0780 | {} | NA | 0.0000 | NA | 0.0000 | NA | NA | NA | 0.0000 | succeeded | 0.0000 | cage_trace_only |
| 0.0000 | 0.0000 | 0.0000 | NA | antmaze-giant-navigate-v0 | 1.0000 | 0.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | antmaze-giant-navigate-v0__seed42__cage_safe_full | False | NA | 0.2280 | 42 | 0.0109 | {} | NA | 0.0000 | NA | 0.0000 | NA | NA | NA | 12.0000 | succeeded | 0.0000 | cage_safe_full |
| 0.0000 | 0.0000 | 0.0000 | 0.3977 | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | antmaze-giant-navigate-v0__seed42__cage_contract_rank | False | NA | 0.1716 | 42 | 0.4717 | {"cage": 0.2159090909090909, "committed": 0.3977272727272727, "gas": 0.29545454545454547, "path_later": 0.09090909090909091} | NA | 0.0000 | NA | 0.0000 | 0.2159 | 0.3977 | 0.2955 | 27.0000 | succeeded | 0.0000 | cage_contract_rank |
| 9.0000 | 171.0000 | 2.0000 | NA | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 0.0030 | antmaze-giant-navigate-v0__seed42__cage_contract_intervene | False | -0.3707 | 0.1703 | 42 | 0.0526 | {"gas": 1.0} | NA | 0.0000 | NA | 0.0000 | NA | NA | 1.0000 | 25.0000 | succeeded | 0.0000 | cage_contract_intervene |
