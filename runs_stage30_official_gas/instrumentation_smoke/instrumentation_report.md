# Stage30 Official GAS Instrumentation Report

Status: OFFICIAL_GAS_NON_INVASIVE_TRACE.
Pre-Stage30 BARS/Stage28/Stage29 evidence: ARCHIVED_INTERNAL_EXPLORATION_NOT_GAS_EVIDENCE.
The control loop mirrors official `evaluate_with_graph`; logs are diagnostic only.

## Episode Summary

| env_name | episodes | success_rate | no_path_rate | timeout_rate | stuck_rate | divergence_rate |
| --- | --- | --- | --- | --- | --- | --- |
| antmaze-medium-navigate-v0 | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Edge Categories Observed

- temporal_like_distance_proxy: 85

## Files

- episode traces: `runs_stage30_official_gas/instrumentation_smoke/official_gas_episode_traces.csv`
- path edge traces: `runs_stage30_official_gas/instrumentation_smoke/official_gas_path_edges.csv`
