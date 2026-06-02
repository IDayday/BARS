# Stage30 Official GAS Global Diagnosis Collector

Status: OFFICIAL_GAS_GLOBAL_DIAGNOSIS_AGGREGATE.
Pre-Stage30 BARS/Stage28/Stage29 evidence: ARCHIVED_INTERNAL_EXPLORATION_NOT_GAS_EVIDENCE.
No GAS graph, planner, policy, subgoal selection, or action outputs are modified by this collector.

## Aggregate Counts

- episodes: 6000
- edge probe rows: 23040
- unresolved taxonomy: 5534/6000 (0.9223)
- dominant evidence-backed label: `POLICY_LOCAL_FAILURE` count=247, rate=0.0412

## Probe Modes

- exact_semantic_probe: 11520
- nearest_execution_probe: 11520

## Taxonomy By Env

| env_name | label | count | episodes | rate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| antmaze-giant-navigate-v0 | LONG_HOP_FAILURE | 21 | 1500 | 0.0140 | 0.0092 | 0.0213 |
| antmaze-giant-navigate-v0 | SUBGOAL_SEQUENCE_DRIFT | 119 | 1500 | 0.0793 | 0.0667 | 0.0941 |
| antmaze-giant-navigate-v0 | POLICY_LOCAL_FAILURE | 85 | 1500 | 0.0567 | 0.0461 | 0.0695 |
| antmaze-giant-navigate-v0 | UNRESOLVED | 1275 | 1500 | 0.8500 | 0.8310 | 0.8672 |
| antmaze-giant-stitch-v0 | LONG_HOP_FAILURE | 11 | 1500 | 0.0073 | 0.0041 | 0.0131 |
| antmaze-giant-stitch-v0 | SUBGOAL_SEQUENCE_DRIFT | 41 | 1500 | 0.0273 | 0.0202 | 0.0369 |
| antmaze-giant-stitch-v0 | POLICY_LOCAL_FAILURE | 101 | 1500 | 0.0673 | 0.0557 | 0.0812 |
| antmaze-giant-stitch-v0 | UNRESOLVED | 1347 | 1500 | 0.8980 | 0.8817 | 0.9123 |
| antmaze-medium-navigate-v0 | LONG_HOP_FAILURE | 3 | 1500 | 0.0020 | 0.0007 | 0.0059 |
| antmaze-medium-navigate-v0 | SUBGOAL_SEQUENCE_DRIFT | 11 | 1500 | 0.0073 | 0.0041 | 0.0131 |
| antmaze-medium-navigate-v0 | POLICY_LOCAL_FAILURE | 18 | 1500 | 0.0120 | 0.0076 | 0.0189 |
| antmaze-medium-navigate-v0 | UNRESOLVED | 1468 | 1500 | 0.9787 | 0.9700 | 0.9848 |
| antmaze-medium-stitch-v0 | LONG_HOP_FAILURE | 2 | 1500 | 0.0013 | 0.0004 | 0.0048 |
| antmaze-medium-stitch-v0 | SUBGOAL_SEQUENCE_DRIFT | 11 | 1500 | 0.0073 | 0.0041 | 0.0131 |
| antmaze-medium-stitch-v0 | POLICY_LOCAL_FAILURE | 43 | 1500 | 0.0287 | 0.0214 | 0.0384 |
| antmaze-medium-stitch-v0 | UNRESOLVED | 1444 | 1500 | 0.9627 | 0.9518 | 0.9711 |

## Files

- all episode traces: `runs_stage30_official_gas/layered_full_official_gas_diag_active/global/stage30_all_episode_traces.csv`
- all path edges: `runs_stage30_official_gas/layered_full_official_gas_diag_active/global/stage30_all_path_edges.csv`
- all keygraph edges: `runs_stage30_official_gas/layered_full_official_gas_diag_active/global/stage30_all_keygraph_edges.csv`
- all edge probes: `runs_stage30_official_gas/layered_full_official_gas_diag_active/global/stage30_all_edge_probe.csv`
- global taxonomy: `runs_stage30_official_gas/layered_full_official_gas_diag_active/global/stage30_failure_taxonomy.csv`
