# Stage37 ECG Edge Provenance Audit

Graph: `results/cage_ecg/transition_contract_graph/contract_graph_augmented.json`
Planner audit: `results/cage_ecg/transition_contract_planner/offline_plan_audit.csv`

| metric | value |
|---|---:|
| edge_count | 51542 |
| knn_edge_rate | 0.7666 |
| observed_edge_rate | 0.1746 |
| improved_planner_rows | 452 |
| improved_paths_with_knn_rate | 1.0000 |
| improved_paths_observed_only_count | 0 |

## Status

- provenance_status: `KNN_DEPENDENT_PLANNER_SIGNAL`
- trusted_signal_status: `TRUSTED_SIGNAL_NOT_OBSERVED_IN_FULL_AUDIT`

## Planner Edge Type Usage

{
  "knn_bridge_candidate": 11545,
  "original_contract": 112,
  "path_adjacency": 386,
  "recovery_candidate": 9,
  "temporal_transition": 5456
}

## Bottleneck Edge Type Counts

{
  "knn_bridge_candidate": 407,
  "original_contract": 47,
  "path_adjacency": 38,
  "recovery_candidate": 2,
  "temporal_transition": 328
}

KNN bridge candidate 不是观测 transition。若 planner improvement 主要依赖 KNN，则只能算离线候选信号，不能写成已验证可执行路径。
