# Stage36 Transition Contract Graph Build

Graph: `results/cage_ecg/transition_contract_graph/contract_graph.json`

| metric | value |
|---|---:|
| node_count | 7479 |
| edge_count | 51542 |
| boundary_contract_count | 28848 |
| weak_component_count | 5 |
| largest_weak_component | 5703 |
| strong_component_count | 2522 |
| largest_strong_component | 4725 |
| avg_out_degree | 6.1806 |
| path_pair_reachability_proxy | 0.0037 |
| low_contract_edge_rate | 0.0770 |
| high_negative_edge_rate | 0.3581 |
| final_goal_edge_rate | 0.0581 |
| recovery_edge_rate | 0.0007 |
| qtrain_supported_edge_rate | 0.0000 |
| transition_edge_rate | 0.9113 |

## Edge Type Counts

{
  "final_goal_candidate": 2995,
  "knn_bridge_candidate": 39514,
  "original_contract": 1542,
  "path_adjacency": 340,
  "recovery_candidate": 34,
  "temporal_transition": 7117
}

## Notes

No online benchmark was run. KNN bridge edges are candidate connectivity edges, not observed transitions.
