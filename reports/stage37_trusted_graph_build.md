# Stage37 Trusted ECG Contract Graph Build

Source graph: `results/cage_ecg/transition_contract_graph/contract_graph_augmented.json`
Output dir: `results/cage_ecg/trusted_graph`

| variant | nodes | edges | avg_out | weak | largest_weak | strong | largest_strong | final_rate | recovery_rate | knn_rate | low_contract | high_negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| observed_only | 7467 | 8999 | 1.0479 | 8 | 2225 | 7467 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.1799 | 0.4641 |
| observed_plus_final | 7475 | 11994 | 1.4475 | 8 | 2229 | 7475 | 1 | 0.2497 | 0.0000 | 0.0000 | 0.3307 | 0.4481 |
| trusted_conservative | 7479 | 44635 | 5.3633 | 5 | 5703 | 3139 | 2273 | 0.0671 | 0.0000 | 0.7313 | 0.0889 | 0.2592 |
| full | 7479 | 51542 | 6.1806 | 5 | 5703 | 2522 | 4725 | 0.0581 | 0.0007 | 0.7666 | 0.0770 | 0.3581 |

## Edge Type Distributions

### observed_only

{
  "original_contract": 1542,
  "path_adjacency": 340,
  "temporal_transition": 7117
}

### observed_plus_final

{
  "final_goal_candidate": 2995,
  "original_contract": 1542,
  "path_adjacency": 340,
  "temporal_transition": 7117
}

### trusted_conservative

{
  "final_goal_candidate": 2995,
  "knn_bridge_candidate": 32641,
  "original_contract": 1542,
  "path_adjacency": 340,
  "temporal_transition": 7117
}

### full

{
  "final_goal_candidate": 2995,
  "knn_bridge_candidate": 39514,
  "original_contract": 1542,
  "path_adjacency": 340,
  "recovery_candidate": 34,
  "temporal_transition": 7117
}
