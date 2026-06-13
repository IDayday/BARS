# Phase 3D Sampling Ablation Summary

This is an offline supervised GCBC sampling ablation. It compares action-prediction
MSE under different edge-sampling schemes and does not measure closed-loop rollout
success.

Phase 3C closed-loop execution remains gated on a reliable reset-to-state probe.
Current `env_unavailable` reset probes are environment dependency blockers, not
evidence that the benchmark environments themselves lack reset support.

Dataset: `antmaze-large-stitch-v0`

Top rows by final validation action MSE:

| dataset | phase2_run | sampling_mode | seed | final_val_action_mse | best_val_action_mse | bottleneck_edge_val_mse | non_bottleneck_edge_val_mse | high_support_edge_val_mse | low_support_edge_val_mse | short_horizon_edge_val_mse | long_horizon_edge_val_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-large-stitch-v0 | core_plus_bottleneck_budget120_H10 | uniform_transition | 0 | 0.20759 | 0.20759 | 0.192223 | 0.207276 | 0.20635 | 0.192619 | 0.199653 | 0.199224 |
| antmaze-large-stitch-v0 | core_plus_bottleneck_budget120_H10 | bottleneck_weighted | 0 | 0.21067 | 0.21067 | 0.192964 | 0.205668 | 0.209443 | 0.188719 | 0.199092 | 0.199134 |
| antmaze-large-stitch-v0 | core_plus_bottleneck_budget120_H10 | uniform_edge | 0 | 0.2118 | 0.2118 | 0.195536 | 0.20386 | 0.210851 | 0.188213 | 0.199943 | 0.198858 |
| antmaze-large-stitch-v0 | density_budget120_H10 | uniform_transition | 0 | 0.212987 | 0.212987 | 0.190071 | 0.207726 | 0.214392 | 0.18281 | 0.197329 | 0.201218 |
| antmaze-large-stitch-v0 | density_budget120_H10 | bottleneck_weighted | 0 | 0.214098 | 0.214098 | 0.188595 | 0.203568 | 0.217407 | 0.174207 | 0.196692 | 0.194241 |
