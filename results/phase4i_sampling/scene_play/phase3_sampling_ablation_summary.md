# Phase 3D Sampling Ablation Summary

This is an offline supervised GCBC sampling ablation. It compares action-prediction
MSE under different edge-sampling schemes and does not measure closed-loop rollout
success.

Phase 3C closed-loop execution remains gated on a reliable reset-to-state probe.
Current `env_unavailable` reset probes are environment dependency blockers, not
evidence that the benchmark environments themselves lack reset support.

Dataset: `scene-play-v0`

Top rows by final validation action MSE:

| dataset | phase2_run | sampling_mode | seed | final_val_action_mse | best_val_action_mse | bottleneck_edge_val_mse | non_bottleneck_edge_val_mse | high_support_edge_val_mse | low_support_edge_val_mse | short_horizon_edge_val_mse | long_horizon_edge_val_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene-play-v0 | core_plus_bottleneck_budget192_H5 | uniform_transition | 1 | 0.00801192 | 0.00801192 | 0.00921571 | 0.00881308 | 0.00822321 | 0.00982076 | 0.00878659 | 0.0110742 |
| scene-play-v0 | core_plus_bottleneck_budget192_H5 | uniform_transition | 0 | 0.00807977 | 0.00807977 | 0.00933452 | 0.00901348 | 0.00761439 | 0.0108052 | 0.00933908 | 0.00746976 |
| scene-play-v0 | core_plus_bottleneck_budget192_H5 | uniform_edge | 1 | 0.00900928 | 0.00900928 | 0.00864092 | 0.00922241 | 0.0092221 | 0.00863332 | 0.00898842 | 0.00838874 |
| scene-play-v0 | core_plus_bottleneck_budget192_H5 | uniform_edge | 0 | 0.00934638 | 0.00934638 | 0.00979863 | 0.00918264 | 0.00912746 | 0.00987596 | 0.00965976 | 0.00776082 |
| scene-play-v0 | core_plus_bottleneck_budget192_H5 | bottleneck_weighted | 1 | 0.00954295 | 0.00954295 | 0.00877074 | 0.0100605 | 0.00963947 | 0.00918348 | 0.00948729 | 0.00872262 |
