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
| scene-play-v0 | core_plus_bottleneck_budget192_H5 | uniform_transition | 0 | 0.0245533 | 0.0245533 | 0.0296357 | 0.0287219 | 0.0231865 | 0.0354173 | 0.0297268 | 0.023397 |
| scene-play-v0 | core_plus_bottleneck_budget192_H5 | uniform_edge | 0 | 0.0251311 | 0.0251311 | 0.028884 | 0.0281007 | 0.0237892 | 0.0333895 | 0.0289567 | 0.0235937 |
| scene-play-v0 | core_plus_bottleneck_budget192_H5 | bottleneck_weighted | 0 | 0.0255824 | 0.0255824 | 0.0291003 | 0.0288295 | 0.0244341 | 0.033677 | 0.0292601 | 0.0258362 |
| scene-play-v0 | core_plus_bottleneck_budget192_H10 | uniform_transition | 0 | 0.0293939 | 0.0293939 | 0.0353824 | 0.0333515 | 0.0296834 | 0.0391446 | 0.0361891 | 0.0311623 |
| scene-play-v0 | core_plus_bottleneck_budget192_H10 | bottleneck_weighted | 0 | 0.0303283 | 0.0303283 | 0.0344187 | 0.0335037 | 0.0308039 | 0.0371745 | 0.0355914 | 0.0310739 |
