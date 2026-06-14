# Phase 3A Supervised GCBC Summary

This summary is reset-free and offline-only. It reports action-prediction
MSE on held-out edge BC samples from Phase 2 support-certified option
segments.

Run dir: `results/phase3/antmaze_large_stitch/core_plus_bottleneck`
Phase 2 run dir: `results/phase2/antmaze_large_stitch/core_plus_bottleneck_budget120_H10`

Final train action MSE: `0.02016831`
Final val action MSE: `0.04263893`
Best val action MSE: `0.04263893`

The existing 100000-step GCBC run has final `val_action_mse = 0.0426389`.
This shows that the state-based GCBC model can fit held-out edge BC
samples. It does not prove option-edge executability or online rollout
success.

Rollout remains skipped while Phase 3 preflight reports `env_unavailable`
because this Python environment lacks env-construction dependencies. That
is an environment dependency blocker, not evidence that AntMaze or Scene
lack reset-to-state support.

Note: Offline action MSE is not edge execution success.

## Grouped Metrics

| group | num_edges | num_val_samples | val_action_mse |
| --- | ---: | ---: | ---: |
| all_edges | 433 | 8192 | 0.02938521 |
| high_bottleneck_edges | 220 | 4119 | 0.03021615 |
| low_bottleneck_edges | 213 | 4073 | 0.02852695 |
| high_support_edges | 217 | 6966 | 0.03974281 |
| low_support_edges | 216 | 1226 | 0.01897965 |
| short_horizon_edges | 260 | 7124 | 0.03537017 |
| long_horizon_edges | 173 | 1068 | 0.02039046 |
| high_compatibility_edges | 217 | 6253 | 0.03705667 |
| low_compatibility_edges | 216 | 1939 | 0.02167823 |
