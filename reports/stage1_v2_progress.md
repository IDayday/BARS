# Stage 1 v2 Progress

Last updated: 2026-05-12 18:10 CST

## Preflight

- Old `runs_stage1_diag` jobs: stopped
- Old `runs_stage1_diag` logs: archived for all 12 runs
- Patched sanity gate: completed
- D4RL AntMaze HDF5 verification: passed
- D4RL AntMaze sequential prefetch: passed for all 4 envs

## Current Stage 1 v2 Status

- Log root: `runs_stage1_diag_v2`
- Sweep: `configs/sweeps/d4rl_stage1_diag_3090.json`
- Planned runs: 12
- Launched runs: 12
- Current state snapshot: 12 running, 0 completed, 0 failed, 0 stale
- Scheduler PID: `161049`
- Monitor PID: `161474`

## Notes

- GPUs `0,1,2,3` were used as planned.
- Initial runs are progressing through `train_tdr`, `train_policy`, `train_reachability`, and early `graph_build_start`.
- No `dataset_truncated` or immediate infrastructure failures were observed during launch.
- Stage 1 v2 analysis is not available yet because the sweep is still running.
