# Stage 1 v2 Launch Notes

- Branch at launch prep: `auto-bars-stage1-3-20260512-164636`
- Planned GPUs: `0,1,2,3`
- GPU adjustment needed: `no`
- Sweep: `configs/sweeps/d4rl_stage1_diag_3090.json`
- Log root: `runs_stage1_diag_v2`
- Base config: `configs/d4rl_antmaze_quick.json`
- Resources: `default_mem_mb=3500`, `max_jobs_per_gpu=4`
- D4RL AntMaze HDF5 verification: `BAD_COUNT=0`
- Sequential prefetch: `passed` for medium-play, medium-diverse, large-play, large-diverse
- Patched sanity gate: `completed`
- Sanity archive: `runs_sanity/antmaze-medium-play-v2/full_bars/sanity_seed0_20260512_170251/archives/logs_sanity_seed0_20260512_170251_1778580373.tar.gz`
- Sanity note: graph build completed successfully but took about 3339s between `graph_build_start` and `graph_build_end` on `antmaze-medium-play-v2`
