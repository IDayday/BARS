# Stage23 Decisions

- GO_REACHABILITY_SEED_EXPANSION
- HOLD_FINAL_REACHABILITY_CLAIM
- REPAIR_FALLBACK_V3
- HOLD_BOUNDARY
- GO_D4RL_PROTOCOL_REPAIR
- HOLD_INTEGRATED_BARS_V3

## Evidence

- Completed Stage23 key-claim medium seed0 matrix: 12 jobs, 1200 episodes, 0 failed jobs.
- `fallback=none`: reachability is positive on both medium envs.
- `progress_stall_v3`: relative gains exist in some cells, but absolute success is lower than `fallback=none`; keep it out of the planner claim until causal ablation passes.
- Boundary remains diagnostic-only because Stage22R boundary budget reject rate was about 0.985.
- Official-control adapter protocol repair is within 1.4pp of official GAS route B.
