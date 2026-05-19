# Stage22 Live Summary

Updated: 2026-05-19 09:44:36

## Jobs
- completed: 35
- failed: 4
- failed classified: 4

## Eval
```csv
env,seed,variant,budget,episodes,success
antmaze-medium-navigate-v0,0,gas_boundary_budget,2.0,100,0.0
antmaze-medium-navigate-v0,0,gas_boundary_budget,3.0,100,0.0
antmaze-medium-navigate-v0,0,gas_reachability_budget,2.0,100,0.88
antmaze-medium-navigate-v0,0,gas_reachability_budget,3.0,100,0.85
antmaze-medium-navigate-v0,0,gas_shortest,2.0,100,0.86
antmaze-medium-navigate-v0,0,gas_shortest,3.0,100,0.86
antmaze-medium-stitch-v0,0,gas_boundary_budget,2.0,100,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,3.0,100,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,5.0,6,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,8.0,6,1.0
antmaze-medium-stitch-v0,0,gas_reachability_budget,2.0,100,0.86
antmaze-medium-stitch-v0,0,gas_reachability_budget,3.0,100,0.93
antmaze-medium-stitch-v0,0,gas_reachability_budget,5.0,100,0.89
antmaze-medium-stitch-v0,0,gas_reachability_budget,8.0,100,0.89
antmaze-medium-stitch-v0,0,gas_shortest,2.0,100,0.89
antmaze-medium-stitch-v0,0,gas_shortest,3.0,100,0.9
```

## Failures
- boundary feasibility: runs_stage22_eval_logs/antmaze-medium-stitch-v0/seed0/gas_boundary_budget/budget8.0/fallback_progress_stall_v2/evaluate.log
- boundary feasibility: runs_stage22_eval_logs/antmaze-medium-stitch-v0/seed0/gas_boundary_budget/budget8.0/fallback_none/evaluate.log
- boundary feasibility: runs_stage22_eval_logs/antmaze-medium-stitch-v0/seed0/gas_boundary_budget/budget5.0/fallback_progress_stall_v2/evaluate.log
- boundary feasibility: runs_stage22_eval_logs/antmaze-medium-stitch-v0/seed0/gas_boundary_budget/budget5.0/fallback_none/evaluate.log