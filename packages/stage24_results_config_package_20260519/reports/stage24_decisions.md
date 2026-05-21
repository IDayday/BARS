# Stage24 Decisions

## Gates
- reachability_confirm: HOLD_REACHABILITY_WEAK_EFFECT
- local_drift_repair: HOLD_LOCAL_DRIFT_REPAIR
- oracle_headroom: PENDING_ORACLE_HEADROOM_SCAN
- p_bridge: SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM
- boundary: HOLD_BOUNDARY_DIAGNOSTIC_ONLY
- integrated: SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE
- d4rl_protocol: HOLD_D4RL_PROTOCOL_REPAIR

## Evidence

Reachability confirmation:
```csv
env,seed,variant,success,steps,success_delta_vs_shortest,steps_inflation_vs_shortest,source
antmaze-medium-navigate-v0,0,gas_reachability_budget_calibrated,0.92,341.49,0.030000000000000027,-0.03296236513465292,stage23_seed0_prior
antmaze-medium-navigate-v0,0,gas_reachability_soft_calibrated,0.9,337.86,0.010000000000000009,-0.04324186560190293,stage23_seed0_prior
antmaze-medium-navigate-v0,0,gas_shortest,0.89,353.13,,,stage23_seed0_prior
antmaze-medium-stitch-v0,0,gas_reachability_budget_calibrated,0.87,349.21,0.010000000000000009,-0.05316956781085628,stage23_seed0_prior
antmaze-medium-stitch-v0,0,gas_reachability_soft_calibrated,0.91,327.88,0.050000000000000044,-0.11100265712271569,stage23_seed0_prior
antmaze-medium-stitch-v0,0,gas_shortest,0.86,368.82,,,stage23_seed0_prior
```

Oracle headroom:
_No rows yet._

Local drift:
```csv
env,seed,variant,success,primary_failure_type,local_drift_score,progress_stall_count,oscillation_score
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.0,0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.0,0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.0,0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.0,0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.0,0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,0,F6_LONG_PATH_ACCUMULATION,0.0,0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.0,0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.0,0,0.0
```

## Decision
- HOLD integrated BARS-v3: oracle headroom has not passed for any Stage24 env/seed.
- HOLD reachability claim until seeds 1/2 complete and paired gates pass.
- HOLD local-drift repair claim until a repair cuts F4 failures by at least 30% without success loss.
- STOP using progress_stall_v3/direct-goal fallback as planner evidence.
- KEEP boundary diagnostic-only unless coverage, supported-gap, and psi gates pass.

## Next commands
```bash
bash scripts/stage24_run_reachability_confirm.sh CONFIG=configs/stage24_reachability_confirm.json ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0,1,2 EPISODES=100 GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1
bash scripts/stage24_run_reachability_confirm.sh CONFIG=configs/stage24_reachability_confirm.json ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0,1,2 EPISODES=100 VARIANTS=gas_shortest_replan_on_local_drift,gas_shortest_adaptive_subgoal_horizon,gas_reachability_budget_replan_on_local_drift STAGE24_ROOT=runs_stage24_local_drift LOG_ROOT=runs_stage24_local_drift_logs GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1
bash scripts/stage24_oracle_headroom_scan.sh ENVS=scene-play-v0 SEEDS=0 GPUS=${GPUS:-0} EDGE_EXEC_PILOT=1 TOP_K_BRIDGE=4 MAX_SOURCES=200 WAIT=1
python scripts/stage24_local_drift_diagnostic.py --eval-roots runs_stage24_reachability_confirm,runs_stage24_local_drift --out reports/stage24_local_drift.csv
python scripts/stage24_analyze.py
```
