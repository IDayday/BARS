# Stage24 Decisions

## Gates
- reachability_confirm: HOLD_REACHABILITY_WEAK_EFFECT
- local_drift_repair: HOLD_LOCAL_DRIFT_REPAIR
- oracle_headroom: NO_ORACLE_UPPER_BOUND
- p_bridge: SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM
- boundary: HOLD_BOUNDARY_DIAGNOSTIC_ONLY
- integrated: SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE
- d4rl_protocol: HOLD_D4RL_PROTOCOL_REPAIR

## Evidence

Reachability confirmation:
```csv
env,seed,variant,success,steps,success_delta_vs_shortest,steps_inflation_vs_shortest,source
antmaze-medium-navigate-v0,0,gas_reachability_budget_calibrated,0.92,345.04,0.02,-0.04078285285368758,stage24_run
antmaze-medium-navigate-v0,0,gas_reachability_soft_calibrated,0.86,367.22,-0.04,0.02087792944316268,stage24_run
antmaze-medium-navigate-v0,0,gas_shortest,0.9,359.71,,,stage24_run
antmaze-medium-navigate-v0,1,gas_reachability_budget_calibrated,0.89,356.21,-0.02,0.0630912943564031,stage24_run
antmaze-medium-navigate-v0,1,gas_reachability_soft_calibrated,0.88,351.73,-0.03,0.049720953830542945,stage24_run
antmaze-medium-navigate-v0,1,gas_shortest,0.91,335.07,,,stage24_run
antmaze-medium-navigate-v0,2,gas_reachability_budget_calibrated,0.91,329.94,-0.03,0.06490656166284742,stage24_run
antmaze-medium-navigate-v0,2,gas_reachability_soft_calibrated,0.94,308.08,0.0,-0.005648258722525256,stage24_run
```

Oracle headroom:
```csv
env,seed,bridge_count,shorter_path_rate,oracle_bridge_count,oracle_shorter_path_rate,oracle_mean_path_cost_reduction,safe_local_success_rate,set_state_rate,gate
scene-play-v0,0,817,0.041,268,0.014,0.1033805274963378,1.0,0.0,NO_ORACLE_UPPER_BOUND
```

Local drift:
```csv
env,seed,variant,success,primary_failure_type,local_drift_score,progress_stall_count,oscillation_score
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,1.3441482743413848,14,0.9740472975297862
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,4.884295541353798,15,0.9780819407421388
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,3.7025130613469055,17,0.9798300958698868
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.9186709760899344,24,0.9580189855892797
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.9622415051328664,28,0.9644607750484668
antmaze-medium-navigate-v0,0,gas_shortest,1,F6_LONG_PATH_ACCUMULATION,2.5962784464150257,34,0.9850238106270313
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,2.9637963257232576,20,0.9862467864208644
antmaze-medium-navigate-v0,0,gas_shortest,1,SUCCESS,0.8237777649182564,37,0.9814406454181821
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
