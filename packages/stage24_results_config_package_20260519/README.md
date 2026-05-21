# Stage24 Results/Config Package

Generated: 2026-05-19

## Completion Status

Stage24 implementation and report scaffolding are complete. The full Stage24 experimental tasks are not complete yet.

Current gates from `reports/stage24_gate_status.json`:

- reachability_confirm: HOLD_REACHABILITY_WEAK_EFFECT
- local_drift_repair: HOLD_LOCAL_DRIFT_REPAIR
- oracle_headroom: PENDING_ORACLE_HEADROOM_SCAN
- p_bridge: SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM
- boundary: HOLD_BOUNDARY_DIAGNOSTIC_ONLY
- integrated: SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE
- d4rl_protocol: HOLD_D4RL_PROTOCOL_REPAIR

Interpretation: current reports contain Stage24 wiring plus available prior/current evidence. Seeds 1/2 reachability confirmation and the new `scene-play-v0` oracle headroom scan still need to run before any GO claim.

## Contents

- `configs/`: Stage24 reachability and oracle scan configs.
- `scripts/`: Stage24 launchers, analyzer, and local drift diagnostic.
- `reports/`: Generated Stage24 gate/status/evidence reports.
- `code/`: Touched planner/evaluator files plus `stage24_code_diff.patch`.
- `MANIFEST.txt`: File list for this package.

## Main Next Commands

```bash
bash scripts/stage24_run_reachability_confirm.sh CONFIG=configs/stage24_reachability_confirm.json ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0,1,2 EPISODES=100 GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1

bash scripts/stage24_run_reachability_confirm.sh CONFIG=configs/stage24_reachability_confirm.json ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0,1,2 EPISODES=100 VARIANTS=gas_shortest_replan_on_local_drift,gas_shortest_adaptive_subgoal_horizon,gas_reachability_budget_replan_on_local_drift STAGE24_ROOT=runs_stage24_local_drift LOG_ROOT=runs_stage24_local_drift_logs GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1

bash scripts/stage24_oracle_headroom_scan.sh ENVS=scene-play-v0 SEEDS=0 GPUS=${GPUS:-0} EDGE_EXEC_PILOT=1 TOP_K_BRIDGE=4 MAX_SOURCES=200 WAIT=1

python scripts/stage24_analyze.py
```
