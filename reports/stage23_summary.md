# Stage23 Summary

## 1. Official GAS Reproduction
- Matrix rows: 6; completed 4, skipped 2, failed 0.
```csv
env,seed,official_B_success,adapter_C_success,adapter_minus_official_pp
antmaze-medium-navigate-v0,0,0.9399999999999998,0.86,-7.999999999999985
antmaze-medium-stitch-v0,0,0.966,0.87,-9.599999999999998
```
- Adapter-vs-official max absolute gap: 9.6pp.
- Reproduction is HOLD until skipped official-pretrained routes are explained or full official training completes.
- Reproduction is HOLD because route C differs from route B by more than 3pp.
- Protocol repair route:
```csv
env,seed,official_B_success,adapter_original_success,adapter_official_control_success,official_control_minus_official_pp,episodes,mean_steps
antmaze-medium-navigate-v0,0,0.9399999999999998,0.86,0.93,-0.9999999999999788,100,295.6
antmaze-medium-stitch-v0,0,0.966,0.87,0.98,1.4000000000000012,100,252.14
```
- Repaired official-control adapter is within 1.4pp of official route B.
- Protocol audit rows: 2; non-ok rows: 0.

## 2. Failure Atlas
```csv
env,seed,variant,primary_failure_type,episodes,success,steps,subgoal_reach_rate,no_path_rate,risky_bridge_count
antmaze-medium-navigate-v0,0,gas_shortest,F4_LOCAL_EXECUTION_DRIFT,13,0.0,1000.0,0.1494797649383435,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,F6_LONG_PATH_ACCUMULATION,1,0.0,1000.0,0.4455958549222797,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,SUCCESS,86,1.0,284.4651162790698,0.68507181867679,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,F4_LOCAL_EXECUTION_DRIFT,13,0.0,1000.0,0.2370841461359818,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,SUCCESS,87,1.0,271.6896551724138,0.6489287954561059,0.0,0.0
```
- Main failure mode: F4_LOCAL_EXECUTION_DRIFT (13 grouped episodes).

## 3. Bridge And Oracle
- Bridge existence: PASS_BRIDGE_EXISTENCE (9 graph rows with shorter/bridge-using paths).
- Edge execution labels: rollout-backed; rows by type available.
```csv
env,seed,edge_type,edges,success_rate,set_state_rate
antmaze-large-explore-v0,0,aggressive_tdr_bridge,160,0.54375,1.0
antmaze-large-explore-v0,0,bottleneck_bridge,160,0.5625,1.0
antmaze-large-explore-v0,0,gas_cross,34,0.5,1.0
antmaze-large-explore-v0,0,safe_local,120,0.9916666666666668,1.0
```
- Oracle gate: NO_ORACLE_UPPER_BOUND.
```csv
env,seed,graph_id,node_count,edge_count,bridge_count,compared_paths,no_path_g0,no_path_graph,mean_path_cost_reduction,median_path_cost_reduction,shorter_path_rate,bridge_usage_rate
antmaze-large-explore-v0,0,G0,2514,34616,0,1000,0,0,0.0,0.0,0.0,0.0
antmaze-large-explore-v0,0,G3,2514,47693,13077,1000,0,0,5.110724902629852,3.9851980209350586,0.966,0.966
antmaze-large-explore-v0,0,G_oracle,2514,34793,177,1000,0,0,0.0229389200210571,0.0,0.089,0.089
```

## 4. p_bridge And Boundary
- p_bridge gate: PARTIAL_P_BRIDGE_HOLD_FP_REDUCTION.
```csv
env,seed,selected_bridge_AUROC,selected_bridge_AUPRC,selected_bridge_base_success_rate,accepted_bridge_success_rate@0.6,accepted_bridge_success_rate@0.7,false_positive_bridge_relative_reduction@0.6
antmaze-large-explore-v0,0,0.7124518613607189,0.7453022195335978,0.5189873417721519,0.6111111044883728,0.7083333134651184,0.1915204540679329
```
- Boundary gate: HOLD_BOUNDARY_DIAGNOSTIC_ONLY.
```csv
junction_count,psi_mean,psi_q50,supported_pair_rate,psi_AUROC_for_conditional_success,conditional_success_rate,supported_success_rate,unsupported_success_rate,supported_gap,coverage,env,seed
360277,0.5648943729408208,0.6,0.0,0.6542397660818715,0.3870967741935484,,0.3870967741935484,,0.0002581347130124,antmaze-large-explore-v0,0
```

## 5. Integrated BARS-v3
- PENDING: no integrated no-fallback eval rows.

## 6. Fallback Causal
- HOLD_FALLBACK: causal trigger-state ablation has not been run.

## 7. Current Decision
- NO_BARS_HEADROOM_ON_TESTED_ORACLE_ENV

## 8. Next Commands
```bash
bash scripts/stage23_pipeline.sh MODE=repro ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1
bash scripts/stage23_pipeline.sh MODE=bridge ENVS=antmaze-large-explore-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1
bash scripts/stage23_pipeline.sh MODE=edge_exec ENVS=antmaze-large-explore-v0 SEEDS=0 EDGE_EXEC_PILOT=1 GPUS=${GPUS:-0} WAIT=1
```
