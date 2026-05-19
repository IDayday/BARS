# Stage23 Summary

## 1. Official GAS Reproduction
- Matrix rows: 6; completed 4, skipped 2, failed 0.
```csv
env,seed,official_B_success,adapter_C_success,adapter_minus_official_pp
antmaze-medium-navigate-v0,0,0.9399999999999998,0.86,-7.999999999999985
antmaze-medium-stitch-v0,0,0.966,0.87,-9.599999999999998
```
- Adapter-vs-official max absolute gap: 9.6pp.
- Raw three-route matrix has skipped official-pretrained rows; the repaired control route is used for the current gate.
- Original route C differs from route B by more than 3pp, so adapter conclusions use the repaired official-control path.
- Protocol repair route:
```csv
env,seed,official_B_success,adapter_original_success,adapter_official_control_success,official_control_minus_official_pp,episodes,mean_steps
antmaze-medium-navigate-v0,0,0.9399999999999998,0.86,0.93,-0.9999999999999788,100,295.6
antmaze-medium-stitch-v0,0,0.966,0.87,0.98,1.4000000000000012,100,252.14
```
- Repaired official-control adapter is within 1.4pp of official route B.
- Reproduction gate: GO_REPRO_REPAIRED.
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
- Bridge existence: PASS_BRIDGE_EXISTENCE (15 graph rows with shorter/bridge-using paths).
- Edge execution labels: rollout-backed; rows by type available.
```csv
env,seed,edge_type,edges,success_rate,set_state_rate
antmaze-giant-navigate-v0,0,aggressive_tdr_bridge,40,0.25,1.0
antmaze-giant-navigate-v0,0,bottleneck_bridge,40,0.025,1.0
antmaze-giant-navigate-v0,0,gas_cross,40,0.15,1.0
antmaze-giant-navigate-v0,0,safe_local,30,0.9666666666666668,1.0
antmaze-giant-stitch-v0,0,aggressive_tdr_bridge,40,0.575,1.0
antmaze-giant-stitch-v0,0,bottleneck_bridge,40,0.275,1.0
antmaze-giant-stitch-v0,0,gas_cross,40,0.225,1.0
antmaze-giant-stitch-v0,0,safe_local,30,1.0,1.0
antmaze-large-explore-v0,0,aggressive_tdr_bridge,160,0.54375,1.0
antmaze-large-explore-v0,0,bottleneck_bridge,160,0.5625,1.0
antmaze-large-explore-v0,0,gas_cross,34,0.5,1.0
antmaze-large-explore-v0,0,safe_local,120,0.9916666666666668,1.0
```
- Oracle gate: NO_ORACLE_UPPER_BOUND.
```csv
env,seed,graph_id,node_count,edge_count,bridge_count,compared_paths,no_path_g0,no_path_graph,mean_path_cost_reduction,median_path_cost_reduction,shorter_path_rate,bridge_usage_rate
antmaze-giant-navigate-v0,0,G0,923,6608,0,1000,0,0,0.0,0.0,0.0,0.0
antmaze-giant-navigate-v0,0,G3,923,12914,6306,1000,0,0,10.6426924793479,9.274191887967149,0.986,0.986
antmaze-giant-navigate-v0,0,G_oracle,923,6619,11,1000,0,0,0.0016052114048783,0.0,0.005,0.005
antmaze-giant-stitch-v0,0,G0,1937,31986,0,1000,0,0,0.0,0.0,0.0,0.0
antmaze-giant-stitch-v0,0,G3,1937,43052,11066,1000,0,0,12.747188004191027,5.8131488521374735,0.975,0.975
antmaze-giant-stitch-v0,0,G_oracle,1937,32020,34,1000,0,0,0.0026380278536111,0.0,0.007,0.007
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
118179,0.5211006185532113,0.6,0.0,0.95,0.0625,,0.0625,,0.0002707756877279,antmaze-giant-navigate-v0,0
338935,0.5702524082788738,0.6,0.0,,0.0,,0.0,,5.015710977030994e-05,antmaze-giant-stitch-v0,0
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
# Do not run integrated BARS-v3 on the tested antmaze hard envs until a new oracle upper bound appears.
PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD bash scripts/stage22_prepare_gas_backbone.sh ENVS=scene-play-v0 SEEDS=0 GPUS=${GPUS:-0} ARTIFACT_ROOT=artifacts/gas PREFER_PRETRAINED=1 TRAIN_IF_MISSING=0 LOG_ROOT=runs_stage23_prepare_scene
PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD bash scripts/stage23_pipeline.sh MODE=bridge ENVS=scene-play-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1
PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD bash scripts/stage23_pipeline.sh MODE=edge_exec ENVS=scene-play-v0 SEEDS=0 EDGE_EXEC_PILOT=1 GPUS=${GPUS:-0} WAIT=1
PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD bash scripts/stage23_pipeline.sh MODE=oracle ENVS=scene-play-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1
```
