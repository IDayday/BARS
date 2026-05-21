# Stage24 Execution Status

Generated: 2026-05-19 17:40 CST

## Summary

Stage24 was executed as far as the available artifacts allow.

- Oracle-headroom scan on `scene-play-v0/seed0`: completed.
- Reachability confirmation on medium OGBench seeds 0/1/2: blocked by missing GAS artifacts.
- Local-drift repair variants on medium OGBench seeds 0/1/2: blocked by the same missing GAS artifacts.
- Integrated BARS-v3: correctly skipped because oracle headroom did not pass and p_bridge is gated behind oracle headroom.

## Oracle Scan Result

`scene-play-v0/seed0` completed graph construction, edge execution pilot, oracle graph construction, and Stage24 gate analysis.

Key row from `reports/stage24_oracle_headroom.csv`:

| env | seed | bridge_count | shorter_path_rate | oracle_bridge_count | oracle_shorter_path_rate | oracle_mean_path_cost_reduction | safe_local_success_rate | risky_bridge_success_rate | set_state_rate | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scene-play-v0 | 0 | 812 | 0.041 | 274 | 0.017 | 0.0915 | 1.0 | 0.9013 | 0.0 | NO_ORACLE_UPPER_BOUND |

Interpretation: the non-oracle G3 graph has weak path-level improvement, and the oracle-success bridge graph does not create enough path-level headroom. Because `set_state_rate=0.0`, edge labels are not strong reset labels for this family. The Stage24 oracle gate remains `NO_ORACLE_UPPER_BOUND`.

## Medium Reachability Confirmation

The required Stage24 reachability command was run in no-training/protocol-preserving mode:

```bash
PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD \
bash scripts/stage24_run_reachability_confirm.sh \
  CONFIG=configs/stage24_reachability_confirm.json \
  ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 \
  SEEDS=0,1,2 EPISODES=100 GPUS=1,2,3,4 \
  MAX_PARALLEL_EVAL=4 WAIT=1 \
  PREFER_PRETRAINED=1 TRAIN_IF_MISSING=0 QUICK=0 EVAL_ON_CPU=1
```

Result:

- `runs_stage24_reachability_confirm_logs/failed_jobs.csv`: 18 failed eval jobs.
- Failure class: `checkpoint/artifact missing`.
- Direct evaluator error: `RuntimeError: Missing GAS keygraph`.

Artifact check:

- Local `artifacts/gas` has no `antmaze-medium-navigate-v0` or `antmaze-medium-stitch-v0` GAS keygraph/policy.
- Official HF probe found no medium `keygraph.pkl`, `params_1000000.pkl`, or `params_500000.pkl` under the tested GAS paths.

Decision: do not claim PASS_REACHABILITY_CONFIRM. Current reachability rows in `reports/stage24_reachability_confirm.csv` are Stage23 seed0 prior evidence only.

## Local Drift Repair

The Stage24 repair variants were run in the same no-training/protocol-preserving mode:

```bash
PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD \
bash scripts/stage24_run_reachability_confirm.sh \
  CONFIG=configs/stage24_reachability_confirm.json \
  ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 \
  SEEDS=0,1,2 EPISODES=100 \
  VARIANTS=gas_shortest_replan_on_local_drift,gas_shortest_adaptive_subgoal_horizon,gas_reachability_budget_replan_on_local_drift \
  STAGE24_ROOT=runs_stage24_local_drift \
  LOG_ROOT=runs_stage24_local_drift_logs \
  GPUS=1,2,3,4 MAX_PARALLEL_EVAL=4 WAIT=1 \
  PREFER_PRETRAINED=1 TRAIN_IF_MISSING=0 QUICK=0 EVAL_ON_CPU=1
```

Result:

- `runs_stage24_local_drift_logs/failed_jobs.csv`: 18 failed eval jobs.
- Failure class: `checkpoint/artifact missing`.
- No repair eval rows were generated, so `PASS_LOCAL_DRIFT_REPAIR` cannot be evaluated.

## Final Gates

From `reports/stage24_gate_status.json`:

- reachability_confirm: HOLD_REACHABILITY_WEAK_EFFECT
- local_drift_repair: HOLD_LOCAL_DRIFT_REPAIR
- oracle_headroom: NO_ORACLE_UPPER_BOUND
- p_bridge: SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM
- boundary: HOLD_BOUNDARY_DIAGNOSTIC_ONLY
- integrated: SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE
- d4rl_protocol: HOLD_D4RL_PROTOCOL_REPAIR

## Next Required Input

To complete medium reachability and local-drift repair experimentally, provide or train protocol-aligned GAS artifacts for:

- `antmaze-medium-navigate-v0/seed0`
- `antmaze-medium-stitch-v0/seed0`

With `PREFER_PRETRAINED=1`, the current loader can copy seed0 artifacts to seeds 1/2 for same-backbone seed-level evaluation. Without those artifacts, Stage24 cannot honestly complete the medium confirmation or repair gates.
