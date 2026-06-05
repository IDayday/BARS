# CAGE-CLP0 Closed-Loop Probe Report

## Scope

Checkpoint root:
`/mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138`

Environments:

- `antmaze-giant-navigate-v0`, seed 42
- `antmaze-giant-stitch-v0`, seed 42
- `humanoidmaze-large-navigate-v0`, seed 44

No TDR retraining, keygraph reconstruction, low-level policy retraining, threshold tuning, learned reachability, risk-aware path search, or 8-env benchmark was run.

## Reset Capability

| env | reset works | set_state | qpos/qvel runtime | dataset qpos/qvel | obs equals qpos+qvel | reset mode |
| --- | --- | --- | --- | --- | --- | --- |
| antmaze-giant-navigate-v0 | yes | yes | yes | no | yes | dataset_state_ref |
| antmaze-giant-stitch-v0 | yes | yes | yes | no | yes | dataset_state_ref |
| humanoidmaze-large-navigate-v0 | yes | yes | yes | no | no | observation_only_not_exact |

Humanoid runtime can capture exact state, but existing dataset q_G rows cannot reconstruct exact `qpos/qvel`. Humanoid closed-loop probes from existing artifacts are therefore blocked.

Reset audit artifacts:

- `results/cage_clp0/reset_audit/reset_capability.json`
- `results/cage_clp0/reset_audit/reset_capability.md`

## q_G StateRef Coverage

| env | q_G rows | exact/probeable rows | initial planner rows | path-edge rows | main blocker |
| --- | ---: | ---: | ---: | ---: | --- |
| antmaze-giant-navigate-v0 | 2500 | 105 | 105 | 2395 | path-edge rows are keygraph phi nodes, not raw simulator states |
| antmaze-giant-stitch-v0 | 2500 | 101 | 101 | 2399 | path-edge rows are keygraph phi nodes, not raw simulator states |
| humanoidmaze-large-navigate-v0 | 2500 | 0 | 181 | 2319 | observation-only dataset rows are not exact |

Only initial planner targets from AntMaze are exact-reset probeable in CLP0. Keygraph path-edge nodes are phi cluster centers, not dataset simulator states.

## Closed-Loop Probe Results

Horizon: 32

| env | pair source | valid pairs | hit_rate | mean_delta_phi | mean_normalized_progress | negative_progress_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| antmaze-nav | q_G initial planner target | 105 | 1.0000 | 0.6528 | 0.0726 | 0.4381 |
| antmaze-nav | q_train matched control | 128 | 0.9922 | 0.5727 | 0.0895 | 0.2969 |
| antmaze-stitch | q_G initial planner target | 101 | 1.0000 | 0.7747 | 0.1129 | 0.3861 |
| antmaze-stitch | q_train matched control | 128 | 1.0000 | 0.4196 | 0.0294 | 0.3672 |
| humanoid-large-nav | q_G | 0 | NA | NA | NA | NA |

Analysis artifacts:

- `results/cage_clp0/probes/antmaze_nav_seed42_edge_probe.jsonl`
- `results/cage_clp0/probes/antmaze_stitch_seed42_edge_probe.jsonl`
- `results/cage_clp0/probes/humanoid_large_nav_seed44_edge_probe.jsonl`
- `results/cage_clp0/probes/antmaze_nav_seed42_qtrain_matched_probe.jsonl`
- `results/cage_clp0/probes/antmaze_stitch_seed42_qtrain_matched_probe.jsonl`
- `results/cage_clp0/analysis/closed_loop_probe_summary.md`
- `results/cage_clp0/datasets/closed_loop_contracts.jsonl`

## Answers

Is phi-space support sufficient?

- No. GP0 already showed coarse support does not explain failure. CLP0 shows AntMaze initial planner targets are executable, but this does not address non-probeable keygraph path edges or humanoid.

Is graph distance sufficient?

- No. In AntMaze, q_G initial planner targets and matched q_train controls have similar high hit rates despite different progress statistics. Distance alone is not a complete execution contract.

Does humanoid have lower R_pi at matched d_phi?

- Not answered in CLP0. Existing humanoid q_G artifacts do not contain exact reset state refs. Any answer would require approximate reset or inference from support, which CLP0 forbids.

Are CAGE recovery/final pairs low-contract?

- Not answered for recovery. Repair-0 traces were episode-level only and did not store exact recovery target state refs.
- Final-phase AntMaze initial-target rows were few and had hit_rate 1.0, but this is not enough to claim final-interface safety.

Does fixed commitment help because contracts need uninterrupted execution?

- Current evidence is consistent with that hypothesis but not conclusive. AntMaze initial targets are locally executable, while CAGE failures previously came from target switching, recovery retrying, and replanning churn. Humanoid needs exact rollout-state refs to verify this directly.

What should the execution contract model use?

- `phi_s`, `phi_g`, `d_phi`
- pair source and path position
- final/recovery flags
- q_train support when available
- closed-loop progress labels: hit, normalized progress, negative progress
- action norm / saturation indicators
- reset exactness metadata

## Contract Dataset

Output:

- `results/cage_clp0/datasets/closed_loop_contracts.jsonl`
- `results/cage_clp0/datasets/closed_loop_contracts_summary.md`

Current label rates over 462 valid AntMaze rows:

- hit: 0.9978
- contractive: 0.2035
- negative progress: 0.3680
- good contract: 0.9978

The `label_unstable` rate is high because the current default action-norm threshold is conservative for vector action norms. Treat it as a placeholder diagnostic threshold, not a calibrated instability classifier.

## Blockers

- Humanoid q_G rows are not exact-reset capable from existing dataset observations.
- Keygraph path-edge rows are phi nodes, not raw simulator states.
- Recovery target rows need step-level debug traces with exact `state_ref_s` captured during rollout.
- q_train support was not merged into the CLP0 probe rows, so support-bin probe analysis is currently unavailable.

## Next Command

Collect exact rollout StateRefs in a tiny debug trace before attempting humanoid CLP probes:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH WANDB_MODE=disabled /root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py --manifest_path results/cage_repair0/minipilot_humanoid_large_nav/manifests/minipilot_manifest.jsonl --max_jobs 1 --dry_run
```

The required follow-up patch is instrumentation-only: record `make_state_ref_from_env(...)` at selected target/recovery/final transitions when `--cage_debug` is enabled.
