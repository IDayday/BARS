# CAGE Pilot-0 Report

## 1. Repository Root And Working Directory

- Repository root: `/mnt/project/BARS`
- Working directory: `/mnt/project/BARS`
- Branch during this run: `codex/cage-mvp`
- Note: the working tree already contained unrelated experiment/status changes. They were not reverted.

## 2. Detected GAS Evaluator Path

- GAS evaluator: `external_src/GAS/evaluate_gas.py`
- Evaluator run cwd used by the manifest runner: `external_src/GAS`

## 3. Detected CAGE Package Path

- CAGE package: `external_src/GAS/cage/`

## 4. Detected Experiment Script Paths

- `scripts/build_cage_eval_command.py`
- `scripts/cage_experiment_manifest.py`
- `scripts/run_cage_manifest.py`
- `scripts/aggregate_cage_experiments.py`
- `scripts/plot_cage_diagnostics.py`
- `scripts/audit_cage_checkpoints.py`

## 5. Python Executable

- Local actual runtime: `/root/miniconda3/envs/gcrlo/bin/python`
- Local Python version: `3.10.20`
- Initial base environment audit used `/root/miniconda3/envs/navsim/bin/python`, but real smoke/minipilot was run in `gcrlo`.

## 6. Dependency Audit

Machine-readable audits:

- Base env: `results/cage_pilot0/preflight/dependency_audit.json`
- Local `gcrlo`: `results/cage_pilot0/preflight/dependency_audit_gcrlo.json`

Local `gcrlo` status:

- Available for OGBench AntMaze evaluation: `ogbench`, `gymnasium`, `gym`, `jax`, `mujoco`, `numpy`, `torch`, `matplotlib`, `pandas`, `pytest`
- Nonblocking legacy issue for this AntMaze pilot: `d4rl`/`mujoco_py` fail under the current legacy MuJoCo library path.

Remote `training-rl-zt3` status:

- `/root/miniconda3/envs/gcrlo/bin/python` exists.
- Blocking missing dependency: `ogbench`.
- Remote GAS evaluation was not continued after this failure because `external_src/GAS/O_utils/env_utils.py` imports `ogbench` at startup.
- Remote failure logs:
  - `results/cage_pilot0/minipilot_remote_antmaze_stitch/logs/antmaze-giant-stitch-v0__seed42__gas.stderr`
  - `results/cage_pilot0/minipilot_remote_antmaze_stitch/logs/antmaze-giant-stitch-v0__seed42__cage_fixed_commit.stderr`

## 7. Checkpoint Audit

Machine-readable audit:

- `results/cage_pilot0/preflight/checkpoint_audit.json`
- `results/cage_pilot0/preflight/checkpoint_audit.md`

Checkpoint root:

- `/mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138`

Runnable selected seeds:

| env_name | selected_seed | note |
| --- | --- | --- |
| `antmaze-giant-navigate-v0` | 42 | artifact seed, not official seed0 |
| `antmaze-giant-stitch-v0` | 42 | artifact seed, not official seed0 |
| `humanoidmaze-large-navigate-v0` | 44 | artifact seed, not official seed0; seed42/43 missing policy |

## 8. Manifest Commands

Smoke:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python scripts/cage_experiment_manifest.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_pilot0/smoke \
  --envs antmaze-giant-navigate-v0 \
  --seeds 42 \
  --variants gas cage_full \
  --episodes_per_goal 1 \
  --goals_per_env 1 \
  --eval_horizon default \
  --manifest_path results/cage_pilot0/smoke/manifests/smoke_manifest.jsonl \
  --strict_paths
```

Completed local navigate minipilot:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python scripts/cage_experiment_manifest.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_pilot0/minipilot_local_antmaze_nav \
  --envs antmaze-giant-navigate-v0 \
  --seeds 42 \
  --variants gas cage_fixed_commit cage_drift_only cage_recovery_only cage_full \
  --episodes_per_goal 5 \
  --goals_per_env 5 \
  --eval_horizon default \
  --manifest_path results/cage_pilot0/minipilot_local_antmaze_nav/manifests/minipilot_manifest.jsonl \
  --strict_paths
```

Completed local stitch minipilot:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python scripts/cage_experiment_manifest.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_pilot0/minipilot_local_antmaze_stitch \
  --envs antmaze-giant-stitch-v0 \
  --seeds 42 \
  --variants gas cage_fixed_commit cage_drift_only cage_recovery_only cage_full \
  --episodes_per_goal 5 \
  --goals_per_env 5 \
  --eval_horizon default \
  --manifest_path results/cage_pilot0/minipilot_local_antmaze_stitch/manifests/minipilot_manifest.jsonl \
  --strict_paths
```

Completed local humanoid minipilot:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python scripts/cage_experiment_manifest.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_pilot0/minipilot_local_humanoid_large_nav \
  --envs humanoidmaze-large-navigate-v0 \
  --seeds 44 \
  --variants gas cage_fixed_commit cage_drift_only cage_recovery_only cage_full \
  --episodes_per_goal 5 \
  --goals_per_env 5 \
  --eval_horizon default \
  --manifest_path results/cage_pilot0/minipilot_local_humanoid_large_nav/manifests/minipilot_manifest.jsonl \
  --strict_paths
```

## 9. Run Commands

Smoke and navigate minipilot used:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH WANDB_MODE=disabled \
  /root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py \
  --manifest_path MANIFEST.jsonl \
  --max_jobs N
```

Completed stitch command:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH WANDB_MODE=disabled \
  /root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py \
  --manifest_path results/cage_pilot0/minipilot_local_antmaze_stitch/manifests/minipilot_manifest.jsonl \
  --max_jobs 5
```

Completed humanoid command:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH WANDB_MODE=disabled \
  /root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py \
  --manifest_path results/cage_pilot0/minipilot_local_humanoid_large_nav/manifests/minipilot_manifest.jsonl \
  --max_jobs 5
```

## 10. Aggregation Commands

Completed local navigate minipilot:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python scripts/aggregate_cage_experiments.py \
  --input_root results/cage_pilot0/minipilot_local_antmaze_nav \
  --manifest_path results/cage_pilot0/minipilot_local_antmaze_nav/manifests/minipilot_manifest.jsonl \
  --out_csv results/cage_pilot0/minipilot_local_antmaze_nav/tables/minipilot_summary.csv \
  --out_md results/cage_pilot0/minipilot_local_antmaze_nav/tables/minipilot_summary.md \
  --out_json results/cage_pilot0/minipilot_local_antmaze_nav/tables/minipilot_summary.json
```

Plot command:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python scripts/plot_cage_diagnostics.py \
  --input results/cage_pilot0/minipilot_local_antmaze_nav/tables/minipilot_summary.csv \
  --output_dir results/cage_pilot0/minipilot_local_antmaze_nav/plots
```

## 11. Results Summary

One-pair smoke, `antmaze-giant-navigate-v0 seed42`, 1 episode x 1 task:

| variant | success | normalized_score | target_switch_count | stall_count | recovery_attempt_count | segment_target_reach_rate |
| --- | --- | --- | --- | --- | --- | --- |
| gas | 1.00 | 100.00 | NA | NA | NA | NA |
| cage_full | 0.00 | 0.00 | 76 | 13 | 2 | 0.00 |

Completed local navigate minipilot, `antmaze-giant-navigate-v0 seed42`, 5 episodes x 5 tasks:

| variant | success_rate | normalized_score | target_switch_count | stall_count | drift_count | recovery_attempt_count | recovery_success_rate | global_replan_request_count | segment_target_reach_rate | final_goal_on_rate | final_goal_stall_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gas | 0.60 | 60.00 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| cage_fixed_commit | 0.00 | 0.00 | 68.24 | 37.80 | 0.00 | 0.00 | NA | 0.00 | 0.00 | 0.00 | 0.00 |
| cage_drift_only | 0.00 | 0.00 | 68.24 | 37.80 | 0.04 | 0.00 | NA | 0.04 | 0.00 | 0.00 | 0.00 |
| cage_recovery_only | 0.00 | 0.00 | 61.80 | 20.84 | 0.12 | 2.00 | 0.04 | 567.64 | 0.0014 | 0.00 | 0.00 |
| cage_full | 0.36 | 36.00 | 59.72 | 8.68 | 0.00 | 2.00 | 0.02 | 481.20 | 0.0110 | 0.60 | 15.76 |

Paired deltas vs GAS:

| variant | delta_success_rate | delta_normalized_score |
| --- | --- | --- |
| cage_fixed_commit | -0.60 | -60.00 |
| cage_drift_only | -0.60 | -60.00 |
| cage_recovery_only | -0.60 | -60.00 |
| cage_full | -0.24 | -24.00 |

Interpretation:

- CAGE-MVP did not improve Pilot-0 navigate success in this seed; it regressed relative to baseline GAS.

## Repair-0 Follow-Up

Pilot-0 motivated two additional opt-in variants for diagnosis:

- `cage_trace_only`: constructs CAGE and writes traces while passing through the original GAS subgoal exactly.
- `cage_safe_full`: keeps `cage_full` available unchanged, but adds explicit churn guardrails for diagnostic runs.

The Repair-0 report is in `docs/cage_repair0_report.md`. The short conclusion is that trace-only parity passed for antmaze navigate and humanoid, and safe full prevented replan storms, but safe full did not restore humanoid success. Therefore Pilot-0 should not be expanded to the full 8-env benchmark until a preregistered commitment-first CAGE-v0.2 variant is implemented.
- CAGE full reduced stall versus fixed commitment and drift-only ablations, but recovery success remained very low and global replan requests were very high.
- The main current failure signature is execution-interface oscillation/recovery churn: high target switches, low segment target reach rate, low recovery success, and hundreds of global replan requests.
- This is a small artifact-seed pilot and is not a final benchmark claim.

Completed local stitch minipilot, `antmaze-giant-stitch-v0 seed42`, 5 episodes x 5 tasks:

| variant | success_rate | normalized_score | target_switch_count | stall_count | drift_count | recovery_attempt_count | recovery_success_rate | global_replan_request_count | segment_target_reach_rate | final_goal_on_rate | final_goal_stall_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gas | 0.84 | 84.00 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| cage_fixed_commit | 0.12 | 12.00 | 63.12 | 30.88 | 0.00 | 0.00 | NA | 0.00 | 0.0018 | 0.00 | 0.00 |
| cage_drift_only | 0.12 | 12.00 | 63.12 | 30.88 | 0.00 | 0.00 | NA | 0.00 | 0.0018 | 0.00 | 0.00 |
| cage_recovery_only | 0.12 | 12.00 | 59.08 | 21.32 | 0.00 | 2.00 | 0.00 | 493.44 | 0.0009 | 0.00 | 0.00 |
| cage_full | 0.76 | 76.00 | 62.12 | 6.00 | 0.00 | 1.96 | 0.0204 | 249.92 | 0.0060 | 0.76 | 1.76 |

Stitch interpretation:

- CAGE full is still below GAS on success, but the gap is smaller than navigate: -0.08 success and -8 normalized score.
- Fixed commitment, drift-only, and recovery-only remain poor; full CAGE is materially better than its ablations.
- The same recovery weakness appears: recovery success is about 2%, segment target reach remains very low, and full CAGE still requests many global replans.

Artifacts:

- Smoke summary: `results/cage_pilot0/smoke/tables/smoke_summary.md`
- Navigate minipilot summary: `results/cage_pilot0/minipilot_local_antmaze_nav/tables/minipilot_summary.md`
- Navigate minipilot plots: `results/cage_pilot0/minipilot_local_antmaze_nav/plots/`
- Stitch minipilot summary: `results/cage_pilot0/minipilot_local_antmaze_stitch/tables/minipilot_summary.md`
- Stitch minipilot plots: `results/cage_pilot0/minipilot_local_antmaze_stitch/plots/`
- Humanoid minipilot summary: `results/cage_pilot0/minipilot_local_humanoid_large_nav/tables/minipilot_summary.md`
- Humanoid minipilot plots: `results/cage_pilot0/minipilot_local_humanoid_large_nav/plots/`

Completed local humanoid minipilot, `humanoidmaze-large-navigate-v0 seed44`, 5 episodes x 5 tasks:

| variant | success_rate | normalized_score | target_switch_count | stall_count | drift_count | recovery_attempt_count | recovery_success_rate | global_replan_request_count | segment_target_reach_rate | final_goal_on_rate | final_goal_stall_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gas | 0.20 | 20.00 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| cage_fixed_commit | 0.32 | 32.00 | 77.36 | 48.60 | 0.00 | 0.00 | NA | 0.00 | 0.00 | 0.00 | 0.00 |
| cage_drift_only | 0.04 | 4.00 | 9.48 | 8.24 | 15.68 | 0.00 | NA | 3738.48 | 0.00 | 0.00 | 0.00 |
| cage_recovery_only | 0.00 | 0.00 | 6.72 | 8.12 | 16.00 | 1.84 | 0.00 | 3802.00 | 0.00 | 0.00 | 0.00 |
| cage_full | 0.04 | 4.00 | 7.36 | 6.04 | 16.24 | 1.84 | 0.00 | 3815.52 | 0.00 | 0.04 | 7.88 |

Humanoid interpretation:

- Fixed commitment only improved success over GAS in this single artifact-seed pilot: +0.12 success and +12 normalized score.
- Drift/recovery/full CAGE regressed sharply. The trace signature is high path drift, zero segment target reach, zero recovery success, and thousands of global replan requests.
- This supports the current failure diagnosis: the CAGE full controller is too eager to classify path drift/recovery failure in humanoid and can enter replan churn.

## 12. Blockers

- Remote `training-rl-zt3` `gcrlo` cannot run this GAS evaluator until `ogbench` is available.
- Official seed0 checkpoints are not present in the audited artifact root; Pilot-0 used artifact seeds 42 and 44.
- GAS baseline currently has no CAGE trace fields, so paired deltas for target switching/stall/drift are unavailable unless baseline tracing is added in a future infrastructure-only milestone.
- `cage_reachability` and `cage_risk_path` remain unsupported as expected.

## 13. Static Validation Commands And Return Codes

All validation commands returned `0` in local `gcrlo`:

```bash
python -m py_compile scripts/cage_experiment_manifest.py scripts/run_cage_manifest.py scripts/aggregate_cage_experiments.py scripts/plot_cage_diagnostics.py scripts/audit_cage_checkpoints.py
cd external_src/GAS && python -m py_compile evaluate_gas.py cage/*.py
pytest tests/test_cage_experiment_manifest.py tests/test_cage_aggregation.py tests/test_cage_state_machine.py tests/test_cage_subgoal_selector.py -q
```

Test result:

- `13 passed`

## 14. GPU Utilization Guard

Local:

- GPUs 0 and 1 were kept at 100% during the run.

Remote:

- GPUs 0-7 were kept above 50% after launching pressure jobs for idle devices.
- Added remote pressure for GPU3 after it briefly dropped to 0%.

## 15. Exact Next Recommended Command

After `training-rl-zt3` has `ogbench` available in `gcrlo`, run this remote dependency gate first:

```bash
ssh training-rl-zt3 'cd /mnt/project/BARS && PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python -c "import ogbench, gymnasium; print(\"remote_gcrlo_ready\")"'
```

Then start the remote focused minipilot:

```bash
ssh training-rl-zt3 'cd /mnt/project/BARS && export PATH=/root/miniconda3/envs/gcrlo/bin:$PATH WANDB_MODE=disabled; /root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py --manifest_path results/cage_pilot0/minipilot_remote_antmaze_stitch/manifests/minipilot_manifest.jsonl --max_jobs 5'
```
