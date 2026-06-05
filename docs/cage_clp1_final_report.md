# CAGE-CLP1 Final Report

## Repo Review

- Repo root: `/mnt/project/BARS`
- Branch: `codex/cage-mvp`
- Current base commit during this milestone: `1574580`
- GAS evaluator: `external_src/GAS/evaluate_gas.py`
- CAGE package: `external_src/GAS/cage/`
- GP0/CLP0/Repair0 reports were detected; see `docs/cage_clp1_repo_review.md`.

## StateRef Capability

Audit output:
- `results/cage_clp1/state_ref_audit/state_ref_capability.json`
- `results/cage_clp1/state_ref_audit/state_ref_capability.md`

Exact MuJoCo restore passed with max observation error `0` for:
- `antmaze-giant-navigate-v0`
- `antmaze-giant-stitch-v0`
- `humanoidmaze-large-navigate-v0`

## Segment Capture

Main 2x2 capture:
- output root: `results/cage_clp1/segment_capture/`
- envs/seeds: `antmaze-giant-navigate-v0 seed42`, `humanoidmaze-large-navigate-v0 seed44`
- variants: `gas`, `cage_trace_only`, `cage_fixed_commit`, `cage_safe_full`
- status: 8/8 jobs succeeded

All segment starts had exact StateRefs. Segment counts:

| env | variant | segments | exact |
| --- | --- | ---: | ---: |
| antmaze | gas | 3392 | 3392 |
| antmaze | cage_trace_only | 3438 | 3438 |
| antmaze | cage_fixed_commit | 288 | 288 |
| antmaze | cage_safe_full | 1515 | 1515 |
| humanoid | gas | 609 | 609 |
| humanoid | cage_trace_only | 609 | 609 |
| humanoid | cage_fixed_commit | 432 | 432 |
| humanoid | cage_safe_full | 605 | 605 |

Candidate-aware 1x1 capture:
- output root: `results/cage_clp1/segment_capture_candidate/`
- adds `path_phi` and `final_goal_phi` for nearest/farther/final branchable probes
- status: 8/8 jobs succeeded

## Branchable Probe

Main probes:
- `results/cage_clp1/probes/`
- 8 files, 64 segment starts per trace, horizons 16/32/64
- 9216 probe rows

Candidate-aware probes:
- `results/cage_clp1/probes_candidate/`
- 8 files, 32 segment starts per trace, horizons 16/32/64
- 4608 probe rows

## Contract Oracle

Candidate-aware oracle:
- `results/cage_clp1/oracle_candidate/contract_oracle_summary.md`
- `num_segments`: 768
- original hit rate: 0.3880
- oracle hit rate: 0.3880
- original progress mean: 0.0100
- oracle progress mean: 0.0352

At horizon 64:
- AntMaze original target hit 0.6953; nearest path target hit 0.6094; farther target hit 0.0312; final goal hit 0.
- Humanoid original target hit 0.0859; nearest path target hit 0.0859; farther/final hit 0.

Interpretation: available inference-time target alternatives did not improve hit rate in this small sample. Humanoid failures look like low closed-loop policy contractibility, not just wrong subgoal selection.

## Contract Dataset

Main dataset:
- `results/cage_clp1/datasets/closed_loop_contracts.jsonl`
- rows: 9216
- contract-positive rate: 0.0750
- contract-negative rate: 0.0576
- policy-weak rate: 0.3438

Candidate-aware dataset:
- `results/cage_clp1/datasets_candidate/closed_loop_contracts.jsonl`
- rows: 4608
- contract-positive rate: 0.1280
- contract-negative rate: 0.2528
- policy-weak rate: 0.1094

## Contract Model

Sanity-check linear contract model:
- main metrics: `results/cage_clp1/models/contract_model_eval_metrics.json`
- candidate metrics: `results/cage_clp1/models_candidate/contract_model_eval_metrics.json`

Candidate set same-data metrics:
- hit AUROC: 0.9991
- contract-positive AUROC: 0.9873
- negative-progress AUROC: 0.7869

d_phi-only baseline on candidate set:
- hit AUROC: 0.9990
- contract-positive AUROC: 0.9856
- negative-progress AUROC: 0.5956

This is not a held-out result. The only meaningful signal is that negative-progress prediction benefits from richer phi-pair features; hit/positive labels remain largely distance-explained in this small sample.

## Policy Alignment Feasibility

Candidate hard-goal dataset:
- `results/cage_clp1/policy_alignment_candidate/hard_goal_dataset.jsonl`
- examples: 2501
- hard_positive: 0
- hard_unlabeled: 1336
- hard_negative: 1165
- available action supervision rate: 0

Policy finetuning is blocked for CLP1: there are no verified hard-positive supervised examples in the generated hard-goal set. These hard goals are useful for contract/ranking/conservative objectives, not naive behavior cloning.

## Decisions

- Trace-only parity: acceptable in humanoid segment count; AntMaze differs slightly in segment count but success is the same in the tiny capture.
- Humanoid has far lower `R_pi` than AntMaze under the same branchable probe protocol.
- Final-goal targets are low-contract in both envs at this budget.
- Farther path targets are low-contract, especially in humanoid.
- Recovery candidates are too sparse for a stable CLP1 conclusion.
- q_train matched controls are still missing and should be the next instrumentation target.

Next milestone should be CLP1.1: record q_train matched branchable targets and path candidate metadata more deliberately, then run held-out contract-model evaluation. Do not implement risk-aware graph search yet.

## Validation

Passed:
- `python -m py_compile` for CLP1 scripts, `evaluate_gas.py`, `O_utils/evaluation.py`, and CAGE modules.
- `pytest tests/test_cage_state_ref.py tests/test_closed_loop_contract_dataset.py tests/test_graph_induced_policy_dataset.py -q`: 9 passed.
- `pytest tests/test_cage_experiment_manifest.py tests/test_cage_aggregation.py -q`: 4 passed.
- `pytest tests/test_cage_state_machine.py tests/test_cage_subgoal_selector.py -q`: 9 passed.
- `pytest tests/test_cage_churn_guard.py tests/test_cage_trace_only.py -q`: 8 passed.

## Blockers

- q_train matched target mode is not yet implemented for branchable probes.
- Path candidate capture currently stores phi path snapshots, not exact target StateRefs.
- Contract model metrics are same-data sanity checks, not held-out model evidence.
- No supervised hard-positive policy-alignment examples were found, so policy finetuning was not run.

## Next Command

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/run_contract_capture_smoke.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_clp1/segment_capture_candidate \
  --env_seed_pairs antmaze-giant-navigate-v0:42 humanoidmaze-large-navigate-v0:44 \
  --variants gas cage_trace_only cage_fixed_commit cage_safe_full \
  --episodes_per_goal 2 \
  --goals_per_env 2 \
  --status_path results/cage_clp1/segment_capture_candidate/status_2x2.jsonl
```
