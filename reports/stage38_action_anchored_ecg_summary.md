# Stage38 Action-Anchored ECG Summary

- branch: `codex/cage-mvp`
- starting commit: `d8663ff`
- objective: build the first trainable action-anchored ECG loop from raw/offline trajectories, train contract and policy-adapter models, build a trusted action-anchored graph, connect evaluator variants, and run minimal AntMaze online validation.

## Code Changes

新增/修改的核心文件：

- `scripts/build_action_anchored_contract_dataset.py`
- `scripts/train_action_anchored_ecg_contract.py`
- `scripts/train_ecg_policy_adapter.py`
- `scripts/build_action_anchored_ecg_graph_v2.py`
- `scripts/train_ecg_planner_score.py`
- `external_src/GAS/cage/ecg_planner_runtime.py`
- `external_src/GAS/cage/ecg_policy_adapter.py`
- `external_src/GAS/cage/config.py`
- `external_src/GAS/cage/contract_graph.py`
- `external_src/GAS/cage/contract_model.py`
- `external_src/GAS/evaluate_gas.py`
- `external_src/GAS/O_utils/evaluation.py`
- `scripts/build_cage_eval_command.py`
- `scripts/cage_experiment_manifest.py`
- `tests/test_action_anchored_contract_dataset.py`
- `tests/test_action_anchored_contract_model.py`
- `tests/test_ecg_policy_adapter.py`
- `tests/test_action_anchored_ecg_graph_v2.py`
- `tests/test_ecg_planner_score.py`
- `tests/test_ecg_runtime_variants.py`

## Validation

- `python -m py_compile ...`: PASS
- `pytest tests/test_action_anchored_contract_dataset.py tests/test_action_anchored_contract_model.py tests/test_ecg_policy_adapter.py tests/test_action_anchored_ecg_graph_v2.py tests/test_ecg_planner_score.py tests/test_ecg_runtime_variants.py tests/test_contract_graph.py tests/test_contract_planner.py -q`: PASS, 11 passed.

GPU utilization was kept high with `/mnt/project/gpu_stress.py`; active stress PID during the run was `1477661`.

## Action-Anchored Dataset

- output: `results/cage_ecg/action_anchored_dataset/action_contracts.jsonl`
- examples: 600000
- action_supervision_rate: 1.0
- positive_with_action_count: 322888
- final_goal_with_action_count: 550
- phi source: Stage27 offline trajectory derivatives with `tdr_emb`
- KNN/action matching route: not used as the main route.

## Contract Model

- output: `results/cage_ecg/action_anchored_models/contract/model.pt`
- status: `CONTRACT_MODEL_READY`
- contract-positive AUROC test: 0.9951 vs d_phi baseline 0.9841
- negative-contract AUROC test: 0.9951 vs d_phi baseline 0.9841
- final-goal label is very sparse; final-goal metrics should not be treated as strong evidence.

## Policy Adapter

- output: `results/cage_ecg/action_anchored_models/policy_adapter/model.pt`
- status: `POLICY_ADAPTER_READY`
- validation MSE: 0.2426
- mean-action baseline validation MSE: 0.4765
- offline BC gate: PASS
- online adapter safety: FAIL

## Action-Anchored ECG Graph

- output: `results/cage_ecg/action_anchored_graph_v2/contract_graph.json`
- nodes: 42502
- edges: 100000
- action_anchored_edge_rate: 1.0
- final_goal_edge_count: 50
- unverified KNN main edge count: 0

## Planner Score

- output: `results/cage_ecg/action_anchored_models/planner_score/weights.json`
- status: `PLANNER_SCORE_READY`
- learned path min-contract: 0.8990 vs shortest 0.8707
- learned path negative risk: 0.1272 vs shortest 0.0741
- gate: PARTIAL. It improves min contract but increases negative risk.

## Online Smoke

All 6 smoke jobs returned code 0. See `reports/stage38_action_anchored_ecg_smoke.md`.

## AntMaze Minipilot

| env | gas | ecg_trace_only | ecg_planner | ecg_adapter |
| --- | ---: | ---: | ---: | ---: |
| antmaze-giant-navigate-v0 | 0.60 | 0.60 | 0.64 | 0.00 |
| antmaze-giant-stitch-v0 | 0.80 | 0.80 | 0.80 | 0.00 |

`ecg_fallback_count=1.0` in ECG planner traces, so the planner result is mostly GAS fallback rather than active ECG path execution. The adapter result is a real failure: replacing the low-level action with the adapter collapses success to 0.

## Gate Table

| Gate | Status | Evidence |
| --- | --- | --- |
| Action Dataset Gate | PASS | 600k examples, action supervision 1.0, positive_with_action_count 322888 |
| Contract Model Gate | PASS | contract-positive/negative AUROC beat d_phi baseline |
| Policy Adapter Offline Gate | PASS | validation MSE below mean-action baseline |
| Policy Adapter Online Gate | FAIL | ECG adapter success 0.00 on both AntMaze envs |
| Action-Anchored Graph Gate | PASS | 100k action-anchored edges, no unverified KNN main edges |
| Planner Score Gate | PARTIAL | min-contract improves; negative risk worsens |
| Smoke Gate | PASS | all return codes 0 |
| AntMaze Safety Gate | FAIL for adapter; planner is degenerate PASS | planner falls back to GAS; adapter collapses |
| Online Readiness Gate | FAIL | no active ECG planner path and adapter unsafe |

## Recommendation

Do not enter full AntMaze benchmark, humanoid/teleport, or SOTA benchmark.

The next technical blocker is runtime graph anchoring: the action-anchored graph is trainable and trusted, but online ECG nearest-node/path lookup does not connect current GAS states to usable graph paths. The policy adapter should not be used online until evaluated under a target distribution where ECG actually produces action-anchored targets and adapter action error is probed closed-loop.

## Next Command

Run a focused ECG runtime anchoring debug with light traces:

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/cage_experiment_manifest.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_ecg/action_anchored_eval/runtime_anchor_debug \
  --envs antmaze-giant-navigate-v0 \
  --seeds 42 \
  --variants gas cage_ecg_planner_trace_only cage_ecg_planner \
  --episodes_per_goal 1 \
  --goals_per_env 1 \
  --manifest_path results/cage_ecg/action_anchored_eval/runtime_anchor_debug/manifest.jsonl \
  --strict_paths \
  --cage_debug_light \
  --ecg_graph_path results/cage_ecg/action_anchored_graph_v2/contract_graph.json \
  --ecg_contract_model_path results/cage_ecg/action_anchored_models/contract/model.pt \
  --ecg_policy_adapter_path results/cage_ecg/action_anchored_models/policy_adapter/model.pt \
  --ecg_planner_score_path results/cage_ecg/action_anchored_models/planner_score/weights.json
```
