# Phase 5K Preplan Policy Mismatch

Phase 5K makes GCBC policy mismatch available before graph search. Earlier
phases used policy MSE only during subgoal selection, after a graph path had
already been chosen. Phase 5K scores every outgoing support-edge candidate at
replan time by evaluating the GCBC one-step action prediction on sampled offline
initiation-to-termination segments.

No unsupported edges are added.

## Design

For each candidate edge from the current cluster:

```text
preplan_policy_action_mse =
    mean_u || pi(obs_u, goal=obs_T) - action_u ||^2
```

where `(u, T)` are real support segments for that edge. The score is written to
`preplan_policy_mismatch_scores.csv`, copied into
`state_conditioned_outcome_scores.csv`, and can be used by the Phase 5I
state-conditioned outcome model through feature column
`preplan_policy_action_mse`.

## Commands

```bash
conda run -n gcrlo pytest -q tests/test_phase3_synthetic.py

conda run -n gcrlo python scripts/run_phase5j_state_outcome_calibration.py \
  --config configs/phase5k_preplan_policy_calibration_antmaze.yaml

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5k_preplan_policy_mismatch_antmaze_corebot100k.yaml \
  --device cpu
```

## Calibration

Compared with Phase 5J's planner-side feature set, adding
`preplan_policy_action_mse` improves heldout attempt discrimination:

| model | val Brier | val log loss | val AUC | val risk separation |
| --- | ---: | ---: | ---: | ---: |
| Phase 5J state-outcome | 0.1155 | 0.3995 | 0.5556 | 0.0110 |
| Phase 5K + preplan policy mismatch | 0.1142 | 0.3914 | 0.5979 | 0.0299 |

The selected penalty weight remains `0.5` under the same mean-penalty budget.

## Online Smoke

AntMaze natural-start, task id 1, seeds 0/1/2, 120-step cap:

| method | success | mean final L2 | mean L2 improvement | mean completed edges | mean replans | mean failed attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5I learned outcome, weight 0.5 | 0.0 | 39.7197 | 4.1465 | 2.333 | 7.000 | 6.000 |
| Phase 5K preplan mismatch, weight 0.25 | 0.0 | 40.4602 | 3.9329 | 2.667 | 6.667 | 5.667 |
| Phase 5K preplan mismatch, weight 0.5 | 0.0 | 39.8942 | 3.9258 | 1.667 | 6.333 | 5.333 |
| Phase 5K preplan mismatch, weight 0.75 | 0.0 | 41.1478 | 1.9882 | 1.667 | 6.333 | 5.667 |

Outputs:

- `results/phase3f/antmaze_large_stitch/preplan_policy_calibration_phase5k/`
- `results/phase3f/antmaze_large_stitch/preplan_policy_mismatch_w0p25_3ep_corebot100k_H10_B120/`
- `results/phase3f/antmaze_large_stitch/preplan_policy_mismatch_w0p5_3ep_corebot100k_H10_B120/`
- `results/phase3f/antmaze_large_stitch/preplan_policy_mismatch_w0p75_3ep_corebot100k_H10_B120/`

## Analysis

Phase 5K is useful but not a breakthrough. It improves heldout attempt-risk
discrimination and reduces replanning / failed-attempt counts in the online
smoke, but it does not beat Phase 5I weight `0.5` on final goal distance and
still gives zero task success.

The result supports a narrower conclusion: policy mismatch is a real risk
feature and should remain in the model, but ranking edges by risk is still not
enough. The executor also needs a recovery or local control improvement when a
selected edge starts drifting.

## Next Step

The next mature step should address low-level execution, not only graph risk:

- collect more online attempts across seeds/tasks;
- learn an edge-local recovery trigger from subgoal-distance growth;
- switch or replan before the edge horizon expires;
- validate against Phase 5I/5K with paired natural-start seeds.
