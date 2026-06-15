# Phase 5I State-Conditioned Outcome Model

Phase 5I replaces part of the hand-weighted Phase 5H planner heuristic with a
small learned attempt-outcome model. The model is still support-only: it scores
candidate support edges during replanning and never creates unsupported edges.

## Design

The model trains on prior natural-start hierarchical rollout traces. Each
attempt is labeled as:

- `completed = 1` if the episode enters the candidate edge destination cluster;
- `timeout = 1` otherwise.

Attempt features are planner-side quantities available before executing the
next edge:

- `edge_static_risk_penalty` from Phase 5F/5G online memory and offline prior;
- `edge_state_risk_penalty` from Phase 5H nearest-initiation risk;
- `edge_failure_count` within the current episode;
- `base_planning_cost`;
- `selected_init_distance`.

The current implementation fits an L2 logistic model with a Beta-smoothed
fallback failure probability. During each replan, outgoing support edges are
scored by predicted failure probability and the resulting penalty is added as
`edge_learned_state_risk_penalty`.

## Commands

```bash
conda run -n gcrlo pytest -q tests/test_phase3_synthetic.py

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5i_state_outcome_antmaze_corebot100k.yaml \
  --device cpu
```

## Results

AntMaze natural-start, task id 1, seeds 0/1/2, 120-step cap:

| method | success | mean final L2 | mean L2 improvement | mean completed edges | mean replans | mean failed attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5F outcome-only | 0.0 | 43.0624 | 0.9920 | 1.667 | 7.333 | 6.333 |
| Phase 5G offline prior + outcome | 0.0 | 42.0333 | 1.7897 | 0.333 | 7.667 | 6.667 |
| Phase 5H state-conditioned risk | 0.0 | 42.1237 | 1.7527 | 1.667 | 8.333 | 7.333 |
| Phase 5I learned outcome, weight 0.5 | 0.0 | 39.7197 | 4.1465 | 2.333 | 7.000 | 6.000 |
| Phase 5I learned outcome, weight 1.0 | 0.0 | 42.3117 | 1.4185 | 2.000 | 6.333 | 5.333 |
| Phase 5I learned outcome, weight 2.0 | 0.0 | 40.8721 | 2.9853 | 1.333 | 7.333 | 6.333 |

The trained model used 80 historical attempt examples with 69 failures. The
main fitted weights assign the largest positive coefficient to
`selected_init_distance`, matching the Phase 5H observation that current-state
distance to offline initiation support is an important edge-risk signal.

Outputs:

- `results/phase3f/antmaze_large_stitch/state_outcome_w0p5_3ep_corebot100k_H10_B120/`
- `results/phase3f/antmaze_large_stitch/state_outcome_3ep_corebot100k_H10_B120/`
- `results/phase3f/antmaze_large_stitch/state_outcome_w2p0_3ep_corebot100k_H10_B120/`

## Analysis

Phase 5I weight `0.5` is the strongest online partial-progress result so far in
this AntMaze smoke. It improves mean final goal distance and completed edge
count relative to Phase 5F, Phase 5G, and Phase 5H. The learned model also
reduces failed attempts modestly.

The result is not task success. All methods still have success `0.0`, and the
training set is very small and failure-heavy. The model should therefore be
treated as an online-attempt risk prior, not as a calibrated execution-success
probability.

The useful pattern is now clearer:

- static offline support priors improve goal-distance progress but can suppress
  edge completion;
- state-conditioned initiation distance recovers edge progress;
- a mild learned outcome penalty combines these signals better than either
  alone;
- too large a learned penalty again becomes over-conservative.

## Next Step

The next mature version should collect more online attempts across tasks and
seeds, split them into train/validation, and calibrate the model against heldout
attempt completion. The planner should then use a validation-selected penalty
weight rather than a hand-picked smoke value.
