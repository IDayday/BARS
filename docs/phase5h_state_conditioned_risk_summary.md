# Phase 5H State-Conditioned Initiation Risk

Phase 5H addresses the main limitation left by Phase 5G: a static edge-id risk
prior cannot tell whether the current online state is actually close to the
offline initiation samples of the next support edge.

## Design Rationale

The same related-work pattern from SoRB, HIQL, and TTGS applies here: graph
search needs a local reachability or execution-risk signal, not only graph
connectivity. Phase 5H keeps BARS support-only semantics and makes the first
edge after each replan state-conditioned.

For each replan, Phase 5H scores outgoing support edges from the current
cluster:

```text
min_initiation_distance =
    min_i || online_obs[dims] - offline_initiation_i[dims] ||

state_conditioned_risk_score =
    1 - exp(-min_initiation_distance / distance_scale)

state_conditioned_risk_penalty =
    state_risk_penalty_weight * state_conditioned_risk_score
```

For AntMaze the distance dims are `[0, 1]`, matching the grid-xy cluster
geometry. This dynamic penalty is added to:

- Phase 5F online outcome memory penalty;
- Phase 5G offline edge prior;
- within-episode failure penalty.

No unsupported edge is added.

## Commands

```bash
conda run -n gcrlo pytest -q tests/test_phase3_synthetic.py

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5h_state_conditioned_antmaze_corebot100k.yaml \
  --device cpu
```

## Results

AntMaze natural-start, task id 1, seeds 0/1/2, 120-step cap:

| method | success | mean final L2 | mean L2 improvement | mean completed edges | mean replans |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 5F outcome-only | 0.0 | 43.0624 | 0.9920 | 1.667 | 7.333 |
| Phase 5G offline prior + outcome | 0.0 | 42.0333 | 1.7897 | 0.333 | 7.667 |
| Phase 5H state-conditioned risk | 0.0 | 42.1237 | 1.7527 | 1.667 | 8.333 |

Outputs:

- `results/phase3f/antmaze_large_stitch/state_conditioned_3ep_corebot100k_H10_B120/`
- `results/phase3f/antmaze_large_stitch/state_conditioned_w0p5_3ep_corebot100k_H10_B120/`
- `results/phase3f/antmaze_large_stitch/state_conditioned_w2p0_3ep_corebot100k_H10_B120/`

Dynamic risk diagnostics:

- `state_conditioned_risk_scores.csv` contains 146 replan-candidate rows in the
  main run.
- Mean min initiation distance: `3.6451`.
- Mean state-conditioned risk penalty: `0.3013`.
- The selected-edge traces now separate `edge_static_risk_penalty` from
  `edge_state_risk_penalty`.

Small weight check:

| state risk weight | mean final L2 | mean L2 improvement | mean completed edges |
| ---: | ---: | ---: | ---: |
| 0.5 run A | 42.1237 | 1.7527 | 1.667 |
| 0.5 run B | 40.7735 | 3.8549 | 1.333 |
| 2.0 | 41.7904 | 0.6624 | 1.000 |

The natural-start smoke has visible variance. The safer conclusion is that
mild state-conditioned risk is directionally useful, while a stronger dynamic
penalty is not clearly better.

## Analysis

Phase 5G improved goal distance but reduced completed edge count sharply. Phase
5H recovers completed-edge progress while keeping most of the goal-distance
gain. That is a better trade-off for a hierarchical executor: the planner is
less likely to choose a first edge whose offline initiation set is far from the
current online state, but the penalty is not so strong that all edge progress is
suppressed.

The remaining failure mode is still low-level execution and recovery. Phase 5H
does not learn a policy or a calibrated success model. It is a planner-side
state-conditioned heuristic derived from support segments.

## Conclusion

Phase 5H is the most useful Phase 5 planner-side modification so far, but it is
not a finished algorithm and still gives zero task success in the AntMaze
smoke.

The next step should learn a small state-conditioned edge success model from:

- current online state;
- nearest offline initiation distance;
- offline support/certification features;
- GCBC action-fit proxy;
- compatibility context;
- online attempt outcomes.

That model should replace the hand-weighted risk sum and be validated against
held-out online attempt traces before larger online sweeps.
