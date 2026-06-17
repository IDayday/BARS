# Stage64 Research Progress Summary

Date: 2026-06-17

## Scope

This summary covers the current offline-RL graph-repair research line built on
top of GAS artifacts and OGBench datasets.

Method boundary:

- trainable components use only fixed offline datasets;
- actor-aware labels/features use frozen policies on offline data only;
- graph patching changes planner-time edge costs only;
- environment rollouts are used only for reporting and diagnosis.

## Completed Milestones

### Stage44: Local Humanoid/Visual GAS Retrain

Run root:

```text
runs_stage44_humanoid_visual_retrain/20260615_231837
artifacts/gas_ogbench_stage44_humanoid_visual_retrain_20260615_231837
```

Result:

| env | local GAS success |
| --- | ---: |
| `humanoidmaze-large-navigate-v0` | 0.776 |
| `humanoidmaze-large-stitch-v0` | 0.848 |
| `visual-antmaze-large-explore-v0` | 0.040 |
| `visual-scene-play-v0` | 0.004 |

Reading:

- humanoid local retrain is usable as a baseline;
- visual local retrain is very weak and should be treated as a stress-test
  baseline, not as a paper-headline evaluation family.

Reference:

- `docs/stage44_humanoid_visual_retrain_plan.md`

### Stage45: Offline Contract Dataset + CAP-lite Scorer

Key outputs:

- `scripts/stage45_build_offline_contract_dataset.py`
- `scripts/stage45_train_offline_contract_scorer.py`
- `scripts/stage45_make_caplite_edge_scores.py`

Main conclusion:

- a learned offline contract scorer extracts signal beyond raw support counts
  and geometry;
- broad global soft graph patching is viable;
- support count alone is not a complete algorithm.

Reference:

- `docs/stage46_parallel_research_workplan.md`
- `docs/stage47_actor_contract_results.md`

### Stage47-50: Actor-Aware and Sequence-Aware Extensions

Key outputs:

- `scripts/stage47_add_actor_agreement_features.py`
- `scripts/stage48_build_actor_conditioned_contract_labels.py`
- `scripts/stage49_add_sequence_actor_contract_features.py`
- `scripts/stage50_make_hybrid_contract_edge_scores.py`
- `scripts/stage50_launch_hybrid_eval.py`

Main conclusion:

- raw actor features are diagnostically useful but weak as a standalone closed
  loop gain source on AntMaze;
- sequence-level verifier improves offline discrimination;
- using low sequence probability directly as a negative planner penalty is not
  yet well calibrated across environment families.

Reference:

- `docs/stage50_hybrid_eval_results.md`

### Stage54-55: Relative Endpoint Filtering

Key output:

- endpoint-relative support filtering added to
  `scripts/stage45_build_offline_contract_dataset.py`

Main conclusion:

- dense support neighborhoods were a real failure mode, especially on visual
  environments;
- repairing the candidate support graph improves offline contract signal a lot;
- actor-aware refinements become more useful after graph density is repaired.

Reference:

- `docs/stage54_endpoint_filter_results.md`

### Stage61: Humanoid Confirmed Gain

Main result:

| variant | success |
| --- | ---: |
| original | 0.80 |
| filtered base @ `0.10` | 0.81 |
| filtered actor-conditioned @ `0.10` | 0.83 |

Main conclusion:

- strong fixed penalty such as `0.25` is too aggressive;
- mild planner penalty around `0.10` works;
- on humanoid, `filtered + actor-conditioned + mild penalty` is the strongest
  confirmed recipe so far.

Reference:

- `docs/stage61_humanoid_weight_sweep_results.md`

### Stage62: AntMaze Weight Calibration

Representative official-artifact results:

| env | original | best patched confirm | reading |
| --- | ---: | ---: | --- |
| `antmaze-giant-navigate-v0` | 0.77 | 0.78 | small gain only |
| `antmaze-large-explore-v0` | 0.94 | 0.94 | effectively saturated |

Main conclusion:

- the Stage61 calibration lesson transfers: mild penalty is safer than strong
  penalty;
- but AntMaze gains remain small;
- Stage49 hybrid is not yet clearly stronger than simpler Stage45 base.

Reference:

- `docs/stage62_antmaze_weight_sweep_results.md`

### Stage63: Adaptive Penalty

Key output:

- `scripts/stage63_make_adaptive_risk_edge_scores.py`

Representative results:

| env | variant | result |
| --- | --- | --- |
| `antmaze-giant-navigate-v0` | adaptive | no clear gain over fixed |
| `humanoidmaze-large-navigate-v0` | adaptive @ `0.50` | `0.76 -> 0.80` |
| `humanoidmaze-large-navigate-v0` | fixed actor-conditioned @ `0.10` | `0.73 -> 0.87` |

Main conclusion:

- adaptive planner penalties are viable;
- but current edge-static adaptive scaling is not better than the best fixed
  actor-conditioned mild penalty;
- adaptive should be treated as a promising branch, not the new mainline.

Reference:

- `docs/stage63_adaptive_penalty_results.md`

## Current Best-Supported Algorithm Shape

The strongest current mainline is:

```text
relative endpoint support filtering
+ learned offline contract scorer
+ actor-conditioned refinement
+ mild fixed planner penalty
```

This is stronger and more complete than:

- raw support-count patching;
- actor feature add-ons without label changes;
- strong global penalties;
- direct sequence-probability negative penalties.

## What We Can Claim Now

1. strictly offline graph-repair signals can improve over the original GAS
   graph on selected environments;
2. graph repair and penalty calibration must be separated;
3. actor-conditioned refinement is useful when built on top of repaired support;
4. the best confirmed gains currently come from humanoid, not from visual;
5. adaptive penalty is viable but not yet the dominant recipe.

## What We Cannot Claim Yet

1. that the sequence-level hybrid is already the final method;
2. that adaptive penalty already dominates fixed mild penalty;
3. that visual closed-loop results support the main paper claim;
4. that the current method is uniformly better across all tasks within one
   environment family;
5. that the algorithm is already at final ICLR-ready form.

## Current Algorithm Stage

The work is past "small GAS tweaks" and has become a coherent offline-RL method
prototype, but it is not complete yet.

Current stage:

```text
working method family with one confirmed strong recipe,
one promising adaptive branch,
and a clear remaining bottleneck in task/path-local calibration
```

## Next Research Step

The next useful step is not another global weight sweep.

The highest-value next move is:

1. move from static edge-local penalty scaling to path-local or task-local
   calibration;
2. use joint evidence from contract risk, actor mismatch, sequence evidence,
   and graph role;
3. test whether that calibration can keep humanoid gains while reducing AntMaze
   task tradeoffs.

## Document Index

- `docs/stage44_humanoid_visual_retrain_plan.md`
- `docs/stage46_parallel_research_workplan.md`
- `docs/stage47_actor_contract_results.md`
- `docs/stage50_hybrid_eval_results.md`
- `docs/stage54_endpoint_filter_results.md`
- `docs/stage61_humanoid_weight_sweep_results.md`
- `docs/stage62_antmaze_weight_sweep_results.md`
- `docs/stage63_adaptive_penalty_results.md`
