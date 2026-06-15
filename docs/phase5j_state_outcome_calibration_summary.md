# Phase 5J State-Outcome Calibration

Phase 5J turns the Phase 5I state-conditioned outcome model into an auditable
heldout calibration workflow. The goal is to stop selecting planner risk weights
only from online smoke outcomes.

## Method

The script reads prior natural-start hierarchical rollout traces and extracts
attempt-level examples. Splitting is done by trace group
`run:seed:episode`, so attempts from the same episode do not appear in both
train and validation.

The calibrated model uses the same planner-visible features as Phase 5I:

- static edge risk;
- state-conditioned initiation risk;
- within-episode edge failure count;
- base planning cost;
- selected initiation distance.

The penalty-weight selector is deliberately conservative. It chooses the largest
validation risk separation subject to mean learned-penalty budgets. This makes
the selected weight a planner-cost scale, not a probability threshold.

## Command

```bash
conda run -n gcrlo python scripts/run_phase5j_state_outcome_calibration.py \
  --config configs/phase5j_state_outcome_calibration_antmaze.yaml
```

## Results

Inputs:

- 8 AntMaze natural-start rollout result dirs from Phase 5F/5G/5H/5I;
- 213 extracted edge-attempt examples;
- 24 trace groups;
- train/val split: 152 / 61 examples.

Heldout validation:

| split | examples | failures | completions | Brier | log loss | AUC | risk separation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 152 | 124 | 28 | 0.1358 | 0.4346 | 0.6999 | 0.0828 |
| val | 61 | 54 | 7 | 0.1155 | 0.3995 | 0.5556 | 0.0110 |

Penalty selector:

- selected weight: `0.5`;
- validation mean penalty at weight `0.5`: `0.3996`;
- validation mean completed-edge penalty at weight `0.5`: `0.3947`;
- higher weights violate the configured mean-penalty budget `0.5`.

Outputs:

- `results/phase3f/antmaze_large_stitch/state_outcome_calibration_phase5j/attempt_examples.csv`
- `results/phase3f/antmaze_large_stitch/state_outcome_calibration_phase5j/calibration_metrics.csv`
- `results/phase3f/antmaze_large_stitch/state_outcome_calibration_phase5j/penalty_weight_selection.csv`
- `results/phase3f/antmaze_large_stitch/state_outcome_calibration_phase5j/phase5j_calibration_summary.json`

## Analysis

Phase 5J supports the Phase 5I weight choice only weakly. The budget rule
selects `0.5`, which matches the best Phase 5I smoke, but heldout predictive
separation is small. Validation AUC is only `0.5556`, so the current model is
not a mature calibrated edge-success model.

The main useful finding is diagnostic: `selected_init_distance` remains the
largest positive fitted risk coefficient, and `edge_state_risk_penalty` is also
positive. This reinforces the emerging rule that online state must be matched
to offline initiation support before trusting a graph edge.

A quick offline check showed that adding plan-time GCBC policy mismatch features
can improve heldout AUC from about `0.56` to about `0.60`, but the current
planner does not yet compute that feature for every outgoing edge before graph
search. The next implementation should make candidate-edge policy mismatch a
first-class preplan feature rather than using it only inside subgoal selection.

## Conclusion

Phase 5J improves experimental discipline, not task performance. It prevents
overclaiming Phase 5I as a calibrated algorithm and points to the next concrete
algorithmic improvement: preplan candidate-edge policy mismatch plus more
online attempts across seeds/tasks.
