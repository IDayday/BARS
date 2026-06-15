# Phase 5J State-Outcome Calibration

This is a heldout attempt-level calibration check for the Phase 5I
state-conditioned outcome model. It does not claim task success.

## Summary

- dataset: `antmaze-large-stitch-v0`
- num examples: `213`
- train examples: `152`
- val examples: `61`
- selected penalty weight: `0.5`
- val Brier: `0.11554173858554204`
- val AUC: `0.5555555555555556`
- val risk separation: `0.011043365889632506`

The selected weight is constrained by validation mean-penalty budgets;
it is a conservative planner-cost scale, not a calibrated success
probability threshold.
