# CAGE Focused Experiment Protocol

## A. Purpose

This protocol evaluates the graph-to-control execution interface in GAS-style offline HRL. The focused question is whether GAS failures arise after planning, when the low-level policy drifts away from the planned subgoal chain, and whether CAGE reduces that execution mismatch without changing the graph, TDR, or low-level policy.

## B. Frozen Components

Every compared run must use the same:

- TDR checkpoint.
- Keygraph.
- Low-level policy checkpoint.
- Evaluation goal protocol.
- Seeds.
- Environment horizon.
- Task IDs selected by `--eval_max_tasks`.

The experiment scripts do not retrain TDR, reconstruct keygraphs, or train policies. They only route evaluation through baseline GAS or opt-in CAGE execution variants.

## C. Compared Variants

- `gas`: baseline GAS evaluation. It does not pass `--use_cage`.
- `cage_trace_only`: CAGE controller and tracing are constructed, but the executed target is the original GAS subgoal. This is a parity/instrumentation check and should match GAS as closely as possible.
- `cage_fixed_commit`: CAGE enabled with subgoal commitment only. Drift-triggered control, local recovery, adaptive horizon, and final-goal controller are disabled.
- `cage_drift_only`: CAGE enabled with drift-triggered global replanning behavior isolated from recovery, adaptive horizon, and final-goal controller.
- `cage_recovery_only`: CAGE enabled with local recovery before global replanning, with adaptive horizon and final-goal controller disabled.
- `cage_full`: current CAGE-MVP controller.
- `cage_safe_full`: current CAGE-MVP controller plus explicit churn guardrails. It does not replace `cage_full`; it is a separate repair variant for measuring whether replan/recovery storms can be prevented.
- `cage_reachability`: reserved for CAGE-v1 with a trained reachability model. It is marked unsupported until the evaluator can load and use such a model.
- `cage_risk_path`: reserved for CAGE-v2 risk-aware path execution. It is marked unsupported until explicitly implemented.

## D. Focused Environments

Default focused list:

- `antmaze-giant-navigate-v0`
- `antmaze-giant-stitch-v0`
- `humanoidmaze-medium-navigate-v0`
- `humanoidmaze-large-navigate-v0`
- `humanoidmaze-large-stitch-v0`
- `scene-play-v0`
- `kitchen-partial-v0`
- `visual-antmaze-giant-stitch-v0`

## E. Metrics

Success alone is insufficient because the hypothesis concerns why a rollout fails after a graph path exists. The trace must explain whether a score change is caused by improved execution alignment or by unrelated variance.

Primary diagnostics:

- `success_rate`
- `normalized_score`
- `no_path_rate`
- `target_switch_count`
- `early_switch_count`
- `mean_commitment_length`
- `stall_count`
- `drift_count`
- `recovery_attempt_count`
- `recovery_success_rate`
- `global_replan_request_count`
- `global_replan_request_rate_per_100_steps`
- `max_consecutive_replan_burst`
- `segment_target_reach_rate`
- `mean_segment_progress`
- `mean_distance_to_path`
- `final_goal_on_rate`
- `final_goal_switch_count`
- `final_goal_stall_count`
- `timeout_rate`
- `path_changed_from_gas_rate`, when future variants change path execution.
- `path_min_reachability`, when future reachability models are available.
- `churn_guard_trigger_count`, `fallback_to_gas_step_count`, and replan/recovery suppression counts for `cage_safe_full`.

## F. Success Criteria

- GAS is behaviorally unchanged when CAGE is disabled.
- Standard or easier tasks do not materially regress.
- Giant AntMaze and HumanoidMaze show reduced drift, stall, and target switching under CAGE variants.
- Scene and Kitchen improvements are interpreted cautiously because semantic goal interfaces may differ from geometric path execution.
- Any improvement must be explained by trace metrics, not only final score.

## G. Non-Goals

- No new graph construction.
- No TDR retraining.
- No low-level policy retraining.
- No benchmark-wide claim until this focused protocol is stable.
