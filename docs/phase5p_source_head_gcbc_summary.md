# Phase 5P Source-Head Planner GCBC Summary

## Goal

Phase 5P tested whether separating target families with source-conditioned
action heads fixes the Phase 5N policy conflict between:

- final-goal hindsight targets;
- support-edge local targets;
- planner-first-edge replay targets.

The runtime convention is:

- direct final-goal rollout uses target source `0`;
- hierarchical edge/subgoal rollout uses target source `2`;
- old checkpoints remain compatible because source heads are opt-in.

## Training Result

Run:

`results/phase5p/antmaze_large_stitch/core_plus_bottleneck_budget120_H10_source_head`

Final 100000-step supervised metrics:

| metric | value |
| --- | ---: |
| val_action_mse | 0.054434 |
| final_goal_hindsight_val_mse | 0.085486 |
| support_edge_local_val_mse | 0.036956 |
| planner_first_edge_replay_val_mse | 0.032911 |
| planner_used_edge_val_mse | 0.037611 |
| low_support_edge_val_mse | 0.026013 |
| long_horizon_edge_val_mse | 0.027956 |

Compared with Phase 5N, source heads did not improve supervised fitting:

| run | val_action_mse | final-goal MSE | support-edge MSE | planner-replay MSE |
| --- | ---: | ---: | ---: | ---: |
| Phase 5N planner GCBC | 0.044359 | 0.087105 | 0.032122 | 0.029160 |
| Phase 5P source-head GCBC | 0.054434 | 0.085486 | 0.036956 | 0.032911 |

The only clear supervised improvement is a small final-goal MSE reduction. It
comes with worse overall, support-edge, and planner-replay MSE.

## Rollout Result

Both rollouts use 3 natural-start episodes on `antmaze-large-stitch-v0`, task
ID 1, max 120 steps.

| method | success_rate | mean_final_goal_l2 |
| --- | ---: | ---: |
| Phase 5P direct source-head | 0.0 | 42.168 |
| Phase 5P state-outcome hierarchical | 0.0 | 40.564 |
| Phase 5N direct planner GCBC | 0.0 | 44.736 |
| Phase 5N state-outcome hierarchical | 0.0 | 36.032 |
| old state-outcome GCBC | 0.0 | 39.720 |

Phase 5P slightly improves direct final-goal distance versus Phase 5N direct
and the older direct GCBC reference, but it degrades the best hierarchical
state-outcome result and still has zero task success.

## Interpretation

Source-specific heads are not the breakthrough. The result confirms that the
target-family conflict is real, but simple head factorization is too weak:

- the final-goal head remains high-MSE;
- the planner/subgoal heads are worse than Phase 5N;
- closed-loop hierarchical progress regresses.

This supports the Stage36 priority: test BARS graph evidence on a mature GAS
actor before spending more GPU on isolated GCBC variants. For BARS-native policy
training, the next useful direction should be stronger than source heads, such
as separate direct/option policies, a skill-space policy matched to graph
targets, or joint graph-policy training with closed-loop feedback.

Offline action MSE remains diagnostic only. The current success rate is still
zero and does not support a SOTA claim.
