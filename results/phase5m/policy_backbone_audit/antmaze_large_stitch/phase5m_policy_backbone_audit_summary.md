# Phase 5M Policy Backbone Audit

This audit separates graph/planner evidence from low-level policy evidence.
The goal is to find the fastest success-rate validation path rather than
continuing graph-only improvements.
Cross-policy reuse is diagnostic only; final BARS evidence requires a
joint graph-policy-training loop.

## Evidence

- environment: `antmaze-large-stitch-v0`
- ready official GAS backbones: `3`
- live local GAS backbones: `0`
- official GAS success rate: `0.9520`
- best current BARS smoke: `bars_phase5i_state_outcome_w0p5` success `0.0000`, mean final L2 `39.7197`

## Recommended Matrix

| experiment | status | policy | planner | required work |
| --- | --- | --- | --- | --- |
| `official_gas_policy_official_gas_graph` | `blocked_missing_live_gas_artifacts` | `official_gas_actor` | `official_gas_keygraph_shortest_path` | Run or reuse official evaluate_gas.py outputs. |
| `bars_support_graph_bars_gcbc` | `completed_smoke` | `bars_phase3_gcbc` | `bars_support_option_graph` | Already available for 3-episode AntMaze smoke. |
| `bars_support_graph_gas_actor` | `blocked_missing_live_gas_artifacts` | `official_gas_actor` | `bars_support_option_graph` | First audit BARS target phi distribution against GAS keygraph/policy target distribution; only then implement GAS-phi adapter. |
| `official_gas_graph_bars_gcbc` | `not_direct_without_decoder_or_nearest_raw_node` | `bars_phase3_gcbc` | `official_gas_keygraph_shortest_path` | Map GAS keygraph phi targets to nearest raw observations or add a raw-goal reconstruction path. |
| `bars_planner_subgoal_replay_policy` | `next_training_path` | `bars_gcbc_retrained_on_planner_subgoals` | `bars_support_option_graph` | Build a joint graph-policy training loop: graph-derived subgoals, aligned goal/skill representation, matched sampler, and natural-start eval. |

## Interpretation

The first diagnostic is not rollout; it is target-distribution feasibility.
`bars_support_graph_gas_actor` should run only if BARS support targets mapped
through GAS `get_phi` lie near the GAS keygraph/policy target distribution.
The GAS actor is trained with GAS's TDR/graph/skill distribution, so raw BARS
cluster/termination targets may be out of distribution even after phi mapping.

The reverse composition, `official_gas_graph_bars_gcbc`, is not directly
comparable because GAS planner targets live in TDR phi/skill space while the
current BARS GCBC consumes raw observation goals.

The main algorithm path remains `bars_planner_subgoal_replay_policy`: train
the BARS low-level policy on the same graph-derived goal/skill distribution
that BARS will execute at test time, then evaluate natural-start success.
