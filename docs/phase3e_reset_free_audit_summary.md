# Phase 3E Reset-Free Offline Edge Certification + GAS Graph Audit

Phase 3E is reset-free and offline-only. It does not run environment rollout,
does not require arbitrary reset-to-state, and does not claim online success.
The current `env_unavailable` status is a Python dependency blocker for env
construction, not evidence that AntMaze or Scene lack reset support.

## Phase 3A Supervised GCBC

The 100000-step AntMaze GCBC run has final `val_action_mse = 0.0426389`.
This shows the state-based GCBC can fit held-out edge BC samples, but offline
action MSE is not edge execution success.

Grouped metrics are written to:

- `results/phase3/antmaze_large_stitch/core_plus_bottleneck/grouped_val_metrics.csv`
- `results/phase3/antmaze_large_stitch/core_plus_bottleneck/phase3a_supervised_summary.json`

## Phase 3D Sampling Ablation

Sampling ablation remains an offline supervised comparison of
`uniform_transition`, `uniform_edge`, and `bottleneck_weighted`. It does not
measure rollout success.

Outputs:

- `results/phase3_sampling/antmaze_large_stitch/all_sampling_ablation_summary.csv`
- `results/phase3_sampling/scene_play/all_sampling_ablation_summary.csv`
- `docs/phase3_sampling_ablation_summary.md`

## Phase 3E Offline Certification

The certification score combines heldout episode support, GCBC action-fitting
proxy, compatibility context, and a simple OOD penalty.

Current runs:

- AntMaze `core_plus_bottleneck_budget120_H10`: 40 / 582 edges certified offline
  at the current threshold, rate `0.068729`.
- Scene `core_plus_bottleneck_budget192_H5`: 209 / 1897 edges certified offline
  at the current threshold, rate `0.110174`.

Outputs:

- `results/phase3e/antmaze_large_stitch/core_plus_bottleneck_budget120_H10/`
- `results/phase3e/scene_play/core_plus_bottleneck_budget192_H5/`

These are reset-free risk proxies. They are intended to filter high-risk edges,
not to replace closed-loop rollout validation.

## GAS/TDR-Style Graph Audit

The audit compares Phase 2 support-certified graph edges with kNN, random, and
`GAS_style_threshold_graph` approximation edges. The GAS-style graph is a
diagnostic proximity approximation, not an official GAS graph.

Current summaries:

- AntMaze: highest path coverage is `random_graph`, lowest unsupported edge
  rate is `support_graph`, and the strongest unsupported shortcut reliance is
  `random_graph`.
- Scene: highest path coverage is `random_graph`, lowest unsupported edge rate
  is `support_graph`, and the strongest unsupported shortcut reliance is
  `random_graph`.

In both datasets, the support-certified graph reduces path risk by construction:
its graph edges are Phase 2 data-supported option edges. kNN/proximity/random
graphs can appear connected while relying on unsupported shortcuts.

Outputs:

- `results/phase3e_gas_audit/antmaze_large_stitch/`
- `results/phase3e_gas_audit/scene_play/`

## Phase 3F Scaffold

Natural-start rollout interfaces were added, but rollout is skipped while the
environment preflight is `env_unavailable`.

Outputs:

- `results/phase3f/antmaze_large_stitch/support_shortest_path/`
- `results/phase3f/scene_play/support_shortest_path/`
