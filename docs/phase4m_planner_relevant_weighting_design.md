# Phase 4M Planner-Relevant Repair Loss Weighting Design

## Purpose

Phase 4L showed that `loss_support_bottleneck_s03` improves hard repair-edge
groups, but the repaired planner uses only a small subset of those improved
edges. Phase 4M tests whether a supervised GCBC loss weight can target repair
edges that are both hard and planner-relevant.

This remains reset-free offline supervised learning. It is not environment
rollout, not online edge execution, and not TDR/TMD/MQE.

## Related Work Checked

- [Goal-Conditioned Supervised Learning](https://arxiv.org/abs/1912.06088):
  frames goal-conditioned policy learning as supervised action fitting.
- [RvS: What is Essential for Offline RL via Supervised Learning?](https://arxiv.org/abs/2112.10751):
  reinforces the supervised offline RL claim boundary.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952):
  motivates using priority signals, but Phase 4M uses static supervised loss
  weights rather than TD-error replay.
- [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555):
  motivates clipped long-tail reweighting.

## Method

Inputs:

- Phase 4E repaired support graph;
- Phase 4E repaired planner paths;
- Phase 2 base and repair-bank edge segments;
- OGBench offline dataset arrays.

The script reconstructs `augmented_edge_segments.npz` from Phase 4E selected
repair edges and Phase 2 bank segments. It then trains GCBC on the same
support-certified augmented graph under two conditions:

- `augmented_loss_support_bottleneck_s03`: clipped support+bottleneck loss
  weights on the augmented graph;
- `planner_relevant_repair_s04`: support+bottleneck base weights plus an
  additional multiplier for repair edges used by the repaired compatibility
  planner and hard repair edges.

No unsupported proximity edge is introduced. Planner relevance is only a loss
weighting signal.

## Evidence Standard

The primary comparison is against `augmented_loss_support_bottleneck_s03`, not
against the older base-graph model. Metrics:

- final heldout validation action MSE;
- direct repair-edge action MSE;
- planner-used repair-edge action MSE;
- direct repair policy support score.

All metrics are offline supervised proxies and do not imply rollout success.
