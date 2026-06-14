# Phase 4M Planner-Relevant Repair Loss Weighting

This phase is an offline supervised GCBC loss-weighting experiment.
It does not use environment rollout and does not claim edge execution
success.

## Question

Can repair-edge loss weighting target edges that are both hard and
actually used by the compatibility-aware repaired planner?

## Baseline Comparisons

| method | final_val_action_mse | direct_repair_edge_mse | planner_used_repair_edge_mse | final_val_action_mse_ratio_vs_baseline | planner_used_repair_edge_mse_ratio_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| augmented_loss_support_bottleneck_s03 | 0.0942923 | 0.0818255 | 0.090441 | 1 | 1 |
| planner_relevant_repair_s04 | 0.0938325 | 0.081423 | 0.0892153 | 0.995124 | 0.986447 |

## Related Work Checked

- [Goal-Conditioned Supervised Learning](https://arxiv.org/abs/1912.06088): Goal-conditioned supervised policy training framing.
- [RvS: What is Essential for Offline RL via Supervised Learning?](https://arxiv.org/abs/2112.10751): Supervised offline RL claim-boundary reference.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): Priority signal inspiration; Phase 4M uses supervised loss weights, not TD-error replay.
- [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555): Long-tail weighting motivation.

## Interpretation Rules

- The training examples remain support-certified offline segments.
- Planner relevance is a weighting signal, not a new unsupported edge source.
- Direct repair-edge action MSE is a reset-free supervised proxy, not rollout success.
- The useful comparison is against the same augmented graph with ordinary
  support+bottleneck loss weighting.
