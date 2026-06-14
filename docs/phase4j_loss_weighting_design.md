# Phase 4J Mixed/Loss-Weighted GCBC Design

Phase 4I showed that hard edge-level oversampling can improve rare-edge MSE but
causes too much overall validation regression. Phase 4J keeps the broad
`uniform_transition` sampling distribution and instead applies soft per-edge
loss weights.

## Target Failure Mode

Rare, low-support, bottleneck, and long-horizon option edges may be underfit by
plain transition-uniform GCBC. Hard oversampling damages average action fitting.
The Phase 4J hypothesis is that small loss-side weights can improve rare-edge
fitting while preserving transition coverage.

## Weighting Modes

- `support`: inverse-sqrt `num_unique_starts`, normalized to mean 1.
- `bottleneck`: normalized `edge_bottleneck_score` multiplier.
- `support_bottleneck`: product of support and bottleneck components.

The configured Scene study uses strength `0.3` and clips weights to `[0.7, 1.8]`
to avoid the hard-oversampling failure observed in Phase 4I.

## Related Work Checked

- Goal-Conditioned Supervised Learning: https://arxiv.org/abs/1912.06088
- RvS supervised offline RL: https://arxiv.org/abs/2112.10751
- Class-Balanced Loss: https://arxiv.org/abs/1901.05555
- Focal Loss: https://arxiv.org/abs/1708.02002

The long-tail papers motivate loss-side reweighting. They do not justify
interpreting lower action MSE as rollout success.

## Evidence Standard

The study compares weighted variants to the Phase 4I `uniform_transition`
baseline on the same Scene H5 Phase 2 graph and seeds `[0, 1]`.

Primary metrics:

- final validation action MSE;
- bottleneck-edge validation MSE;
- low-support-edge validation MSE;
- long-horizon-edge validation MSE;
- `rare_edge_mean_mse`;
- ratios against `uniform_transition`.

A weighted method is promising only if rare-edge MSE improves with at most 5%
overall validation MSE regret.
