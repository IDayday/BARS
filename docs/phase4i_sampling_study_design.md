# Phase 4I Stronger GCBC Sampling Study Design

Phase 4I tests whether support-aware edge sampling can improve offline GCBC
fitting on rare option edges without giving up average validation accuracy.

## Motivation

Phase 4H showed that a stronger Scene GCBC substantially improves direct
repair-edge action fitting. It did not answer which sampling strategy should be
used for longer training. Phase 3D only ran 200-step smoke ablations, so its
sampling conclusions are too weak.

## New Samplers

Phase 4I adds two edge-level samplers:

- `support_balanced`: edge probability is proportional to
  `1 / sqrt(num_unique_starts)`.
- `bottleneck_support_balanced`: combines the same inverse-sqrt support weight
  with a normalized bottleneck multiplier.

The inverse-sqrt form is deliberately conservative: it lifts long-tail edges but
does not let one-segment edges dominate the whole loader.

## Related Work Checked

- Goal-Conditioned Supervised Learning: https://arxiv.org/abs/1912.06088
- RvS supervised offline RL: https://arxiv.org/abs/2112.10751
- Class-Balanced Loss: https://arxiv.org/abs/1901.05555
- Focal Loss: https://arxiv.org/abs/1708.02002

The long-tail papers motivate support-aware reweighting. They do not imply that
lower action MSE is rollout success.

## Evidence Standard

The study reports:

- final validation action MSE;
- bottleneck edge validation MSE;
- low-support edge validation MSE;
- long-horizon edge validation MSE;
- `rare_edge_mean_mse`, the mean of those three rare-edge metrics;
- ratios versus `uniform_transition`.

A sampler is only considered promising if it improves rare-edge metrics without
a large overall validation MSE regression. This remains an offline supervised
proxy study.
