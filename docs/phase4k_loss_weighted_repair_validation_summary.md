# Phase 4K Loss-Weighted GCBC Direct Repair-Edge Validation Summary

Phase 4K evaluates whether the Phase 4J loss-weighted GCBC checkpoints improve
direct repair-edge supervised evidence, not just ordinary edge validation MSE.
No environment rollout or new training was used.

## Command

```bash
python scripts/run_phase4k_loss_weighted_repair_validation.py --config configs/phase4k_loss_weighted_repair_scene_H5_B192_3000.yaml
```

The run used `scene-play-v0`, Phase 2
`core_plus_bottleneck_budget192_H5`, the Phase 4E/4F repaired graph, and the
same Phase 4G direct repair-edge scoring protocol.

## Result

| method | final val MSE ratio | direct repair MSE | direct repair MSE ratio | direct certified rate | planner uncertified frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uniform_transition_none` | 1.000 | 0.015525 | 1.000 | 0.887 | 0.035252 |
| `loss_support_s03` | 1.023 | 0.015547 | 1.001 | 0.888 | 0.034210 |
| `loss_bottleneck_s03` | 1.034 | 0.015677 | 1.010 | 0.889 | 0.034210 |
| `loss_support_bottleneck_s03` | 1.016 | 0.015314 | 0.986 | 0.890 | 0.034210 |

`loss_support_bottleneck_s03` remains the recommended variant under the
configured 5% ordinary-validation regret rule.

## Analysis

Phase 4K makes the Phase 4J conclusion more precise. The combined
support+bottleneck loss weight transfers to direct repair-edge evidence, but
only modestly: direct repair-edge MSE improves by about 1.4% over
`uniform_transition_none`. The single-signal variants are not convincing:
support-only is essentially flat and bottleneck-only worsens direct repair-edge
MSE.

This supports using small clipped combined loss weights as the current
training-side candidate. It also argues against hard oversampling and against
large single-signal weighting as defaults.

## Claim Boundary

This is reset-free offline supervised evidence. Direct repair-edge MSE is not
closed-loop option execution, and the current result is only Scene H5 with
two seeds and 3000-step checkpoints. AntMaze and Scene H10/H25 replication are
still required before treating the weighting rule as general.

Related work reviewed: GCSL, RvS, Class-Balanced Loss, Focal Loss, and the GCSL
reference implementation.
