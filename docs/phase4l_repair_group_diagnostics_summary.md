# Phase 4L Direct Repair-Edge Group Diagnostics Summary

Phase 4L compares each Phase 4K loss-weighted checkpoint against the matched
same-seed `uniform_transition_none` baseline on the same Scene repair edges.
This is offline supervised direct repair-edge evidence, not rollout success.

## Command

```bash
python scripts/run_phase4l_repair_group_diagnostics.py --config configs/phase4l_repair_group_diagnostics_scene_H5_B192_3000.yaml
```

## Method-Level Result

| method | mean MSE delta | mean ratio | fraction improved | planner usage rate |
| --- | ---: | ---: | ---: | ---: |
| `loss_support_bottleneck_s03` | -0.000211 | 1.000 | 0.497 | 0.178 |
| `loss_support_s03` | +0.000023 | 1.025 | 0.456 | 0.178 |
| `loss_bottleneck_s03` | +0.000152 | 1.028 | 0.381 | 0.178 |

The combined support+bottleneck loss is still the only useful candidate.

## Where Combined Weighting Helps

For `loss_support_bottleneck_s03`:

| group | num edges | MSE delta | sample-weighted delta | fraction improved | planner usage |
| --- | ---: | ---: | ---: | ---: | ---: |
| low support | 259 | -0.000421 | -0.000251 | 0.533 | 0.081 |
| long horizon | 54 | -0.000417 | -0.000126 | 0.491 | 0.130 |
| high bottleneck | 240 | -0.000309 | -0.000082 | 0.513 | 0.138 |
| high compatibility | 250 | -0.000277 | -0.000022 | 0.500 | 0.272 |
| planner used | 89 | -0.000067 | +0.000060 | 0.444 | 1.000 |

The intended hard groups are exactly where the combined weighting helps most:
low-support, long-horizon, and high-bottleneck repair edges.

## Analysis

Phase 4L explains the modest Phase 4K aggregate gain. The combined loss weight
is not uniformly better; it shifts fitting toward harder repair edges. That is
the desired direction for a graph method whose remaining weakness is reliable
execution of sparse support-bank repair edges.

There is also a planner mismatch. The current `calibrated_compat_threshold`
planner uses only 89 / 500 repair edges. The planner-used group has a much
smaller unweighted MSE improvement and a slightly worse sample-weighted delta,
so some training-side gains are happening on repair edges the current planner
rarely uses.

## Next Step

The next useful experiment is not another blind sampler. It should either:

- replicate `loss_support_bottleneck_s03` on AntMaze and Scene H10/H25, or
- make planning/training interact more tightly, for example by upweighting
  planner-relevant low-support/high-bottleneck repair edges rather than all rare
  edges.

## Claim Boundary

This is reset-free offline supervised diagnostics. It does not show closed-loop
edge execution or online task success.
