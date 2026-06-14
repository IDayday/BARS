# Phase 4C Edge Risk Calibration Summary

Phase 4C is reset-free and offline-only. It calibrates edge-risk components and
reruns the Phase 4B support-only planner sweep. It does not train a policy, does
not run environment rollout, and does not claim online success.

Design note:

- `docs/phase4c_edge_risk_calibration_design.md`

Commands run:

```bash
python scripts/run_phase4c_edge_risk_calibration.py --config configs/phase4c_edge_risk_calibration_antmaze.yaml
python scripts/run_phase4c_edge_risk_calibration.py --config configs/phase4c_edge_risk_calibration_scene.yaml
pytest -q tests/test_phase4a_synthetic.py tests/test_phase4b_synthetic.py tests/test_phase4c_synthetic.py
```

Test result: `9 passed`.

## Calibration Method

Phase 4C replaces the Phase 3E aggregate proxy with a conservative geometric
score over:

- heldout support LCB;
- GCBC policy fitting score;
- behavior support, defined as `1 - edge_ood_score`;
- compatibility reliability;
- support diversity from unique starts and episodes.

The calibrated score is still an offline proxy. Heldout-support diagnostics are
pseudo-label checks, not rollout success calibration.

## AntMaze Stitch

Run:

- `results/phase4c/antmaze_large_stitch/core_plus_bottleneck_budget120_H10/`

Score diagnostics:

| metric | original proxy | calibrated reliability |
| --- | ---: | ---: |
| mean score | 0.225 | 0.316 |
| median score | 0.264 | 0.351 |
| Spearman vs heldout support rate | 0.732 | 0.688 |
| Pearson vs heldout support rate | 0.634 | 0.557 |
| Brier vs heldout support binary | 0.490 | 0.388 |

Certification scale:

| metric | value |
| --- | ---: |
| original certified edges | 40 / 582 |
| calibrated certified edges | 434 / 582 |

Recommended calibrated planner:

| metric | Phase 4B recommended | Phase 4C recommended |
| --- | ---: | ---: |
| coverage | 0.544 | 0.566 |
| mean min edge score | 0.217 | 0.273 |
| original uncertified edge fraction | 0.826 | 0.013 |
| base path cost | 55.809 | 57.895 |
| high-OOD fraction | 0.298 | 0.242 |
| high-incompat fraction | 0.150 | 0.109 |

Interpretation:

- Calibrated reliability improves the pseudo-label Brier score substantially.
- It restores full support-shortest-path coverage while selecting paths that are
  almost entirely originally certified under the Phase 3E evidence.
- Spearman correlation with heldout support rate drops slightly, so this is not
  a pure support-rank improvement. The gain comes from conservative component
  composition and planner-facing certification.

## Scene Play

Run:

- `results/phase4c/scene_play/core_plus_bottleneck_budget192_H5/`

Score diagnostics:

| metric | original proxy | calibrated reliability |
| --- | ---: | ---: |
| mean score | 0.163 | 0.236 |
| median score | 0.219 | 0.245 |
| Spearman vs heldout support rate | 0.754 | 0.757 |
| Pearson vs heldout support rate | 0.675 | 0.628 |
| Brier vs heldout support binary | 0.404 | 0.350 |

Certification scale:

| metric | value |
| --- | ---: |
| original certified edges | 209 / 1897 |
| calibrated certified edges | 935 / 1897 |

Recommended calibrated planner:

| metric | Phase 4B recommended | Phase 4C recommended |
| --- | ---: | ---: |
| coverage | 0.160 | 0.160 |
| mean min edge score | 0.092 | 0.178 |
| original uncertified edge fraction | 0.733 | 0.156 |
| base path cost | 11.706 | 12.762 |
| high-OOD fraction | 0.317 | 0.188 |
| high-incompat fraction | 0.793 | 0.792 |

Interpretation:

- Calibrated reliability improves both Spearman support ranking and Brier
  pseudo-label score slightly.
- It keeps coverage fixed while sharply reducing original uncertified edge
  exposure.
- High incompatibility remains nearly unchanged. Scene's remaining bottleneck is
  compatibility, not only scalar edge reliability.

## Cross-Dataset Pattern

- Component-wise calibration is useful. It reduces pseudo-label Brier on both
  datasets and improves planner path risk.
- Planner recommendations still prefer decomposed penalties over high aggregate
  risk weight.
- AntMaze now has a strong full-coverage offline-risk result.
- Scene still needs better compatibility modeling; score calibration alone does
  not solve path composability.

## Next Direction

The next useful step is Phase 4D: compatibility-aware path planning.

Instead of treating edge risk independently, Phase 4D should add adjacent-edge
transition risk into path planning:

- use Phase 2.2 termination bridge coverage between consecutive edges;
- penalize or filter paths with high adjacent incompatibility;
- report path-level compatibility alongside calibrated reliability;
- compare against Phase 4C recommended paths.

This remains offline until environment preflight enables rollout.
