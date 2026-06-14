# Phase 4B Calibrated Risk-Aware Planner Sweep Summary

Phase 4B is reset-free and offline-only. It sweeps support-only risk-aware
planner parameters and reports coverage/risk/cost trade-offs. It does not train
a new policy, does not run environment rollout, and does not claim online
success.

Design note:

- `docs/phase4b_risk_sweep_design.md`

Commands run:

```bash
python scripts/run_phase4b_risk_sweep.py --config configs/phase4b_risk_sweep_antmaze.yaml
python scripts/run_phase4b_risk_sweep.py --config configs/phase4b_risk_sweep_scene.yaml
pytest -q tests/test_phase4a_synthetic.py tests/test_phase4b_synthetic.py
```

Test result: `6 passed`.

## Sweep Scope

Each dataset used 480 `floor_proxy_penalized` configs:

- `risk_weight`: 0, 1, 2, 4, 8
- `ood_weight`: 0, 1
- `incompat_weight`: 0, 1
- `uncertified_weight`: 0, 1
- `proxy_floor`: 0, 0.05, 0.1, 0.2
- `heldout_support_lcb_floor`: 0, 0.01, 0.05

All candidate graphs are support-only. No kNN/proximity/random edge is added.

## AntMaze Stitch

Run:

- `results/phase4b/antmaze_large_stitch/core_plus_bottleneck_budget120_H10/`

Sweep rows: 480
Pareto rows: 212

Baselines:

| method | coverage | mean min proxy | uncertified frac | base cost |
| --- | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.566 | 0.060 | 0.924 | 50.551 |
| proxy_penalized | 0.566 | 0.128 | 0.874 | 52.575 |
| proxy_threshold | 0.204 | 0.260 | 0.877 | 58.100 |
| certified_only | 0.000 | NaN | NaN | NaN |

Recommended sweep config:

| metric | value |
| --- | ---: |
| method | floor_proxy_penalized_s0066 |
| coverage | 0.544 |
| mean min proxy | 0.217 |
| uncertified frac | 0.826 |
| base cost | 55.809 |
| graph edges | 489 |
| risk_weight | 0.0 |
| ood_weight | 1.0 |
| incompat_weight | 0.0 |
| uncertified_weight | 1.0 |
| proxy_floor | 0.1 |
| heldout_support_lcb_floor | 0.0 |
| Pareto | true |

Interpretation:

- A mild proxy floor plus OOD/uncertified penalties gives a strong offline
  trade-off: coverage drops slightly from 0.566 to 0.544, while mean minimum
  edge proxy rises from 0.060 to 0.217.
- Full-coverage Pareto configs exist. The best full-coverage configs improve
  mean minimum edge proxy to about 0.19 with base cost around 55.
- `risk_weight=0` appearing in the recommended config is informative: the
  decomposed OOD and uncertified penalties plus a proxy floor are more stable
  than directly treating the aggregate proxy score as a calibrated probability.

## Scene Play

Run:

- `results/phase4b/scene_play/core_plus_bottleneck_budget192_H5/`

Sweep rows: 480
Pareto rows: 182

Baselines:

| method | coverage | mean min proxy | uncertified frac | base cost |
| --- | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.160 | 0.065 | 0.984 | 10.796 |
| proxy_penalized | 0.160 | 0.104 | 0.855 | 11.338 |
| proxy_threshold | 0.130 | 0.275 | 0.822 | 16.750 |
| certified_only | 0.010 | 0.351 | 0.000 | 14.884 |

Recommended sweep config:

| metric | value |
| --- | ---: |
| method | floor_proxy_penalized_s0060 |
| coverage | 0.160 |
| mean min proxy | 0.092 |
| uncertified frac | 0.733 |
| base cost | 11.706 |
| graph edges | 1897 |
| risk_weight | 0.0 |
| ood_weight | 1.0 |
| incompat_weight | 0.0 |
| uncertified_weight | 1.0 |
| proxy_floor | 0.0 |
| heldout_support_lcb_floor | 0.0 |
| Pareto | true |

Interpretation:

- Scene has a harsher trade-off. Preserving coverage 0.160 still leaves many
  uncertified edges and high incompatibility exposure.
- Higher proxy-floor configs reach mean minimum edge proxy around 0.26, but
  coverage falls to about 0.15 and base cost rises to about 17.
- This suggests Scene bottleneck is no longer just planner weighting. The edge
  certification and compatibility signals need improvement before planning can
  give a much cleaner path set at the same coverage.

## Cross-Dataset Pattern

- Hard certification remains too sparse as a primary planner.
- Soft penalties plus mild floors dominate pure hard filtering for coverage.
- Decomposed risk terms matter. OOD and uncertified penalties repeatedly appear
  in recommended configs, while direct aggregate `risk_weight` is not always
  selected.
- AntMaze benefits more from proxy/support floors than Scene; Scene exposes a
  deeper compatibility/certification bottleneck.

## Next Direction

The next useful step is not another blind weight sweep. The better target is
Phase 4C: improve and calibrate edge risk components:

- separate heldout support, OOD, compatibility, and action-fit terms;
- produce a more stable per-edge reliability score;
- rerun Phase 4B with the calibrated score;
- only then feed selected paths to closed-loop evaluation when env preflight is
  available.

These are still offline conclusions. They do not prove that the GCBC policy can
execute the selected option paths.
