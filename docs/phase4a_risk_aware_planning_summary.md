# Phase 4A Risk-Aware Offline Planning Summary

Phase 4A evaluates reset-free risk-aware planning over Phase 2
support-certified option edges. It does not create kNN/proximity/random edges,
does not train a new policy, and does not run environment rollout.

Design note:

- `docs/phase4a_risk_aware_planning_design.md`

Commands run:

```bash
python scripts/run_phase4a_risk_aware_planning.py --config configs/phase4a_risk_aware_antmaze.yaml
python scripts/run_phase4a_risk_aware_planning.py --config configs/phase4a_risk_aware_scene.yaml
pytest -q tests/test_phase4a_synthetic.py
```

Test result: `3 passed`.

## Planner Variants

- `support_shortest_path`: Phase 2 cost-only support graph baseline.
- `certified_only`: hard filter to Phase 3E certified edges.
- `proxy_threshold`: hard filter by `edge_proxy_score` and
  `heldout_support_lcb`.
- `proxy_penalized`: all support edges retained, but edge cost is multiplied by
  offline risk penalties from proxy score, OOD score, incompatibility, and
  uncertified status.

All variants are support-only: no unsupported graph edge is introduced.

## AntMaze Stitch

Run:

- `results/phase4a/antmaze_large_stitch/core_plus_bottleneck_budget120_H10/`

| method | coverage | mean min proxy | uncertified frac | low proxy frac | mean OOD | base cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.566 | 0.060 | 0.924 | 0.484 | 0.530 | 50.551 |
| certified_only | 0.000 | NaN | NaN | NaN | NaN | NaN |
| proxy_threshold | 0.204 | 0.260 | 0.877 | 0.000 | 0.394 | 58.100 |
| proxy_penalized | 0.566 | 0.128 | 0.874 | 0.334 | 0.439 | 52.575 |

Common-reachable change versus `support_shortest_path`:

| method | coverage delta | base cost delta | min proxy delta | uncertified fraction delta |
| --- | ---: | ---: | ---: | ---: |
| proxy_threshold | -0.362 | +10.229 | +0.202 | -0.017 |
| proxy_penalized | +0.000 | +2.024 | +0.069 | -0.050 |

Interpretation:

- Hard certification is too sparse for AntMaze at the current threshold: 40
  certified edges do not preserve any query connectivity.
- `proxy_threshold` produces cleaner paths but drops coverage from 0.566 to
  0.204.
- `proxy_penalized` preserves baseline coverage while improving the weakest
  edge proxy and reducing uncertified/low-proxy/OOD path exposure at modest
  extra base cost.

## Scene Play

Run:

- `results/phase4a/scene_play/core_plus_bottleneck_budget192_H5/`

| method | coverage | mean min proxy | uncertified frac | low proxy frac | mean OOD | base cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.160 | 0.065 | 0.984 | 0.574 | 0.462 | 10.796 |
| certified_only | 0.010 | 0.351 | 0.000 | 0.000 | 0.300 | 14.884 |
| proxy_threshold | 0.130 | 0.275 | 0.822 | 0.000 | 0.254 | 16.750 |
| proxy_penalized | 0.160 | 0.104 | 0.855 | 0.404 | 0.356 | 11.338 |

Common-reachable change versus `support_shortest_path`:

| method | coverage delta | base cost delta | min proxy delta | uncertified fraction delta |
| --- | ---: | ---: | ---: | ---: |
| certified_only | -0.150 | +5.382 | +0.044 | -1.000 |
| proxy_threshold | -0.030 | +5.943 | +0.195 | -0.159 |
| proxy_penalized | +0.000 | +0.542 | +0.040 | -0.129 |

Interpretation:

- Scene has enough certified/proxy-threshold edges to preserve a small amount of
  connectivity, but hard filtering still reduces coverage.
- `proxy_threshold` has a strong risk-cleaning effect, but at higher path cost
  and lower coverage.
- `proxy_penalized` again preserves baseline coverage while improving proxy and
  uncertified-edge exposure with small extra base cost.

## Current Conclusion

The first Phase 4A attempt supports a narrow offline planning conclusion:

- Hard certification is a useful diagnostic but can collapse graph connectivity.
- Soft risk penalization is a better default for preserving task coverage while
  reducing path risk.
- The best next planner should likely combine soft risk penalties with a mild
  hard floor on extreme low-proxy or zero-heldout-support edges, then sweep the
  penalty weights.

This is not an execution result. The output is an offline path-selection proxy
that should feed later closed-loop evaluation once environment preflight and
reset or natural-start rollout are available.
