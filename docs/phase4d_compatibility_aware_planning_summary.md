# Phase 4D Compatibility-Aware Planning Summary

Phase 4D was run on the same support-only Phase 2 graphs and Phase 4C
calibrated edge certifications. No environment rollout or new policy training
was used.

## Results

AntMaze `core_plus_bottleneck_budget120_H10`:

| method | coverage | min pair bridge | pair incompatible | min edge proxy | original uncertified frac | base cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.566 | 0.043 | 0.161 | 0.105 | 0.327 | 50.551 |
| calibrated_edge_penalized | 0.566 | 0.098 | 0.050 | 0.255 | 0.037 | 55.478 |
| compat_penalized | 0.566 | 0.070 | 0.105 | 0.117 | 0.268 | 52.022 |
| calibrated_compat_penalized | 0.566 | 0.113 | 0.033 | 0.247 | 0.048 | 55.511 |
| compat_threshold | 0.544 | 0.115 | 0.000 | 0.148 | 0.218 | 52.793 |
| calibrated_compat_threshold | 0.544 | 0.146 | 0.000 | 0.261 | 0.043 | 56.024 |

Scene `core_plus_bottleneck_budget192_H5`:

| method | coverage | min pair bridge | pair incompatible | min edge proxy | original uncertified frac | base cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.160 | 0.000 | 0.906 | 0.111 | 0.572 | 10.796 |
| calibrated_edge_penalized | 0.160 | 0.006 | 0.758 | 0.163 | 0.332 | 11.424 |
| compat_penalized | 0.160 | 0.037 | 0.641 | 0.100 | 0.596 | 11.965 |
| calibrated_compat_penalized | 0.160 | 0.036 | 0.748 | 0.155 | 0.364 | 11.459 |
| compat_threshold | 0.150 | 0.116 | 0.000 | 0.175 | 0.261 | 20.671 |
| calibrated_compat_threshold | 0.150 | 0.087 | 0.000 | 0.273 | 0.063 | 22.267 |

Pair table diagnostics:

- AntMaze has 3191 adjacent edge pairs, strict-compatible rate 0.661, and mean
  termination bridge coverage 0.236.
- Scene has 21876 adjacent edge pairs, strict-compatible rate 0.337, median
  termination bridge coverage 0.000, and mean termination bridge coverage 0.103.

## Analysis

AntMaze benefits from combining calibrated edge reliability and pair
compatibility. `calibrated_compat_penalized` preserves coverage at 0.566 while
reducing pair incompatible fraction from 0.161 to 0.033 and improving minimum
pair bridge coverage from 0.043 to 0.113. The hard threshold variants eliminate
incompatible adjacent pairs but lose 0.022 absolute coverage.

Scene confirms that compatibility is the dominant remaining bottleneck. The
support shortest path has high graph coverage relative to other methods, but
its reachable paths have pair incompatible fraction 0.906. Hard compatibility
thresholding reduces that to 0.000, but coverage drops to 0.150 and base path
cost roughly doubles. This is evidence that Scene needs either better node/edge
selection or compatibility-aware subgoal switching before rollout claims would
be credible.

## Conclusion

Phase 4D strengthens the offline graph-layer evidence: support-certified edges
are not enough; adjacent-edge composability must be optimized explicitly.
The best default after this phase is:

- AntMaze: `calibrated_compat_penalized` when preserving coverage matters, or
  `calibrated_compat_threshold` for stricter composability diagnostics.
- Scene: `compat_threshold`/`calibrated_compat_threshold` are useful as risk
  filters, but their cost increase shows the graph needs structural repair.

These results do not prove policy execution success. Closed-loop validation
still requires environment availability and rollout evaluation.

