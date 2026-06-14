# Phase 4E Compatibility Graph Repair Summary

Phase 4E repaired compressed support graphs using only Phase 2 support-bank
edges. No kNN/proximity/random/latent unsupported edges were added.

## Results

AntMaze `core_plus_bottleneck_budget120_H10` repaired from
`all_budget120_H10` with 200 added support edges:

| graph | method | coverage | min pair bridge | pair incompatible | min edge proxy | original uncertified frac | base cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | support_shortest_path | 0.566 | 0.043 | 0.161 | 0.105 | 0.327 | 50.551 |
| repaired | support_shortest_path | 0.642 | 0.041 | 0.157 | 0.089 | 0.354 | 50.962 |
| base | calibrated_compat_penalized | 0.566 | 0.113 | 0.033 | 0.247 | 0.048 | 55.511 |
| repaired | calibrated_compat_penalized | 0.642 | 0.114 | 0.037 | 0.218 | 0.061 | 56.032 |
| base | compat_threshold | 0.544 | 0.115 | 0.000 | 0.148 | 0.218 | 52.793 |
| repaired | compat_threshold | 0.620 | 0.116 | 0.000 | 0.117 | 0.271 | 53.361 |
| base | calibrated_compat_threshold | 0.544 | 0.146 | 0.000 | 0.261 | 0.043 | 56.024 |
| repaired | calibrated_compat_threshold | 0.620 | 0.149 | 0.000 | 0.229 | 0.059 | 56.935 |

Scene `core_plus_bottleneck_budget192_H5` repaired from `all_budget192_H5`
with 500 added support edges:

| graph | method | coverage | min pair bridge | pair incompatible | min edge proxy | original uncertified frac | base cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | support_shortest_path | 0.160 | 0.000 | 0.906 | 0.111 | 0.572 | 10.796 |
| repaired | support_shortest_path | 0.510 | 0.007 | 0.908 | 0.035 | 0.722 | 12.243 |
| base | calibrated_compat_penalized | 0.160 | 0.036 | 0.748 | 0.155 | 0.364 | 11.459 |
| repaired | calibrated_compat_penalized | 0.510 | 0.020 | 0.726 | 0.049 | 0.511 | 13.293 |
| base | compat_threshold | 0.150 | 0.116 | 0.000 | 0.175 | 0.261 | 20.671 |
| repaired | compat_threshold | 0.480 | 0.107 | 0.000 | 0.041 | 0.498 | 23.532 |
| base | calibrated_compat_threshold | 0.150 | 0.087 | 0.000 | 0.273 | 0.063 | 22.267 |
| repaired | calibrated_compat_threshold | 0.480 | 0.096 | 0.000 | 0.072 | 0.250 | 25.310 |

Repair edge summaries:

- AntMaze: 200 repair edges, 100 touched nodes, median support 243.5 segments,
  mean median horizon 7.44.
- Scene: 500 repair edges, 199 touched nodes, median support 24 segments, mean
  median horizon 3.47.

## Analysis

AntMaze repair is clean: adding support-bank edges improves coverage by 0.076
for both shortest-path and strict compatibility-threshold planning. The strict
threshold planner keeps pair incompatible fraction at 0.000 while moving from
0.544 to 0.620 coverage.

Scene repair is the clearest structural result so far. Strict compatibility
threshold coverage rises from 0.150 to 0.480 while keeping pair incompatible
fraction at 0.000. This means compatible paths exist in the broader support
bank, but the compressed graph discarded too much structure.

The trade-off is that added edges do not yet have Phase 3E/4C calibration.
Therefore repaired Scene paths have lower minimum edge proxy and higher original
uncertified edge fraction. The next step should certify repair-bank edges rather
than treating the new coverage as execution-ready.

## Conclusion

Phase 4E identifies a real breakthrough direction: support-only graph repair can
recover compatibility-safe coverage without unsupported shortcuts. The best next
algorithmic step is to run offline certification/calibration on selected repair
edges and then jointly optimize compression, certification, and compatibility.

These results remain offline graph-layer evidence and do not prove rollout
success.

