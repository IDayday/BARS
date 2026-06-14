# Phase 4F Repair-Edge Certification Summary

Phase 4F adds conservative transfer-proxy certification for Phase 4E repair
edges. No environment rollout, policy training, kNN edge creation, or proximity
shortcutting was used.

## Repair Certification Diagnostics

AntMaze:

- Repair edges: 200
- Transfer-certified repair edges: 182
- Transfer-certified rate: 0.910
- Mean calibrated repair reliability: 0.408
- Median calibrated repair reliability: 0.428

Scene:

- Repair edges: 500
- Transfer-certified repair edges: 397
- Transfer-certified rate: 0.794
- Mean calibrated repair reliability: 0.320
- Median calibrated repair reliability: 0.277

Repair edges do not have true heldout support labels in this phase. Their
`heldout_support_lcb` field is a conservative support-transfer proxy used for
planner compatibility.

## Results

AntMaze repaired graph:

| method | coverage | min edge proxy | uncertified frac | pair incompatible | repair edge frac | repair certified frac | base cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.642 | 0.105 | 0.265 | 0.157 | 0.041 | 0.852 | 50.962 |
| calibrated_compat_penalized | 0.642 | 0.259 | 0.023 | 0.026 | 0.131 | 0.983 | 56.120 |
| compat_threshold | 0.620 | 0.146 | 0.156 | 0.000 | 0.059 | 0.927 | 53.361 |
| calibrated_compat_threshold | 0.620 | 0.277 | 0.014 | 0.000 | 0.161 | 0.984 | 56.917 |

Scene repaired graph:

| method | coverage | min edge proxy | uncertified frac | pair incompatible | repair edge frac | repair certified frac | base cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.510 | 0.112 | 0.380 | 0.908 | 0.302 | 0.862 | 12.243 |
| calibrated_compat_penalized | 0.510 | 0.160 | 0.227 | 0.712 | 0.328 | 0.894 | 13.075 |
| compat_threshold | 0.480 | 0.143 | 0.173 | 0.000 | 0.252 | 0.931 | 23.532 |
| calibrated_compat_threshold | 0.480 | 0.230 | 0.044 | 0.000 | 0.401 | 0.936 | 25.711 |

## Comparison To Phase 4E

AntMaze `calibrated_compat_threshold`:

- Coverage remains 0.620.
- Mean minimum edge proxy improves from 0.229 to 0.277.
- Current uncertified edge fraction improves from 0.059 to 0.014.
- Pair incompatible fraction stays 0.000.

Scene `calibrated_compat_threshold`:

- Coverage remains 0.480.
- Mean minimum edge proxy improves from 0.072 to 0.230.
- Current uncertified edge fraction improves from 0.250 to 0.044.
- Pair incompatible fraction stays 0.000.

## Analysis

Phase 4F keeps Phase 4E's structural coverage gain while recovering much of the
planner-facing reliability lost when repair edges were treated as uncertified.
The strongest Scene result is `calibrated_compat_threshold`: it preserves the
large compatibility-safe coverage jump from 0.150 to 0.480 and reduces current
uncertified edge fraction to 0.044.

The remaining caveat is important: repair-edge scores are transfer proxies, not
true heldout GCBC or rollout labels. The next step should compute direct
Phase 3E-style policy likelihood for selected repair edges, then run closed-loop
evaluation once environment dependencies are available.

