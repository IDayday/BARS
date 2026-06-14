# Phase 4G Direct Repair-Edge Policy Evidence Summary

Phase 4G evaluates trained GCBC policies directly on selected repair-bank
segments. No environment rollout or additional policy training was used.

## Direct Repair-Edge Evidence

AntMaze used the 100000-step `core_plus_bottleneck` GCBC model:

- Direct-scored repair edges: 200 / 200
- Mean direct edge action MSE: 0.0554
- Median direct edge action MSE: 0.0521
- Mean direct policy support score: 0.372
- Transfer-to-direct policy score delta: -0.134
- Spearman transfer-vs-direct policy score: 0.055
- Direct certified rate: 0.905 versus transfer certified rate 0.910

Scene used the available 200-step smoke GCBC model:

- Direct-scored repair edges: 500 / 500
- Mean direct edge action MSE: 0.0382
- Median direct edge action MSE: 0.0239
- Mean direct policy support score: 0.561
- Transfer-to-direct policy score delta: +0.220
- Spearman transfer-vs-direct policy score: 0.547
- Direct certified rate: 0.870 versus transfer certified rate 0.794

The Scene numbers are weaker evidence than AntMaze because the available model
is a short smoke run, not a 100000-step trained policy.

## Planner Results

AntMaze direct-policy repaired graph:

| method | coverage | min edge proxy | uncertified frac | pair incompatible | repair edge frac | repair certified frac | base cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.642 | 0.105 | 0.265 | 0.157 | 0.041 | 0.852 | 50.962 |
| calibrated_compat_penalized | 0.642 | 0.258 | 0.023 | 0.027 | 0.129 | 0.983 | 56.074 |
| compat_threshold | 0.620 | 0.146 | 0.156 | 0.000 | 0.059 | 0.927 | 53.361 |
| calibrated_compat_threshold | 0.620 | 0.276 | 0.014 | 0.000 | 0.156 | 0.983 | 56.829 |

Scene direct-policy repaired graph:

| method | coverage | min edge proxy | uncertified frac | pair incompatible | repair edge frac | repair certified frac | base cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.510 | 0.118 | 0.357 | 0.908 | 0.302 | 0.955 | 12.243 |
| calibrated_compat_penalized | 0.510 | 0.173 | 0.196 | 0.708 | 0.325 | 0.956 | 13.159 |
| compat_threshold | 0.480 | 0.148 | 0.164 | 0.000 | 0.252 | 0.977 | 23.532 |
| calibrated_compat_threshold | 0.480 | 0.244 | 0.037 | 0.000 | 0.401 | 0.973 | 25.711 |

## Analysis

AntMaze direct policy evidence mostly preserves Phase 4F's conclusion, but
shows that transfer proxy was slightly optimistic for repair-edge policy
support. The weak Spearman correlation means transfer scores should not be used
as a precise ranking signal when direct GCBC evidence is available.

Scene direct scores are higher than transfer scores and preserve the large
Phase 4E compatibility-safe coverage gain. However, because Scene used a
200-step smoke model, this result should be treated as a smoke-level signal.

The best current offline planner remains `calibrated_compat_threshold`:

- AntMaze: coverage 0.620, pair incompatible fraction 0.000, uncertified
  fraction 0.014.
- Scene: coverage 0.480, pair incompatible fraction 0.000, uncertified fraction
  0.037.

## Remaining Gap

Direct action MSE is still offline supervised evidence. It does not measure
closed-loop edge execution, compounding errors, or task-level success. The next
step should either train stronger Scene GCBC models or unblock environment
preflight for closed-loop edge execution.

