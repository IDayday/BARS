# Phase 3E Offline Edge Certification Summary

This is a reset-free offline proxy. It does not run environment rollout and
is not equivalent to online option execution success.

The goal is to rank and filter data-supported option edges using heldout
trajectory support, GCBC action-fitting proxy, simple behavior/OOD risk,
and Phase 2 compatibility context.

Edges: `1897`
Certified offline edges: `209`
Certified offline rate: `0.110174`
Mean proxy score: `0.162723`

Offline action MSE and proxy scores are risk signals only. Rollout
validation remains Phase 3C/3F work once an environment is available.

## Top Edges By Proxy Score

| edge_id | src | dst | proxy | heldout_lcb | action_mse | certified |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 779 | 176 | 462 | 0.572868 | 0.862019 | 0.00923553 | True |
| 309 | 63 | 146 | 0.558864 | 0.845356 | 0.0114291 | True |
| 737 | 171 | 288 | 0.54051 | 0.771898 | 0.00911684 | True |
| 681 | 164 | 288 | 0.523052 | 0.742422 | 0.0102482 | True |
| 1381 | 347 | 207 | 0.513557 | 0.824115 | 0.0160603 | True |
| 1524 | 385 | 85 | 0.48038 | 0.678721 | 0.0104022 | True |
| 37 | 6 | 245 | 0.480169 | 0.72246 | 0.0109619 | True |
| 291 | 58 | 389 | 0.478861 | 0.784683 | 0.0206284 | True |
| 1284 | 317 | 176 | 0.476329 | 0.467373 | 0.00857834 | True |
| 923 | 230 | 288 | 0.472309 | 0.534491 | 0.0110288 | True |
