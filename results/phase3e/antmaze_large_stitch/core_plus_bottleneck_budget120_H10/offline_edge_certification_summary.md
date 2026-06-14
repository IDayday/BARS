# Phase 3E Offline Edge Certification Summary

This is a reset-free offline proxy. It does not run environment rollout and
is not equivalent to online option execution success.

The goal is to rank and filter data-supported option edges using heldout
trajectory support, GCBC action-fitting proxy, simple behavior/OOD risk,
and Phase 2 compatibility context.

Edges: `582`
Certified offline edges: `40`
Certified offline rate: `0.0687285`
Mean proxy score: `0.225212`

Offline action MSE and proxy scores are risk signals only. Rollout
validation remains Phase 3C/3F work once an environment is available.

## Top Edges By Proxy Score

| edge_id | src | dst | proxy | heldout_lcb | action_mse | certified |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 175 | 127 | 106 | 0.538899 | 0.685307 | 0.00321083 | True |
| 427 | 273 | 292 | 0.493344 | 0.574751 | 0.00897683 | True |
| 546 | 372 | 390 | 0.474871 | 0.569906 | 0.0058575 | True |
| 250 | 208 | 187 | 0.471454 | 0.427096 | 0.00265258 | True |
| 322 | 222 | 204 | 0.462862 | 0.486865 | 0.0029536 | True |
| 137 | 86 | 67 | 0.45459 | 0.464996 | 0.00489093 | True |
| 407 | 252 | 231 | 0.448595 | 0.479807 | 0.00407952 | True |
| 441 | 293 | 272 | 0.434537 | 0.465984 | 0.00694325 | True |
| 152 | 106 | 146 | 0.428271 | 0.375528 | 0.00125006 | True |
| 73 | 27 | 46 | 0.424812 | 0.433333 | 0.00776408 | True |
