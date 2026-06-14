# Phase 4A Risk-Aware Offline Planning Summary

This is a reset-free offline planning result. It does not run rollout
and does not claim policy execution success.

Best path coverage method: `support_shortest_path`
Best mean minimum edge proxy method: `proxy_threshold`
Lowest uncertified edge fraction method: `proxy_penalized`

| method | coverage | mean min proxy | uncertified frac | base cost | graph edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.566 | 0.060 | 0.924 | 50.551 | 582 |
| certified_only | 0.000 | nan | nan | nan | 40 |
| proxy_threshold | 0.204 | 0.260 | 0.877 | 58.100 | 337 |
| proxy_penalized | 0.566 | 0.128 | 0.874 | 52.575 | 582 |

Interpretation: higher proxy scores and lower uncertified fractions are
offline risk indicators only. They are not calibrated rollout success
probabilities.
