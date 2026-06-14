# Phase 4A Risk-Aware Offline Planning Summary

This is a reset-free offline planning result. It does not run rollout
and does not claim policy execution success.

Best path coverage method: `support_shortest_path`
Best mean minimum edge proxy method: `certified_only`
Lowest uncertified edge fraction method: `certified_only`

| method | coverage | mean min proxy | uncertified frac | base cost | graph edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| support_shortest_path | 0.160 | 0.065 | 0.984 | 10.796 | 1897 |
| certified_only | 0.010 | 0.351 | 0.000 | 14.884 | 209 |
| proxy_threshold | 0.130 | 0.275 | 0.822 | 16.750 | 793 |
| proxy_penalized | 0.160 | 0.104 | 0.855 | 11.338 | 1897 |

Interpretation: higher proxy scores and lower uncertified fractions are
offline risk indicators only. They are not calibrated rollout success
probabilities.
