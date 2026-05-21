# Round 002 Prior Evidence Reclassification

Baseline certification failed for the medium GAS backbone, so no Stage19-Round001 result is promoted to a causal failure-mode or same-backbone mechanism claim.

## Summary

| evidence class | allowed claim level | grouped rows |
| --- | --- | ---: |
| E0_SMOKE_ONLY | SMOKE_ONLY | 40130 |
| FALLBACK_PROTOCOL_ONLY | PROTOCOL_DEBUG_ONLY | 84 |
| PROTOCOL_DEBUG_ONLY | PROTOCOL_DEBUG_ONLY | 3029 |

## Downgrade Rules Applied

- Reduced training or smoke/protocol budgets are E0_SMOKE_ONLY.
- Uncertified baseline rows are PROTOCOL_DEBUG_ONLY.
- Direct-goal/progress fallback rows are fallback-protocol evidence, not planner evidence.
- Same-backbone mechanism claims are blocked until both baseline and adapter certification pass.

## Explicit Decision

All Stage19-Round001 BARS results are downgraded to smoke/protocol evidence for scientific interpretation purposes.
