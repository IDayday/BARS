# Round 003 Prior Evidence Reclassification

Round003 adds official-artifact baseline and adapter certification rows. Prior rows are not upgraded unless the exact env and artifact lineage are certified.

- Certified Round003 envs: scene-play-v0
- Medium rows remain downgraded because their local GAS artifacts are 100000-step artifacts, not public-quality 1000000-step artifacts.
- Prior giant/large/scene diagnostics are not automatically upgraded unless their artifact lineage is the same official checkpoint/graph certified in Round003.

## Summary

| evidence class | allowed claim level | grouped rows |
| --- | --- | ---: |
| E0_SMOKE_ONLY | SMOKE_ONLY | 40130 |
| E2_CERTIFIED_BASELINE_DIAGNOSTIC | CERTIFIED_BASELINE_ONLY | 3 |
| E3_SAME_BACKBONE_METHOD_COMPARISON | SAME_BACKBONE_MECHANISM_ON_CERTIFIED_ENV | 1 |
| FALLBACK_PROTOCOL_ONLY | PROTOCOL_DEBUG_ONLY | 84 |
| PROTOCOL_DEBUG_ONLY | PROTOCOL_DEBUG_ONLY | 3031 |
