# Round 003 Adapter Certification

- Adapter certification: PASS_ADAPTER_CERTIFICATION
- The aggregate PASS means at least one official-artifact env has row-level PASS; diagnostics are only unlocked on row-level PASS envs.
- PASS_ADAPTER_CERTIFICATION requires the same official checkpoint, graph, TDR/policy, env, task IDs, goal/start sampling, success threshold, max episode length, seed, and episode count, with abs(adapter_gap_pp) <= 2.0.

| env | official pp | adapter pp | gap pp | protocol match | status | reason |
| --- | ---: | ---: | ---: | --- | --- | --- |
| antmaze-giant-stitch-v0 | 92.0 | 0.0 | -92.0 | True | FAIL_ADAPTER_MISMATCH | adapter gap exceeds 2pp or protocol equality check failed |
| antmaze-large-explore-v0 | 96.8 | 90.4 | -6.3999999999999915 | True | FAIL_ADAPTER_MISMATCH | adapter gap exceeds 2pp or protocol equality check failed |
| scene-play-v0 | 79.60000000000001 | 80.80000000000001 | 1.2000000000000028 | True | PASS_ADAPTER_CERTIFICATION | adapter gap and all protocol equality checks pass |
