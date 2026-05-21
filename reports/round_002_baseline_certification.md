# Round 002 Baseline Certification

Primary question: Can we certify a public-quality strong baseline and BARS adapter before interpreting any BARS failure modes?

## Gate Result

- Baseline registry: PASS_BASELINE_REGISTRY
- Baseline certification: FAIL_UNDERTRAINED_BASELINE
- Adapter certification: SKIP_NO_OFFICIAL_EVAL
- Scientific interpretation: HOLD_SCIENTIFIC_INTERPRETATION_BASELINE_UNCERTIFIED

## GAS Certification Targets

| env | artifact source | train steps | required | official score | lower bound pp | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| antmaze-medium-stitch-v0 | trained | 100000 | 1000000 | 0.966 | 93.1 | FAIL_UNDERTRAINED_BASELINE | local GAS checkpoint has 100000 train steps; public command requires 1000000 |
| antmaze-medium-stitch-v0 | unknown | 100000 | 1000000 |  | 93.1 | FAIL_UNDERTRAINED_BASELINE | local GAS checkpoint has 100000 train steps; public command requires 1000000 |
| antmaze-medium-stitch-v0 | unknown | 100000 | 1000000 |  | 93.1 | FAIL_UNDERTRAINED_BASELINE | local GAS checkpoint has 100000 train steps; public command requires 1000000 |
| antmaze-medium-stitch-v0 | unknown |  | 1000000 |  | 93.1 | SKIP_ARTIFACT_UNAVAILABLE | no complete local GAS policy/keygraph artifact |
| antmaze-medium-navigate-v0 | trained | 100000 | 1000000 | 0.9399999999999998 | 91.3 | FAIL_UNDERTRAINED_BASELINE | local GAS checkpoint has 100000 train steps; public command requires 1000000 |
| antmaze-medium-navigate-v0 | unknown | 100000 | 1000000 |  | 91.3 | FAIL_UNDERTRAINED_BASELINE | local GAS checkpoint has 100000 train steps; public command requires 1000000 |
| antmaze-medium-navigate-v0 | unknown | 100000 | 1000000 |  | 91.3 | FAIL_UNDERTRAINED_BASELINE | local GAS checkpoint has 100000 train steps; public command requires 1000000 |
| antmaze-medium-navigate-v0 | unknown |  | 1000000 |  | 91.3 | SKIP_ARTIFACT_UNAVAILABLE | no complete local GAS policy/keygraph artifact |

## Adapter Comparison

| env | adapter | gap pp | status | reason |
| --- | --- | ---: | --- | --- |
| antmaze-medium-stitch-v0 | gas_shortest_official_control | 1.4000000000000012 | SKIP_BASELINE_UNCERTIFIED | adapter comparison is protocol evidence only because baseline certification did not pass |
| antmaze-medium-stitch-v0 | gas_shortest_official_control |  | SKIP_BASELINE_UNCERTIFIED | adapter comparison is protocol evidence only because baseline certification did not pass |
| antmaze-medium-stitch-v0 | gas_shortest_official_control |  | SKIP_BASELINE_UNCERTIFIED | adapter comparison is protocol evidence only because baseline certification did not pass |
| antmaze-medium-stitch-v0 | gas_shortest_official_control |  | SKIP_BASELINE_UNCERTIFIED | adapter comparison is protocol evidence only because baseline certification did not pass |
| antmaze-medium-navigate-v0 | gas_shortest_official_control | -0.9999999999999787 | SKIP_BASELINE_UNCERTIFIED | adapter comparison is protocol evidence only because baseline certification did not pass |
| antmaze-medium-navigate-v0 | gas_shortest_official_control |  | SKIP_BASELINE_UNCERTIFIED | adapter comparison is protocol evidence only because baseline certification did not pass |
| antmaze-medium-navigate-v0 | gas_shortest_official_control |  | SKIP_BASELINE_UNCERTIFIED | adapter comparison is protocol evidence only because baseline certification did not pass |
| antmaze-medium-navigate-v0 | gas_shortest_official_control |  | SKIP_BASELINE_UNCERTIFIED | adapter comparison is protocol evidence only because baseline certification did not pass |

## Prior Evidence Reclassification

- grouped_rows: 336
- allowed_claim_level_counts: {"PROTOCOL_DEBUG_ONLY": 60, "SMOKE_ONLY": 276}

## What Can Be Concluded

- The public target registry is complete for the requested OGBench state tasks and required algorithms.
- The current medium GAS backbone cannot be certified because the available local artifacts are 100000-step trained artifacts while the public GAS command requires 1000000 steps.
- The repaired official-control adapter comparison remains protocol/debug evidence only until a certified baseline exists.

## What Cannot Be Concluded

- No BARS failure taxonomy, oracle-headroom, p_bridge, boundary, or integrated BARS result can be interpreted scientifically from this round.
- No same-backbone BARS mechanism claim is valid from Stage19-Round001 under the Round 002 gates.
