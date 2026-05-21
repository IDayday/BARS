# Round 002 Public Target Lookup

## Sources
- Public target table: https://openreview.net/pdf?id=73EwiOrN8W
- Official GAS code and command templates: https://github.com/qortmdgh4141/GAS
- Official GAS checkpoint listing: https://huggingface.co/qortmdgh4141/GAS/tree/main

## Protocol Extracted
- Metric: normalized return in percentage points.
- Evaluation: five test-time goals, 50 rollouts per goal, averaged over 4 seeds.
- GAS state-task training command: 1,000,000 TDR steps and 1,000,000 policy steps with batch size 1024.
- Certification lower bound: public_mean_pp - max(2 * public_std_pp, 5pp).

## Exact Target Status
- exact_public_target_missing_rows: 0
- gas_official_artifact_unavailable_rows: 3
- gas_undertrained_local_rows: 2

## GAS Rows

| env | public mean +/- std | lower bound | official artifact | local source | local steps | initial status |
| --- | ---: | ---: | --- | --- | ---: | --- |
| antmaze-medium-stitch-v0 | 98.1 +/- 1.2 | 93.1 | False | reduced_train | 100000 | FAIL_UNDERTRAINED_BASELINE |
| antmaze-medium-navigate-v0 | 96.3 +/- 1.3 | 91.3 | False | reduced_train | 100000 | FAIL_UNDERTRAINED_BASELINE |
| antmaze-large-stitch-v0 | 96.3 +/- 0.9 | 91.3 | False | unknown | None | SKIP_ARTIFACT_UNAVAILABLE |
| antmaze-large-navigate-v0 | 93.2 +/- 0.5 | 88.2 | False | unknown | None | SKIP_ARTIFACT_UNAVAILABLE |
| antmaze-giant-stitch-v0 | 88.3 +/- 3.6 | 81.1 | True | official_checkpoint | 1000000 | READY_FOR_OFFICIAL_EVAL |
| antmaze-giant-navigate-v0 | 77.6 +/- 2.9 | 71.8 | True | unknown | None | SKIP_ARTIFACT_UNAVAILABLE |
| antmaze-large-explore-v0 | 94.2 +/- 3.0 | 88.2 | True | official_checkpoint | 1000000 | READY_FOR_OFFICIAL_EVAL |
| scene-play-v0 | 73.6 +/- 8.0 | 57.6 | True | official_checkpoint | 1000000 | READY_FOR_OFFICIAL_EVAL |
