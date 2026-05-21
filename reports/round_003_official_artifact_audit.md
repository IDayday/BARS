# Round 003 Official GAS Artifact Audit

Evidence class: E1_BASELINE_REGISTRY. This audit is availability and lineage bookkeeping only.

## Summary

- Rows: 9
- Artifact status counts: `{"LOCAL_UNDERTRAINED": 2, "OFFICIAL_FULL_BUDGET": 3, "OFFICIAL_NOT_FOUND": 2, "OFFICIAL_PARTIAL_LOCAL": 2}`
- Medium stitch/navigate remain local 100000-step artifacts, below the public 1000000-step GAS budget.
- Official full-budget local artifacts are present for antmaze-giant-stitch-v0, antmaze-large-explore-v0, and scene-play-v0.
- antmaze-giant-navigate-v0 has a local keygraph and partial policy download only, so it is not certification-ready.

## Audit Rows

| env | public mean pp | lower bound pp | official artifact | local steps | status | action |
| --- | ---: | ---: | --- | ---: | --- | --- |
| antmaze-medium-stitch-v0 | 98.1 | 93.1 | not listed | 100000 | LOCAL_UNDERTRAINED | do_not_certify; acquire_official_artifact_or_run_full_budget_training |
| antmaze-medium-navigate-v0 | 96.3 | 91.3 | not listed | 100000 | LOCAL_UNDERTRAINED | do_not_certify; acquire_official_artifact_or_run_full_budget_training |
| antmaze-large-stitch-v0 | 96.3 | 91.3 | not listed |  | OFFICIAL_NOT_FOUND | official_artifact_unavailable; use_full_budget_training_plan |
| antmaze-large-navigate-v0 | 93.2 | 88.2 | not listed |  | OFFICIAL_NOT_FOUND | official_artifact_unavailable; use_full_budget_training_plan |
| antmaze-giant-stitch-v0 | 88.3 | 81.1 | https://huggingface.co/qortmdgh4141/GAS/tree/main/antmaze-giant-stitch | 1000000 | OFFICIAL_FULL_BUDGET | run_official_eval_then_adapter_certification |
| antmaze-giant-navigate-v0 | 77.6 | 71.8 | https://huggingface.co/qortmdgh4141/GAS/tree/main/antmaze-giant-navigate |  | OFFICIAL_PARTIAL_LOCAL | resume_or_redownload_official_artifact_before_certification |
| antmaze-large-explore-v0 | 94.2 | 88.2 | https://huggingface.co/qortmdgh4141/GAS/tree/main/antmaze-large-explore | 1000000 | OFFICIAL_FULL_BUDGET | run_official_eval_then_adapter_certification |
| scene-play-v0 | 73.6 | 57.599999999999994 | https://huggingface.co/qortmdgh4141/GAS/tree/main/scene-play | 1000000 | OFFICIAL_FULL_BUDGET | run_official_eval_then_adapter_certification |
| kitchen-partial-v0 |  |  | https://huggingface.co/qortmdgh4141/GAS/tree/main/kitchen-partial |  | OFFICIAL_PARTIAL_LOCAL | resume_or_redownload_official_artifact_before_certification |

## Direct Artifact URLs

| env | keygraph | params |
| --- | --- | --- |
| antmaze-giant-stitch-v0 | https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-giant-stitch/keygraph.pkl | https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-giant-stitch/params_1000000.pkl |
| antmaze-giant-navigate-v0 | https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-giant-navigate/keygraph.pkl | https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-giant-navigate/params_1000000.pkl |
| antmaze-large-explore-v0 | https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-large-explore/keygraph.pkl | https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-large-explore/params_1000000.pkl |
| scene-play-v0 | https://huggingface.co/qortmdgh4141/GAS/resolve/main/scene-play/keygraph.pkl | https://huggingface.co/qortmdgh4141/GAS/resolve/main/scene-play/params_1000000.pkl |
| kitchen-partial-v0 | https://huggingface.co/qortmdgh4141/GAS/resolve/main/kitchen-partial/keygraph.pkl | https://huggingface.co/qortmdgh4141/GAS/resolve/main/kitchen-partial/params_500000.pkl |
