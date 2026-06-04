# Stage30 Official GAS Reproduction Report

Status: OFFICIAL_GAS_BASELINE_CERTIFICATION.
Pre-Stage30 BARS/Stage28/Stage29 evidence: ARCHIVED_INTERNAL_EXPLORATION_NOT_GAS_EVIDENCE.
Interpretation rule: these rows are official GAS reproduction evidence; BARS_BASE, Stage28, and Stage29 are not used as GAS evidence.

## Source Identity

- official_repo_url: `https://github.com/qortmdgh4141/GAS.git`
- official_hf_repo: `qortmdgh4141/GAS`
- gas_repo_path: `external_src/GAS`
- official_repo_sha: `UNAVAILABLE_IN_VENDOR_COPY`
- gas_vendor_tree_sha256: `eb5cd4a3e69cbf1a4077650a466352341c8692013846425b62445f9052279cb6`

## Protocol

- evaluate_gas mode: `run`.
- Official planner, policy, graph, and action outputs are not modified.
- If `mode=cached`, the script records existing local official `policy/eval.csv` outputs and artifact hashes.
- If `mode=run`, official `evaluate_gas.py` is executed on copied keygraph/policy files so original artifacts are not overwritten.

## Success Summary

| env_name | seeds | success_mean | success_ci95_low | success_ci95_high |
| --- | --- | --- | --- | --- |
| antmaze-medium-navigate-v0 | 1 | 1.0000 | 1.0000 | 1.0000 |

## Files

- official_gas_eval.csv: `runs_stage30_official_gas/reproduction_run_smoke/official_gas_eval.csv`
- artifact_identity.csv: `runs_stage30_official_gas/reproduction_run_smoke/artifact_identity.csv`
- command_manifest.jsonl: `runs_stage30_official_gas/reproduction_run_smoke/command_manifest.jsonl`
