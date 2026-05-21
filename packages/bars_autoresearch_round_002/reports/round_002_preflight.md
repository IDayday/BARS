# Round 002 Preflight

- generated_at: 2026-05-20T07:03:37.614123+00:00
- pwd: /root/remote/BARS
- git_commit: 7e860d7a4cdaefc0977574a7f1c7b49bd5a6e9b5
- python: 3.9.25

## Git Status

```text
M bars/data/ogbench_dataset.py
 M bars/external/gas_artifacts.py
 M bars/gas_bars/evaluate.py
 M bars/gas_bars/planner.py
 M external_src/GAS/O_utils/env_utils.py
 M scripts/stage24_analyze.py
 M scripts/stage24_oracle_headroom_scan.sh
 M scripts/stage24_run_reachability_confirm.sh
?? AGENTS.md
?? BARS_AUTONOMOUS_RESEARCH_LOOP_BASELINE_FIRST.md
?? configs/stage25_local_drift_v2.json
?? configs/stage25_oracle_scan_matrix.json
?? configs/stage25_reachability_closing.json
?? packages/
?? reports/round_001_failure_atlas_all_variants.csv
?? reports/round_001_failure_atlas_summary.csv
?? reports/round_001_gate_status.json
?? reports/round_001_label_integrity.json
?? reports/round_001_metrics_summary.csv
?? reports/round_001_next_plan.md
?? reports/round_001_reflection.md
?? reports/round_002_adapter_mismatch_report.md
?? reports/round_002_bars_adapter_eval.csv
?? reports/round_002_baseline_cards.jsonl
?? reports/round_002_baseline_registry.csv
?? reports/round_002_gas_official_eval.csv
?? reports/round_002_official_vs_adapter.csv
?? reports/round_002_prior_evidence_reclassification.csv
?? reports/round_002_prior_evidence_reclassification.md
?? reports/round_002_public_target_lookup.md
?? research_state/
?? rounds/
?? scripts/analyze_baseline_certification.py
?? scripts/autoresearch_analyze_round.py
?? scripts/autoresearch_init.py
?? scripts/autoresearch_loop.sh
?? scripts/autoresearch_package_round.sh
?? scripts/autoresearch_plan_next.py
?? scripts/autoresearch_run_round.py
?? scripts/build_baseline_registry.py
?? scripts/certify_gas_baseline.sh
?? scripts/compare_official_vs_bars_adapter.py
?? scripts/fetch_public_baseline_targets.py
?? scripts/package_autoresearch_round.sh
?? scripts/package_stage25_results.sh
?? scripts/reclassify_prior_evidence.py
?? scripts/run_official_gas_eval.py
?? scripts/stage24_env_mirrors.sh
?? scripts/stage24_prefetch_ogbench_datasets.sh
?? scripts/stage25_analyze.py
?? scripts/stage25_boundary_coverage_repair.py
?? scripts/stage25_d4rl_protocol_audit.py
?? scripts/stage25_enrich_failure_atlas_all_variants.py
?? scripts/stage25_oracle_headroom_scan_v2.sh
?? scripts/stage25_rank_oracle_envs.py
?? scripts/stage25_reachability_closing.sh
?? scripts/stage25_run_local_drift_v2.sh
?? scripts/verify_baseline_card.py
?? stage24_results_config_package_20260520_final/
```

## Environment

- WANDB_MODE=disabled
- WANDB_DISABLED=true
- D4RL_SUPPRESS_IMPORT_ERROR=1

## Existing Artifacts

- artifacts/gas contains local GAS manifests for medium-stitch, medium-navigate, giant-stitch, large-explore, and scene-play when present.
- medium-stitch and medium-navigate manifests are local trained 100000-step artifacts, not official Hugging Face checkpoints.
