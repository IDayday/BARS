# CAGE-CLP1 Repo Review

## Git State

- Branch: `codex/cage-mvp`
- Current commit: `1574580`
- Remote tracking: `origin/codex/cage-mvp`
- Note: the working tree contains unrelated stage32/stage35/results/logs changes that are not part of CLP1.

## GP0 Files Detected

- Report: `docs/cage_gp0_alignment_report.md`
- q_G summaries:
  - `results/cage_gp0/focused/antmaze_nav/qG_summary.json`
  - `results/cage_gp0/focused/antmaze_stitch/qG_summary.json`
  - `results/cage_gp0/focused/humanoid_large_nav/qG_summary.json`
- q_train summaries:
  - `results/cage_gp0/focused/antmaze_nav/qtrain_summary.json`
  - `results/cage_gp0/focused/antmaze_stitch/qtrain_summary.json`
  - `results/cage_gp0/focused/humanoid_large_nav/qtrain_summary.json`
- q_G/q_train comparison:
  - `results/cage_gp0/focused/compare/graph_policy_compare.md`
  - `results/cage_gp0/focused/compare/graph_policy_compare.json`
  - `results/cage_gp0/focused/compare/qG_pair_support.jsonl`

## CLP0 Files Detected

- Report: `docs/cage_clp0_closed_loop_report.md`
- Reset audit:
  - `results/cage_clp0/reset_audit/reset_capability.json`
  - `results/cage_clp0/reset_audit/reset_capability.md`
- q_G with StateRefs:
  - `results/cage_clp0/qg_with_state_refs/antmaze-giant-navigate-v0_seed42_qg_pairs.jsonl`
  - `results/cage_clp0/qg_with_state_refs/antmaze-giant-stitch-v0_seed42_qg_pairs.jsonl`
  - `results/cage_clp0/qg_with_state_refs/humanoidmaze-large-navigate-v0_seed44_qg_pairs.jsonl`
- Probe outputs:
  - `results/cage_clp0/probes/antmaze_nav_seed42_edge_probe.jsonl`
  - `results/cage_clp0/probes/antmaze_stitch_seed42_edge_probe.jsonl`
  - `results/cage_clp0/probes/humanoid_large_nav_seed44_edge_probe.jsonl`
- Contract dataset:
  - `results/cage_clp0/datasets/closed_loop_contracts.jsonl`
  - `results/cage_clp0/datasets/closed_loop_contracts_summary.md`

## Repair0 / Pilot0 Files Detected

- `docs/cage_pilot0_report.md`
- `docs/cage_pilot0_postmortem.md`
- `docs/cage_repair0_report.md`
- `results/cage_pilot0/minipilot_local_antmaze_nav/tables/minipilot_summary.md`
- `results/cage_pilot0/minipilot_local_antmaze_stitch/tables/minipilot_summary.md`
- `results/cage_pilot0/minipilot_local_humanoid_large_nav/tables/minipilot_summary.md`
- `results/cage_repair0/minipilot_antmaze_nav/tables/minipilot_summary.md`
- `results/cage_repair0/minipilot_antmaze_stitch/tables/minipilot_summary.md`
- `results/cage_repair0/minipilot_humanoid_large_nav/tables/minipilot_summary.md`

## Blockers / Missing

- CLP1 segment-capture outputs now exist under `results/cage_clp1/segment_capture/`.
- Candidate-aware segment-capture outputs now exist under `results/cage_clp1/segment_capture_candidate/`.
- Branchable probe outputs now exist under `results/cage_clp1/probes/` and `results/cage_clp1/probes_candidate/`.
- Contract oracle outputs now exist under `results/cage_clp1/oracle/` and `results/cage_clp1/oracle_candidate/`.
- Contract datasets now exist under `results/cage_clp1/datasets/` and `results/cage_clp1/datasets_candidate/`.
- Contract model sanity-check outputs now exist under `results/cage_clp1/models/` and `results/cage_clp1/models_candidate/`.
- Graph-induced hard-goal policy-alignment datasets now exist under `results/cage_clp1/policy_alignment/` and `results/cage_clp1/policy_alignment_candidate/`.

Remaining blocker: q_train matched target selection is not yet wired into branchable probes, so q_train matched closed-loop controls remain unavailable in CLP1.
