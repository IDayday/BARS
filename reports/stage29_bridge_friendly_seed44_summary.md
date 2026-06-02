# Stage29 Seed44 Follow-up: Execution-Risk Diagnostics and Bridge-Friendly Risk

Date: 2026-06-02 UTC

## Gate Context

- Stage29-A offline SCG evidence: PASS.
- Stage29-B execution evidence: PASS.
- This report is a follow-up 20ep-focused diagnostic, not a 50ep promotion.
- Planner evidence uses `fallback_mode=none`.
- Boundary and reachability remain unvalidated in online reports because `boundary_loaded=0` and `reachability_loaded=0`.

## Result Files

- Soft execution-risk offline smoke:
  - `runs_stage29f_soft_exec_offline_smoke/antmaze-giant-stitch-v0/full_bars/seed44_default/stage29_support_calibrated_audit.csv`
  - `runs_stage29f_soft_exec_offline_smoke/antmaze-giant-stitch-v0/full_bars/seed44_soft_w04_dt015/stage29_support_calibrated_audit.csv`
- Soft execution-risk online partial:
  - `runs_stage29f_soft_exec_online_seed44/antmaze-medium-stitch-v0/full_bars/seed44/stage29_online_eval_gate.csv`
- Bridge-friendly offline smoke:
  - `runs_stage29g_bridge_friendly_offline_smoke/antmaze-medium-stitch-v0/full_bars/seed44_default/stage29_support_calibrated_audit.csv`
  - `runs_stage29g_bridge_friendly_offline_smoke/antmaze-medium-stitch-v0/full_bars/seed44_bridge_r010_min005/stage29_support_calibrated_audit.csv`
- Bridge-friendly online focused gate:
  - `runs_stage29g_bridge_friendly_online_seed44/antmaze-medium-stitch-v0/full_bars/seed44/stage29_online_eval_gate.csv`
  - `runs_stage29g_bridge_friendly_online_seed44/_analysis/stage29_online_eval_report.md`
  - `runs_stage29g_bridge_friendly_online_seed44/_analysis/stage29_online_eval_summary.csv`
  - `runs_stage29g_bridge_friendly_online_seed44/_analysis/stage29_online_eval_promotion_gate.csv`

## Findings

Soft execution-risk was rejected. Offline smoke improved some path support statistics, but online medium-stitch seed44 regressed:

| planner | original success | soft partial success | original no_path | soft partial no_path |
| --- | ---: | ---: | ---: | ---: |
| STAGE29_LEXICOGRAPHIC | 0.55 | 0.278 | 0.15 | 0.167 |
| SUPPORT_BUDGET_K1 | 0.55 | 0.353 | 0.35 | 0.529 |

Bridge-friendly risk was retained as the current candidate for the next 20ep matrix. It lowers supported/cross bridge risk using:

- `stage29_support.cross_edge_base_risk=0.1`
- `stage29_support.supported_cross_min_risk=0.05`

Offline medium-stitch seed44 smoke on the shared 66-pair prefix showed:

| planner | variant | found | no_path | unsupported | support_risk | mean support |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SUPPORT_BUDGET_K1 | default | 1.000 | 0.000 | 0.000 | 0.735 | 0.599 |
| SUPPORT_BUDGET_K1 | bridge-friendly | 1.000 | 0.000 | 0.000 | 0.199 | 0.608 |

Online medium-stitch seed44 20ep focused gate:

| planner | success | no_path | path_cross | unsupported | executed support score |
| --- | ---: | ---: | ---: | ---: | ---: |
| BARS_BASE | 0.10 | 0.80 | 1.000 | 2.15 | n/a |
| STAGE29_LEXICOGRAPHIC | 0.40 | 0.20 | 0.493 | 0.00 | 0.313 |
| SUPPORT_BUDGET_K1 | 0.60 | 0.00 | 0.654 | 0.00 | 0.459 |

## Decision

- Reject soft execution-risk and temporal densification for promotion.
- Do not promote STAGE29_LEXICOGRAPHIC from this bridge-friendly run; it regressed versus the original seed44 LEX result.
- Keep `SUPPORT_BUDGET_K1` with bridge-friendly risk as the only candidate for the next controlled 20ep gate.
- Do not launch 50ep yet. The automatic promotion gate is correctly `BLOCKED_BY_20EP_GATE` because only one environment was compared.

## Next Experiment

Run 4 envs x seeds 44-46 20ep only for:

- `BARS_BASE`
- original Stage29 `SUPPORT_BUDGET_K1`
- bridge-friendly `SUPPORT_BUDGET_K1`

Promotion criteria remain unchanged: no stitch regression worse than -1pp, navigate not below baseline -1pp, no no-path increase, and false shortcut usage substantially below BARS_BASE.
