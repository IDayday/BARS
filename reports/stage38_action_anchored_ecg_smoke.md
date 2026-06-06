# Stage38 Action-Anchored ECG Smoke

- branch: `codex/cage-mvp`
- commit at run start: `d8663ff`
- env/seed: `antmaze-giant-navigate-v0:42`
- budget: `episodes_per_goal=1`, `goals_per_env=1`
- status: all 6 jobs returned code 0.

## Results

| variant | success | normalized_score | return | status |
| --- | ---: | ---: | ---: | --- |
| gas | 0.00 | 0.00 | 0.00 | succeeded |
| cage_trace_only | 0.00 | 0.00 | 0.00 | succeeded |
| cage_safe_full | 0.00 | 0.00 | 0.00 | succeeded |
| cage_ecg_planner_trace_only | 1.00 | 100.00 | 1.00 | succeeded |
| cage_ecg_planner | 0.00 | 0.00 | 0.00 | succeeded |
| cage_ecg_adapter | 0.00 | 0.00 | 0.00 | succeeded |

## Interpretation

Smoke confirms the evaluator can load the action-anchored graph, planner score, contract model, and policy adapter without runtime errors. The single episode is too small for performance interpretation. ECG planner/adapter variants were substantially slower than GAS because the runtime MVP queries a 100k-edge graph during evaluation.

The one-episode trace-only discrepancy is treated as inconclusive; the minipilot is the binding parity check.

## Paths

- manifest: `results/cage_ecg/action_anchored_eval/smoke/manifest.jsonl`
- status: `results/cage_ecg/action_anchored_eval/smoke/status/status.jsonl`
- summary: `results/cage_ecg/action_anchored_eval/smoke/tables/smoke_summary.md`
