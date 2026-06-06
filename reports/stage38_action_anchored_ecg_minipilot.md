# Stage38 Action-Anchored ECG Minipilot

- branch: `codex/cage-mvp`
- commit at run start: `d8663ff`
- envs/seeds: `antmaze-giant-navigate-v0:42`, `antmaze-giant-stitch-v0:42`
- budget: `episodes_per_goal=5`, `goals_per_env=5`
- status: all 12 jobs returned code 0.

## Main Table

| env | gas | trace_only | safe_full | ecg_trace_only | ecg_planner | ecg_adapter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| antmaze-giant-navigate-v0 | 0.60 | 0.64 | 0.68 | 0.60 | 0.64 | 0.00 |
| antmaze-giant-stitch-v0 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.00 |

## ECG Runtime Diagnostics

`cage_ecg_planner` and `cage_ecg_planner_trace_only` did not produce a measurable online planner intervention in this run. Episode traces show `ecg_fallback_count=1.0`, and aggregate tables have no non-null `ecg_plan_length`/`ecg_contract_lcb`, meaning the runtime graph query failed to find a usable ECG path and fell back to the original GAS target.

Therefore the planner safety result is degenerate: it is safe because it is effectively GAS fallback, not because ECG planning improved execution.

`cage_ecg_adapter` replaced the low-level action while the target selection fell back to GAS, and success fell to 0.00 in both AntMaze envs. This is a clear adapter safety failure despite the offline BC validation loss being below the mean-action baseline.

## Gate Notes

- Trace-only parity: PASS on minipilot. `cage_ecg_planner_trace_only` matches GAS on both envs.
- Planner safety: PASS but degenerate. ECG planner does not materially underperform GAS, but fallback rate is 1.0.
- Adapter safety: FAIL. Adapter success is 0.00 on both envs.
- Online improvement: FAIL. No evidence of ECG online gain.

## Paths

- manifest: `results/cage_ecg/action_anchored_eval/minipilot_antmaze/manifest.jsonl`
- status: `results/cage_ecg/action_anchored_eval/minipilot_antmaze/status/status.jsonl`
- summary: `results/cage_ecg/action_anchored_eval/minipilot_antmaze/tables/minipilot_summary.md`
