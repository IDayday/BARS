# CAGE-Repair-0 Report

## Scope

Repository root: `/mnt/project/BARS`

GAS evaluator: `external_src/GAS/evaluate_gas.py`

CAGE package: `external_src/GAS/cage/`

Experiment scripts: `scripts/`

Environment: local `gcrlo` conda environment via `/root/miniconda3/envs/gcrlo/bin/python`

No TDR training, keygraph construction, low-level policy training, reachability training, or risk-aware path search was run.

## Existing Pilot-0 Baseline

Pilot-0 used the same checkpoint root:
`/mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138`

Existing Pilot-0 outcome:

| env | seed | gas | cage_fixed_commit | cage_full | diagnosis |
| --- | --- | ---: | ---: | ---: | --- |
| antmaze-giant-navigate-v0 | 42 | 0.60 | NA | 0.36 | full regressed |
| antmaze-giant-stitch-v0 | 42 | 0.84 | NA | 0.76 | full mildly regressed |
| humanoidmaze-large-navigate-v0 | 44 | 0.20 | 0.32 | 0.04 | full severely regressed |

Pilot-0 trace diagnosis: the main failure was drift and replan churn, not missing graph paths. Humanoid CAGE full repeatedly detected drift, failed recovery, and requested thousands of global replans.

Postmortem artifacts:

- `results/cage_pilot0/postmortem/churn_analysis.json`
- `results/cage_pilot0/postmortem/churn_analysis.md`
- `docs/cage_pilot0_postmortem.md`

## Repair Changes

Added trace-only parity mode:

- `--cage_trace_only`
- Variant: `cage_trace_only`
- CAGE is constructed and traces are emitted, but the selected subgoal is the original GAS planner target.
- CAGE recovery, adaptive horizon, final-goal override, and CAGE-triggered replanning are disabled in this mode.

Added safe guardrail mode:

- Variant: `cage_safe_full`
- Uses `--use_cage` plus explicit churn guard flags.
- Leaves existing `cage_full` unchanged.
- Adds cooldown, per-episode replan budget, 100-step replan rate guard, consecutive replan burst guard, recovery lockout, and optional fallback-to-GAS target selection.

New diagnostics:

- `scripts/analyze_cage_churn.py`
- replan churn episode rate
- zero segment reach episode rate
- failed recovery churn episode rate
- unstable execution episode rate
- max consecutive replan burst
- guard trigger and fallback counters

## Validation

Commands run:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python -m py_compile external_src/GAS/evaluate_gas.py external_src/GAS/O_utils/evaluation.py external_src/GAS/cage/*.py scripts/analyze_cage_churn.py scripts/cage_experiment_manifest.py scripts/run_cage_manifest.py scripts/aggregate_cage_experiments.py scripts/plot_cage_diagnostics.py scripts/build_cage_eval_command.py
```

Result: return code 0.

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python -m pytest tests/test_cage_experiment_manifest.py tests/test_cage_aggregation.py tests/test_cage_state_machine.py tests/test_cage_subgoal_selector.py tests/test_cage_churn_guard.py tests/test_cage_trace_only.py -q
```

Result: return code 0, `21 passed in 0.28s`.

## Repair Smoke

Smoke scope:

- env: `antmaze-giant-navigate-v0`
- seed: 42
- variants: `gas`, `cage_trace_only`, `cage_fixed_commit`, `cage_full`, `cage_safe_full`
- budget: `episodes_per_goal=1`, `goals_per_env=1`

Artifacts:

- manifest: `results/cage_repair0/smoke_antmaze_nav/manifests/smoke_manifest.jsonl`
- status: `results/cage_repair0/smoke_antmaze_nav/status/smoke_status.jsonl`
- summary: `results/cage_repair0/smoke_antmaze_nav/tables/smoke_summary.md`
- churn: `results/cage_repair0/smoke_antmaze_nav/postmortem/churn_analysis.md`

All 5 smoke jobs returned code 0.

| variant | success | global replans | replan rate per 100 steps | recovery lockout suppressions | segment reach |
| --- | ---: | ---: | ---: | ---: | ---: |
| gas | 0.00 | NA | NA | NA | NA |
| cage_trace_only | 0.00 | 0 | 0.00 | 0 | 0.0780 |
| cage_fixed_commit | 0.00 | 0 | 0.00 | 0 | 0.0000 |
| cage_full | 0.00 | 1234 | 123.40 | 0 | 0.0000 |
| cage_safe_full | 0.00 | 5 | 0.50 | 10 | 0.0109 |

Smoke conclusion: `cage_trace_only` matched GAS success, and `cage_safe_full` prevented the full-mode replan storm in this smoke.

## Repair Minipilot

Minipilot scope:

| env | seed | variants | episodes_per_goal | goals_per_env |
| --- | ---: | --- | ---: | ---: |
| antmaze-giant-navigate-v0 | 42 | gas, trace_only, fixed_commit, full, safe_full | 5 | 5 |
| antmaze-giant-stitch-v0 | 42 | gas, trace_only, fixed_commit, full, safe_full | 5 | 5 |
| humanoidmaze-large-navigate-v0 | 44 | gas, trace_only, fixed_commit, full, safe_full | 5 | 5 |

All 15 minipilot jobs returned code 0.

Artifacts:

- `results/cage_repair0/minipilot_antmaze_nav/tables/minipilot_summary.md`
- `results/cage_repair0/minipilot_antmaze_stitch/tables/minipilot_summary.md`
- `results/cage_repair0/minipilot_humanoid_large_nav/tables/minipilot_summary.md`
- `results/cage_repair0/postmortem/churn_analysis.json`
- `results/cage_repair0/postmortem/churn_analysis.md`

## Main Results

| env | variant | success | target switches | stalls | drift | recovery attempts | recovery success | global replans | replans / 100 steps | max burst | segment reach | mean progress | guard triggers | fallback steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| antmaze-nav | gas | 0.64 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| antmaze-nav | trace_only | 0.64 | 113.16 | 0.00 | 0.00 | 0.00 | NA | 0.00 | 0.00 | 0.00 | 0.0531 | 0.2221 | 0.00 | 0.00 |
| antmaze-nav | fixed_commit | 0.00 | 68.24 | 37.80 | 0.00 | 0.00 | NA | 0.00 | 0.00 | 0.00 | 0.0000 | 0.0766 | 0.00 | 0.00 |
| antmaze-nav | full | 0.56 | 65.08 | 8.92 | 0.00 | 2.00 | 0.02 | 292.08 | 30.53 | 0.00 | 0.0104 | 0.2198 | 0.00 | 0.00 |
| antmaze-nav | safe_full | 0.72 | 80.12 | 10.80 | 0.00 | 2.00 | 0.02 | 5.56 | 0.62 | 0.92 | 0.0131 | 0.2418 | 0.00 | 0.00 |
| antmaze-stitch | gas | 0.80 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| antmaze-stitch | trace_only | 0.84 | 135.88 | 0.00 | 0.00 | 0.00 | NA | 0.00 | 0.00 | 0.00 | 0.0444 | 0.3019 | 0.00 | 0.00 |
| antmaze-stitch | fixed_commit | 0.12 | 63.12 | 30.88 | 0.00 | 0.00 | NA | 0.00 | 0.00 | 0.00 | 0.0018 | 0.1327 | 0.00 | 0.00 |
| antmaze-stitch | full | 0.68 | 58.24 | 7.32 | 0.00 | 1.96 | 0.02 | 336.88 | 34.99 | 0.00 | 0.0054 | 0.2456 | 0.00 | 0.00 |
| antmaze-stitch | safe_full | 0.76 | 72.12 | 8.64 | 0.00 | 1.84 | 0.04 | 5.68 | 0.61 | 0.76 | 0.0080 | 0.2700 | 0.00 | 0.00 |
| humanoid-large-nav | gas | 0.28 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| humanoid-large-nav | trace_only | 0.28 | 112.96 | 0.00 | 0.00 | 0.00 | NA | 0.00 | 0.00 | 0.00 | 0.0000 | 0.1451 | 0.00 | 0.00 |
| humanoid-large-nav | fixed_commit | 0.32 | 77.36 | 48.60 | 0.00 | 0.00 | NA | 0.00 | 0.00 | 0.00 | 0.0000 | 0.1105 | 0.00 | 0.00 |
| humanoid-large-nav | full | 0.00 | 6.88 | 5.72 | 15.76 | 2.00 | 0.00 | 3865.68 | 193.28 | 0.00 | 0.0000 | 0.0044 | 0.00 | 0.00 |
| humanoid-large-nav | safe_full | 0.12 | 108.00 | 3.64 | 21.36 | 0.88 | 0.00 | 49.24 | 2.55 | 1.00 | 0.0000 | 0.1224 | 0.92 | 45.08 |

## Paired Delta vs GAS

| env | variant | delta success | delta normalized score |
| --- | --- | ---: | ---: |
| antmaze-nav | trace_only | 0.00 | 0.00 |
| antmaze-nav | fixed_commit | -0.64 | -64.00 |
| antmaze-nav | full | -0.08 | -8.00 |
| antmaze-nav | safe_full | +0.08 | +8.00 |
| antmaze-stitch | trace_only | +0.04 | +4.00 |
| antmaze-stitch | fixed_commit | -0.68 | -68.00 |
| antmaze-stitch | full | -0.12 | -12.00 |
| antmaze-stitch | safe_full | -0.04 | -4.00 |
| humanoid-large-nav | trace_only | 0.00 | 0.00 |
| humanoid-large-nav | fixed_commit | +0.04 | +4.00 |
| humanoid-large-nav | full | -0.28 | -28.00 |
| humanoid-large-nav | safe_full | -0.16 | -16.00 |

## Interpretation

Trace-only parity:

- `antmaze-giant-navigate-v0`: trace-only exactly matched GAS success, 0.64 vs 0.64.
- `humanoidmaze-large-navigate-v0`: trace-only exactly matched GAS success, 0.28 vs 0.28.
- `antmaze-giant-stitch-v0`: trace-only was 0.84 vs GAS 0.80. This is a small positive difference under a small budget and should be treated as rollout variance, not an algorithmic improvement.

Component diagnosis:

- The only positive component on humanoid remains commitment-only: 0.32 vs GAS 0.28.
- Full CAGE remains harmful: humanoid success is 0.00 with 3865.68 CAGE global replans per episode on average.
- Recovery is not helping in the current implementation. Full CAGE has low or zero recovery success and high failed-recovery churn. Safe full reduces repeated recovery pressure but does not make recovery successful.
- Drift detection is too destabilizing for humanoid-like tasks when it is allowed to trigger immediate global replanning.
- Adaptive/final/full control improves some progress metrics in AntMaze but is not robust enough to expand benchmark claims.

Churn guard result:

- `cage_safe_full` prevents replan storms.
- AntMaze nav: full 292.08 replans to safe_full 5.56.
- AntMaze stitch: full 336.88 replans to safe_full 5.68.
- Humanoid: full 3865.68 replans to safe_full 49.24.
- Humanoid replan rate dropped from 193.28 to 2.55 per 100 steps.
- Humanoid still has zero segment reach and zero recovery success, so the guard fixes churn but not the underlying control-interface failure.

Decision:

- Do not expand to the full 8-env focused benchmark yet.
- Replace `cage_full` with `cage_safe_full` only for guarded diagnostic runs, not as a claimed final algorithm.
- Treat `cage_fixed_commit` as the main positive signal.
- Design CAGE-v0.2 around commitment-first execution and either disable recovery by default on humanoid-like tasks or require a stronger preregistered recovery criterion before enabling it.

## Known Blockers

- Remote `training-rl-zt3` was not used because remote OGBench/dependency readiness is still blocked and remote hostname resolution failed in the current session.
- GAS baseline has no CAGE trace, so paired trace deltas against GAS are unavailable for trace-only diagnostic fields.
- This minipilot has one artifact seed per env and a small episode budget. It is diagnostic only, not a benchmark-wide performance claim.
- `cage_reachability` and `cage_risk_path` remain unsupported as intended.

## Next Command

Recommended next local diagnostic run, not full benchmark expansion:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH WANDB_MODE=disabled /root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py --manifest_path results/cage_repair0/minipilot_humanoid_large_nav/manifests/minipilot_manifest.jsonl --max_jobs 5 --dry_run
```

If a new preregistered CAGE-v0.2 commitment-first variant is added later, rerun the same three-env repair minipilot before any 8-env expansion.
