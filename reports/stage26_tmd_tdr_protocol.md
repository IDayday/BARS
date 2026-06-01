# Stage 26 TMD/TDR Protocol

Branch: `stage26-tdr-factor-tmdcost`
Commit: `f864251`
Dataset root: `/mnt/project/offlinerl_datasets/ogbench`
Runs root: `runs_stage26_tmd_tdr`

## Current Scope

This report tracks Stage 26 progress against `tmd_gas_tdr_experiment_plan_and_codex_goal.md`.
The first active matrix is Phase B: seed-matched GAS graph baseline versus GAS graph plus TMD soft-cost blend.

## Reliability Rules

- Every launched run is recorded in `runs_stage26_tmd_tdr/stage26_phase_b_manifest.tsv`.
- Variants are compared only against GAS rows with matching env, env seed, GAS seed, task set, and episode budget when those rows are available.
- Pilot rows guide follow-up selection; confirmed claims require the pre-registered CI and task-wise gates.
- `fallback=none` is preserved by the underlying evaluator.
