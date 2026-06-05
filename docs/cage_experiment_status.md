# CAGE Focused Experiment Status

## Repository State

- Branch at implementation time: `codex/cage-mvp`.
- The working tree already contained unrelated experiment/status changes before this milestone. They were left in place.

## CAGE Implementation Detected Or Missing

- Detected: `external_src/GAS/cage/config.py`
- Detected: `external_src/GAS/cage/state_machine.py`
- Detected: `external_src/GAS/cage/monitor.py`
- Detected: `external_src/GAS/cage/subgoal_selector.py`
- Detected: `external_src/GAS/cage/recovery.py`
- Detected: `external_src/GAS/cage/tracing.py`
- Detected: `external_src/GAS/evaluate_gas.py` CAGE integration.
- Detected: `scripts/summarize_cage_traces.py`
- Detected: `docs/cage_mvp_plan.md`
- Detected: `docs/cage_mvp_usage.md`

## Experiment Scripts Added

- `scripts/build_cage_eval_command.py`
- `scripts/cage_experiment_manifest.py`
- `scripts/run_cage_manifest.py`
- `scripts/aggregate_cage_experiments.py`
- `scripts/plot_cage_diagnostics.py`

## Commands Run

```bash
python -m py_compile scripts/cage_experiment_manifest.py
python -m py_compile scripts/run_cage_manifest.py
python -m py_compile scripts/aggregate_cage_experiments.py
python -m py_compile scripts/plot_cage_diagnostics.py
python -m py_compile scripts/build_cage_eval_command.py
cd external_src/GAS && python -m py_compile evaluate_gas.py
cd external_src/GAS && python -m py_compile cage/*.py
pytest tests/test_cage_experiment_manifest.py tests/test_cage_aggregation.py -q
pytest tests/test_cage_state_machine.py tests/test_cage_subgoal_selector.py tests/test_cage_experiment_manifest.py tests/test_cage_aggregation.py -q
```

Dry-run manifest:

```bash
python scripts/cage_experiment_manifest.py \
  --checkpoint_root /tmp/nonexistent_checkpoints \
  --output_root /tmp/cage_exp_dryrun \
  --envs antmaze-giant-navigate-v0 \
  --seeds 0 \
  --variants gas cage_fixed_commit cage_drift_only cage_recovery_only cage_full \
  --episodes_per_goal 1 \
  --goals_per_env 1 \
  --manifest_path /tmp/cage_exp_dryrun/manifests/smoke_manifest.jsonl

python scripts/run_cage_manifest.py \
  --manifest_path /tmp/cage_exp_dryrun/manifests/smoke_manifest.jsonl \
  --max_jobs 5 \
  --dry_run
```

Additional checks:

```bash
python scripts/cage_experiment_manifest.py \
  --checkpoint_root /tmp/nonexistent_checkpoints \
  --output_root /tmp/cage_exp_unsupported \
  --envs antmaze-giant-navigate-v0 \
  --seeds 0 \
  --variants cage_risk_path \
  --episodes_per_goal 1 \
  --goals_per_env 1 \
  --manifest_path /tmp/cage_exp_unsupported/manifests/unsupported_manifest.jsonl

python scripts/aggregate_cage_experiments.py \
  --input_root /tmp/cage_exp_dryrun \
  --manifest_path /tmp/cage_exp_dryrun/manifests/smoke_manifest.jsonl \
  --out_csv /tmp/cage_exp_dryrun/tables/smoke_summary.csv \
  --out_md /tmp/cage_exp_dryrun/tables/smoke_summary.md \
  --out_json /tmp/cage_exp_dryrun/tables/smoke_summary.json

python scripts/plot_cage_diagnostics.py \
  --summary_json /tmp/cage_plot_smoke/tables/focused_summary.json \
  --output_dir /tmp/cage_plot_smoke/plots
```

## Validation Results

- All requested `py_compile` checks passed.
- `pytest tests/test_cage_experiment_manifest.py tests/test_cage_aggregation.py -q`: `4 passed`.
- Expanded CAGE test set: `13 passed`.
- Dry-run manifest generated 5 initialized jobs.
- Dry-run runner printed 5 commands and created no evaluation output.
- Unsupported `cage_risk_path` manifest row was marked `unsupported_variant` with a clear reason.
- Aggregation on a trace-free dry-run manifest produced grouped and paired tables without crashing.
- Plot smoke produced all six requested diagnostic PNGs from synthetic summary data.

## Smoke Results If Any

No real MuJoCo/OGBench smoke run was executed. Local antmaze checkpoints exist for seeds 42-46, but the requested tiny smoke condition was seed 0. The current Python environment also lacks `gymnasium` and `d4rl`, so real evaluation cannot run here without changing the environment.

## Known Blockers

- `cage_reachability` is marked unsupported because the current evaluator has no learned reachability model loader.
- `cage_risk_path` is marked unsupported because the current evaluator has no risk-aware path executor.
- Real evaluation requires valid checkpoint paths for matching keygraph and policy files.

## Next Recommended Run

Generate a focused dry-run manifest first:

```bash
python scripts/cage_experiment_manifest.py \
  --checkpoint_root CHECKPOINT_ROOT \
  --output_root OUTPUT_ROOT \
  --envs antmaze-giant-navigate-v0 antmaze-giant-stitch-v0 \
  --seeds 0 1 2 3 \
  --variants gas cage_fixed_commit cage_drift_only cage_recovery_only cage_full \
  --episodes_per_goal 50 \
  --goals_per_env 5 \
  --eval_horizon default \
  --manifest_path OUTPUT_ROOT/manifests/focused_manifest.jsonl
```
