# Stage24 Results/Config Package

Generated: 2026-05-19 CST

This package contains the Stage24 configs, scripts, reports, logs, code snapshots, and completed oracle-scan artifacts.

## Outcome

- `scene-play-v0/seed0` oracle-headroom scan completed and produced `NO_ORACLE_UPPER_BOUND`.
- Medium reachability confirmation and local-drift repair evaluations were attempted, but blocked by missing protocol-aligned GAS artifacts for `antmaze-medium-navigate-v0` and `antmaze-medium-stitch-v0`.
- p_bridge and integrated BARS-v3 were correctly skipped because Stage24 requires `PASS_ORACLE_HEADROOM` before either can run.

## Key Files

- `reports/stage24_execution_status.md`: execution summary, blockers, and final gates.
- `reports/stage24_decisions.md`: gate table, compact evidence, and next commands.
- `reports/stage24_gate_status.json`: machine-readable final gates.
- `reports/stage24_oracle_headroom.csv`: oracle-headroom scan result.
- `logs/reachability_confirm/failed_jobs.csv`: medium reachability blocked jobs.
- `logs/local_drift/failed_jobs.csv`: local-drift repair blocked jobs.
- `logs/medium_prepare_probe/`: GAS artifact probe logs for medium envs.
- `artifacts/stage24/scene-play-v0/seed0/`: completed oracle scan graph and edge-execution outputs.
- `code/stage24_code_diff.patch`: diff for Stage24 code/config/report additions and touched helper files.

## Final Gates

See `reports/stage24_gate_status.json`.

