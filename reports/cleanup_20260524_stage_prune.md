# BARS Stage Cleanup Record

Date: 2026-05-24 Asia/Shanghai

## Baseline-First Rule

Cleanup followed `BARS_AUTONOMOUS_RESEARCH_LOOP_BASELINE_FIRST.md`:

- Baseline and adapter certification evidence is retained.
- Reduced/smoke raw outputs are not treated as scientific evidence.
- Planner evidence must remain tied to report files and gates.

## Retained

- Core code: `bars/`, current `scripts/`, `configs/`, `external_src/`, adapters, README/status files.
- Baseline-first state: `research_state/`, `rounds/`, `reports/round_*`.
- Stage20/Stage21 docs/configs as design lineage for Route-B/GAS/BARS integration.
- Stage22/Stage23 report summaries and CSVs, including finalized summaries, gate status, protocol repair, failure atlas, bridge/oracle, p_bridge, and boundary diagnostics.
- Stage24/Stage25 local report summaries because they record HOLD/diagnostic gates.
- Official and final evidence artifacts under `artifacts/gas`, `artifacts/stage22`, `artifacts/stage23`, `artifacts/stage24`, `artifacts/stage25`.
- Round004/Round005 GAS self-train final checkpoints, eval CSVs, train CSVs, and flags.
- Active Round006 files and outputs. Round006 was running during cleanup and was not modified.

## Removed

- Completed raw run directories for Round003-Round005 and Stage22-Stage25.
- Round-local duplicate raw logs that were already summarized in reports.
- Stage15-Stage19 dedicated scripts, sweep configs, and detailed reports after Round002/Round003 reclassification downgraded those rows to smoke/protocol-debug evidence.
- Stage18/Stage19 protocol-ablation helper scripts and stale README Stage1 sweep references.
- Python `__pycache__` directories.
- Duplicate package tarballs and expanded report package snapshots under `reports/package_*`.
- Package tarballs under `packages/*.tar.gz`.
- Round004/Round005 intermediate checkpoint files `params_100000.pkl` through `params_900000.pkl`; final `params_1000000.pkl` files were retained.
- Incomplete `*.tmp` artifact files under `artifacts/gas`.

## Not Removed

- `runs_round006_gas_dynamic/`.
- `artifacts/gas_selftrain_round006/`.
- Round006 launch/report files.
- Packaged evidence directories under `packages/`, because they are tracked lightweight evidence bundles.

## Post-Cleanup Footprint

Approximate local footprint after cleanup:

- `artifacts/`: 6.7G
- `reports/`: 21M
- `packages/`: 20M
- active `runs_round006_gas_dynamic/`: 7.5M at check time
