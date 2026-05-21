# Stage24 Results Package

This package contains the compact Stage24 result artifacts for the BARS/GAS experiments.
It excludes large checkpoints, OGBench dataset files, and full debug JSONL traces.

## Final Gates

- reachability_confirm: HOLD_REACHABILITY_WEAK_EFFECT
- local_drift_repair: HOLD_LOCAL_DRIFT_REPAIR
- oracle_headroom: NO_ORACLE_UPPER_BOUND
- p_bridge: SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM
- boundary: HOLD_BOUNDARY_DIAGNOSTIC_ONLY
- integrated: SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE
- d4rl_protocol: HOLD_D4RL_PROTOCOL_REPAIR

## Main Evidence

- Reachability confirmation completed 6 env-seed cells with 100 episodes each. Primary variant: `gas_reachability_budget_calibrated`; mean delta over GAS shortest: +0.17pp; paired wins/losses/ties: 2/4/0.
- Local drift repair variants were evaluated with `fallback_mode=none`. F4 drift labels were reduced, but best repair did not preserve overall success versus baseline, so the repair claim stays on hold.
- Oracle scan on `scene-play-v0 seed0` found no path-level oracle upper bound: oracle shorter-path rate 0.014, oracle mean cost reduction 0.103, set_state_rate 0.0.

## Layout

- `reports/`: final CSV/JSON/Markdown reports and oracle scan summaries.
- `raw_eval/`: all Stage24 `eval.csv` files, preserving run-root layout.
- `configs/`: Stage24 configs.
- `scripts/`: Stage24 run/analyze scripts, including mirror/prefetch helpers.
- `logs/`: lightweight job tables, failed-job files, and command logs.
- `code/`: relevant code snapshots and `stage24_code_diff.patch`.
- `commands/stage24_commands.sh`: replay commands used for the final Stage24 runs.
- `MANIFEST.txt` and `CHECKSUMS.sha256`: file inventory and checksums.
