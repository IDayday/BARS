# CAGE-MVP Usage

CAGE is disabled by default. Running `evaluate_gas.py` without `--use_cage` follows the existing GAS evaluation path.

## Enable CAGE

```bash
python evaluate_gas.py \
  --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl \
  --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl \
  --use_cage \
  --cage_trace_path results/cage_mvp/antmaze_giant_navigate_seed0.jsonl \
  --cage_debug
```

The evaluator still uses the existing TDR, graph, planner, and low-level policy. CAGE only changes how the active graph path is converted into the subgoal sent to the policy.

## Flags

- `--use_cage`: enable the CAGE execution wrapper. Default: false.
- `--cage_trace_path`: JSONL trace path. Default: `<save_eval_dir>/cage_trace.jsonl`.
- `--cage_min_commit_steps`: minimum subgoal commitment steps. Default: 8.
- `--cage_stall_window`: rolling progress window for stall detection. Default: 8.
- `--cage_progress_eps`: minimum progress over the stall window. Default: 0.01.
- `--cage_drift_threshold`: distance-to-path threshold for path drift. Default: 16.0.
- `--cage_max_subgoal_dist`: maximum selected subgoal distance in TDR space. Default: 24.0.
- `--cage_min_subgoal_dist`: minimum selected subgoal distance unless near final goal. Default: 2.0.
- `--cage_recovery_commit_steps`: local recovery target commitment length. Default: 12.
- `--cage_max_recovery_attempts`: maximum local recovery attempts before global replanning request. Default: 2.
- `--cage_recovery_suffix_weight`: penalty for recovery nodes that move backward on the path. Default: 0.25.
- `--cage_final_phase_dist`: final-goal phase trigger distance. Default: 8.0.
- `--cage_final_min_commit_steps`: final-goal phase commitment length. Default: 12.
- `--cage_debug`: emit step-level JSONL trace records. Default: false.

Humanoid environments internally use a shorter max horizon and slightly longer commitment defaults.

## Trace Summary

```bash
python scripts/summarize_cage_traces.py results/cage_mvp/antmaze_giant_navigate_seed0.jsonl
```

The summary reports success/no-path rates, target switching, stalls, drift, recovery attempts, recovery success rate, final-goal phase success proxy, segment progress, and segment target reach rate.

## Notes

- CAGE-MVP uses Euclidean distance in TDR space as a proxy progress signal.
- The MVP reachability score is `exp(-distance / tau)`. It is a heuristic only.
- `should_replan` triggers at most a force-closest call to the existing GAS shortest-path function; no graph construction or training code is changed.
- JSONL tracing is append-only and does not modify existing `eval.csv`.
