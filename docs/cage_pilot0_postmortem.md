# CAGE Pilot-0 Postmortem

## Inputs

Analysis command:

```bash
python scripts/analyze_cage_churn.py \
  --input_roots \
    results/cage_pilot0/minipilot_local_antmaze_nav \
    results/cage_pilot0/minipilot_local_antmaze_stitch \
    results/cage_pilot0/minipilot_local_humanoid_large_nav \
  --out_json results/cage_pilot0/postmortem/churn_analysis.json \
  --out_md results/cage_pilot0/postmortem/churn_analysis.md
```

Outputs:

- `results/cage_pilot0/postmortem/churn_analysis.json`
- `results/cage_pilot0/postmortem/churn_analysis.md`

The analyzer deduplicates manifest rows by `job_id`, because the navigate Pilot-0 root contains both the original manifest and a remaining-jobs manifest.

## Summary Table

| env | variant | success | replans | drift | recovery success | segment reach | unstable episode rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze navigate | gas | 0.60 | 0.00 | NA | NA | NA | 0.00 |
| antmaze navigate | cage_fixed_commit | 0.00 | 0.00 | 0.00 | NA | 0.0000 | 1.00 |
| antmaze navigate | cage_recovery_only | 0.00 | 567.64 | 0.12 | 0.04 | 0.0014 | 1.00 |
| antmaze navigate | cage_full | 0.36 | 481.20 | 0.00 | 0.02 | 0.0110 | 0.88 |
| antmaze stitch | gas | 0.84 | 0.00 | NA | NA | NA | 0.00 |
| antmaze stitch | cage_fixed_commit | 0.12 | 0.00 | 0.00 | NA | 0.0018 | 0.92 |
| antmaze stitch | cage_recovery_only | 0.12 | 493.44 | 0.00 | 0.00 | 0.0009 | 1.00 |
| antmaze stitch | cage_full | 0.76 | 249.92 | 0.00 | 0.02 | 0.0060 | 0.92 |
| humanoid large navigate | gas | 0.20 | 0.00 | NA | NA | NA | 0.00 |
| humanoid large navigate | cage_fixed_commit | 0.32 | 0.00 | 0.00 | NA | 0.0000 | 1.00 |
| humanoid large navigate | cage_drift_only | 0.04 | 3738.48 | 15.68 | NA | 0.0000 | 1.00 |
| humanoid large navigate | cage_recovery_only | 0.00 | 3802.00 | 16.00 | 0.00 | 0.0000 | 1.00 |
| humanoid large navigate | cage_full | 0.04 | 3815.52 | 16.24 | 0.00 | 0.0000 | 1.00 |

## Diagnosis

1. Which component appears beneficial?

`cage_fixed_commit` is the only positive component in Pilot-0, and only on `humanoidmaze-large-navigate-v0`: success improves from GAS 0.20 to 0.32. This supports the commitment/hysteresis hypothesis, but it is a single artifact seed and not a benchmark claim.

2. Which component appears harmful?

The harmful component is the combination of drift-triggered replanning and recovery. In humanoid, `cage_drift_only`, `cage_recovery_only`, and `cage_full` all produce thousands of global replan requests and success collapses to 0.04 or 0.00.

3. Is recovery helping or creating churn?

Recovery is creating churn in the current implementation. Recovery success is 0.02 on AntMaze full, 0.02 on Stitch full, and 0.00 on Humanoid full. Recovery-only variants request hundreds to thousands of replans and do not improve success.

4. Is drift detection too sensitive?

For humanoid, yes. Drift counts are about 16 per episode for drift/recovery/full variants, and those variants request about 3.7k-3.8k global replans. This is not a no-path failure; it is closed-loop drift/replan churn.

5. Is adaptive horizon selecting unreachable targets?

The trace evidence is consistent with unreachable or poorly executable targets, especially where segment target reach is near zero. However, fixed commitment also has zero segment reach in humanoid while improving success, so segment reach alone is too strict as a success proxy. The safer conclusion is that adaptive horizon plus drift/recovery is not robust enough yet.

6. Does CAGE full improve trace metrics anywhere?

CAGE full reduces stall compared with fixed/recovery ablations:

- AntMaze navigate stall drops from 37.80 under fixed commitment to 8.68 under full.
- AntMaze stitch stall drops from 30.88 to 6.00.
- Humanoid stall drops from 48.60 under fixed commitment to 6.04 under full.

This stall reduction does not translate into stable execution because recovery success and segment reach remain near zero while replan counts explode.

7. Does CAGE full regress success because of trace-measurable churn?

Yes. The regressions are explained by trace metrics:

- AntMaze navigate: CAGE full success 0.36 vs GAS 0.60, with 481.20 replans and 0.02 recovery success.
- AntMaze stitch: CAGE full success 0.76 vs GAS 0.84, with 249.92 replans and 0.02 recovery success.
- Humanoid: CAGE full success 0.04 vs GAS 0.20, with 3815.52 replans, 16.24 drift count, 0.00 recovery success, and 0.00 segment reach.

8. Should we expand benchmark now?

No. The next step is CAGE-Repair-0, not benchmark expansion. We need trace-only parity and safe guardrails first. Expanding the benchmark with known replan storm behavior would mostly measure a broken control interface.

## Repair Target

CAGE-Repair-0 should verify:

- `cage_trace_only` matches GAS behavior closely enough to rule out instrumentation overhead.
- `cage_safe_full` prevents replan storms without changing default `cage_full`.
- Recovery lockout and fallback-to-GAS reduce repeated failed recovery.
- Humanoid no longer enters zero segment reach, zero recovery success, and thousands of replans.

If `cage_safe_full` still regresses after removing churn, the next algorithmic direction should be commitment-first CAGE-v0.2 with recovery disabled by default for humanoid-like tasks.
