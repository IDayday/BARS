# Stage17 Online Quick Decision

## Scope

- Sweep: `configs/sweeps/d4rl_stage16_online_quick_loaded.json`
- Log root: `runs_stage17_online_quick_loaded`
- Date: `2026-05-13`
- Runs: `2 env x 3 seeds x 3 variants = 18`, all completed with `rc=0`
- Protocol: loaded `Stage16` warmstart, `30` online eval episodes per run

## Core Results

Overall online success by variant:

- `shortest`: `0.0667`
- `reachability`: `0.1333`
- `full_bars`: `0.1500`

Decision deltas:

- `reachability - shortest = +0.0667` (`+6.67pp`)
- `full_bars - reachability = +0.0167` (`+1.67pp`)

Per-env success means:

- `antmaze-medium-play-v2`
  - `shortest`: `0.0111`
  - `reachability`: `0.1333`
  - `full_bars`: `0.0778`
- `antmaze-medium-diverse-v2`
  - `shortest`: `0.1222`
  - `reachability`: `0.1333`
  - `full_bars`: `0.2222`

## Rule Mapping

- `reachability success > shortest success`: true
- Improvement is `>= 5pp`: true (`+6.67pp`)
- `full_bars success >= reachability success`: true overall (`0.1500 >= 0.1333`)

Interpretation:

- Bottom-policy reachability transfers to real online D4RL success.
- `support_modes` boundary is good enough to stay on the mainline.
- `full_bars` is not uniformly better on every env: it wins clearly on `medium-diverse`, but trails `reachability` on `medium-play`.

## Decision

`GO_LARGE_ONLINE`

Reason:

- The medium quick online gate was passed.
- The key gate was the online transfer of `reachability`; that signal is now positive and above the `5pp` threshold.
- `full_bars` is the best overall variant in this sweep, so boundary support should remain in the candidate mainline rather than being dropped.

## Operational Caveat

- `full_bars` had noticeably longer tail latency than `shortest` and `reachability`.
- One run, `antmaze-medium-diverse-v2 full_bars seed1`, finished successfully but took much longer to close out its final episodes.
- Scheduler logs also suggest `max-jobs-per-gpu=1` was not enforced perfectly during refill, so large-online execution should be monitored carefully for runtime fairness and queue behavior.

## Recommended Next Step

- Proceed to larger online evaluation with `shortest`, `reachability`, and `full_bars`.
- Keep the same loaded `Stage16` warmstart protocol.
- Watch runtime and scheduling behavior for `full_bars`, but do not block the next online expansion on that alone.
