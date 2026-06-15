# Phase 4O Regret-Guard Selector Summary

Phase 4O turns the Phase 4N Scene H10 manual choice into a reusable offline
supervised selector for planner-relevant repair loss weighting. It does not
train a new policy, run an environment rollout, or create graph edges. It only
selects among already-run Phase 4M candidates.

## Guard

The selector first applies a strict guard. A planner-relevant candidate is a
strict pass only if all conditions pass:

- final validation MSE ratio <= `1.01`
- direct repair-edge MSE ratio <= `1.00`
- planner-used repair-edge MSE ratio <= `0.99`
- direct repair policy-support score ratio >= `1.00`

If no strict candidate passes, the selector can use a relaxed improvement
fallback: final validation MSE must still be within the same regret budget, and
direct repair-edge MSE, planner-used repair-edge MSE, and policy-support score
must all be no worse than the baseline. If neither strict nor relaxed guards
pass, it falls back to the same augmented-graph `support_bottleneck` baseline.

## Current Selections

| dataset/run | selected method | final val MSE ratio | direct repair MSE ratio | planner-used repair MSE ratio | policy support ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| AntMaze H10 B120 | `planner_relevant_repair_s04` | 0.995124 | 0.995081 | 0.986447 | 1.003763 |
| Scene H10 B192 | `planner_relevant_repair_s02` | 1.005790 | 0.988326 | 0.960478 | 1.002015 |
| Scene H25 B192 | `planner_relevant_repair_s04` | 0.992440 | 0.983579 | 0.991416 | 1.004670 |
| Scene H5 B192 | `planner_relevant_repair_s04` | 0.979038 | 0.969635 | 0.980885 | 1.005549 |

Across these selected candidates, mean final validation MSE ratio is `0.993098`,
mean direct repair-edge MSE ratio is `0.984155`, and mean planner-used
repair-edge MSE ratio is `0.979806`.

## Analysis

The selector preserves the useful Phase 4M/4N pattern while making the default
choice less brittle. Scene H10 no longer depends on the aggressive `s04`
setting: the guard selects `s02`, which keeps planner-used repair-edge MSE
improvement while reducing overall validation-MSE regret. AntMaze H10 and Scene
H5 keep `s04` because it improves all strict guarded supervised proxy metrics.
Scene H25 selects `s04` under the relaxed improvement fallback: planner-used
repair-edge MSE improves only mildly (`0.991x`, just above the strict `0.990x`
cutoff), but final validation MSE, direct repair-edge MSE, and policy support
all improve.

This is a model-selection guard for reset-free supervised evidence. It should be
used before promoting planner-relevant loss weighting into new repaired-graph
runs, but it is not a policy-execution claim.

Artifacts:

- `scripts/run_phase4o_regret_guard_selector.py`
- `phase3e/phase4o_regret_guard.py`
- `results/phase4o/regret_guard/`
