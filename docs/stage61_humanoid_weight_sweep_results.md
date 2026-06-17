# Stage61 Humanoid Weight-Sweep Results

Last updated: 2026-06-16

## Offline Boundary

This line remains within offline RL method development.

- Contract datasets are built from fixed local OGBench datasets and frozen
  GAS artifacts.
- Actor-conditioned labels use frozen policies on offline states/actions.
- Graph patching modifies planner-time edge weights only.
- Environment rollouts are used only for reporting.

## What Changed

Updated:

- `scripts/stage58_launch_humanoid_filtered_eval.py`

Fix:

- tolerate concurrent `latest` symlink updates when multiple evaluation
  launchers run in parallel on one GPU.

## Reference Local Baseline

Stage44 local GAS retrain:

| env | overall success |
| --- | ---: |
| `humanoidmaze-large-navigate-v0` | 0.776 |

Matched Stage59 20-episode re-eval on the same artifact:

| variant | risk weight | episodes | success |
| --- | ---: | ---: | ---: |
| original keygraph | n/a | 20 | 0.80 |

## Stage58 Smoke at `risk_weight=0.25`

Run root:

```text
runs_stage58_humanoid_filtered_eval_gpu3/20260616_170605
```

5-episode smoke looked positive:

| variant | success |
| --- | ---: |
| original | 0.80 |
| filtered base | 0.88 |
| filtered actor raw | 0.84 |
| filtered actor conditioned | 0.88 |

This was encouraging but not stable.

## Stage59 20-Episode Check at `risk_weight=0.25`

Run root:

```text
runs_stage59_humanoid_report_eval_gpu3/20260616_171200
```

| variant | success |
| --- | ---: |
| original | 0.80 |
| filtered base | 0.76 |
| filtered actor conditioned | 0.78 |

Conclusion:

- the 5-episode gain at `0.25` was not robust;
- the planner penalty was too strong;
- graph repair remained promising, but penalty calibration mattered.

## Stage60 Weight Sweep

Run roots:

```text
runs_stage60_humanoid_risk_sweep_gpu3/actor_conditioned_w0p10_10ep
runs_stage60_humanoid_risk_sweep_gpu3/actor_conditioned_w0p15_10ep
runs_stage60_humanoid_risk_sweep_gpu3/base_w0p10_10ep
```

10-episode sweep:

| variant | risk weight | success |
| --- | ---: | ---: |
| filtered actor conditioned | 0.10 | 0.94 |
| filtered actor conditioned | 0.15 | 0.82 |
| filtered base | 0.10 | 0.84 |

Interpretation:

- reducing penalty strength helped a lot;
- actor-conditioned scoring benefited more than filtered-base when the weight
  was mild;
- `0.15` was already noticeably worse than `0.10`.

## Stage61 20-Episode Confirmation

Run roots:

```text
runs_stage61_humanoid_confirm_gpu3/actor_conditioned_w0p10_20ep
runs_stage61_humanoid_confirm_gpu3/base_w0p10_20ep
```

Confirmed 20-episode results:

| variant | risk weight | success | mean length |
| --- | ---: | ---: | ---: |
| original | n/a | 0.80 | 1460.34 |
| filtered base | 0.10 | 0.81 | 1438.11 |
| filtered actor conditioned | 0.10 | 0.83 | 1423.71 |

Task-wise comparison against the matched original 20-episode re-eval:

| variant | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.95 | 0.70 | 1.00 | 0.55 | 0.80 |
| filtered base @0.10 | 0.95 | 0.65 | 1.00 | 0.60 | 0.85 |
| filtered actor conditioned @0.10 | 0.95 | 0.65 | 1.00 | 0.55 | 1.00 |

Key reading:

- the mild penalty recovers the Stage59 over-penalization problem;
- filtered actor-conditioned @ `0.10` is the current best confirmed setting;
- the gain mainly comes from improving the hardest long-horizon `task5`
  without damaging `task1`/`task3`;
- `task2` still drops relative to the original graph, so the method is not yet
  uniformly better across all tasks.

## Current Mainline Conclusion

The strongest confirmed humanoid recipe is now:

```text
relative endpoint support filter
+ contract scorer
+ actor-conditioned refinement
+ mild planner penalty (risk_weight ~= 0.10)
```

This is more precise than the earlier "patch the graph with a fixed strong
penalty" story.

What now looks wrong:

- strong planner penalties such as `0.25` on humanoid;
- interpreting short 5-episode smoke gains as sufficient evidence.

What now looks right:

- graph repair must be separated from penalty calibration;
- actor refinement helps when paired with the repaired graph and a mild
  planner-time intervention.

## Immediate Next Steps

1. Re-run the same mild-weight recipe on a stronger antmaze setting
   (`antmaze-giant-navigate-v0` or `antmaze-large-explore-v0`) with matched
   20-episode reporting.
2. Replace one global `risk_weight` with an adaptive penalty rule based on
   offline uncertainty indicators such as contract entropy, contract support,
   or actor disagreement.
3. Investigate why `task2` regresses under the repaired graph even when
   `task5` improves sharply; this likely points to a planner tradeoff rather
   than a scorer failure.
