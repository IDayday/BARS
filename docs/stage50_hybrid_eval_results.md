# Stage50 Hybrid Evaluation Results

Last updated: 2026-06-16

## Offline Boundary

Stage50/51 remain offline-RL method-development steps. Contract rows, actor
features, sequence labels, edge scores, and path-safety gates are built only from
fixed local OGBench datasets and frozen GAS artifacts. Closed-loop rollouts below
are used only for reporting and diagnosis, not for training labels, tuning
thresholds, or adding online data.

## Stage50 Closed-Loop Result

Run root:

```text
runs_stage50_hybrid_eval_gpu3/20260616_140855
```

Launcher:

```text
scripts/stage50_launch_hybrid_eval.py
```

All three jobs were launched in parallel with GPU id `3`, using the same
20-episode evaluation protocol as Stage45.

| env | Stage45 CAP-lite | Stage50 hybrid | delta | mean length |
| --- | ---: | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.81 | 0.82 | +0.01 | 738.83 |
| `antmaze-large-explore-v0` | 0.95 | 0.95 | +0.00 | 390.23 |
| `scene-play-v0` | 0.76 | 0.73 | -0.03 | 366.99 |

Interpretation:

- Stage50 does not break AntMaze; giant is slightly higher than Stage45 in this
  20-episode eval.
- Large-explore is unchanged.
- Scene regresses back to the original GAS-level success, so negative sequence
  penalties are not yet safe for manipulation/Scene.

## Offline Diagnosis

Stage49 sequence labels are strong as an offline verifier, but direct negative
penalties can remove useful paths. Scene is the clearest case:

| method | changed vs Stage45 | mean same-traj support | unsupported fraction |
| --- | ---: | ---: | ---: |
| Stage45 CAP-lite | 0.0000 | 0.4508 | 0.1878 |
| Stage50 hybrid | 0.0068 | 0.4295 | 0.1884 |
| Stage51 strict path-safety | 0.0003 | 0.4518 | 0.1877 |

Only 25 cached paths differ between Stage45 and Stage50 on Scene, but those
changes are enough to lower the offline support profile. Conservative drop caps
and positive-only sequence boosts avoid the regression by effectively reverting
to Stage45 paths, which is safe but not a real algorithmic improvement.

For `antmaze-giant-navigate-v0`, strict path-safety rejects all Stage50 candidate
path changes: every candidate has lower support and higher Stage50 contract risk
under the current offline metrics. The observed +0.01 closed-loop gain is
therefore not strong enough evidence that negative sequence penalties are
reliably beneficial.

## Current Conclusion

The current best research framing is:

```text
BARS-CAP-Seq is promising as an offline actor-compatibility verifier, but the
planner should not treat low sequence probability as a hard or direct negative
edge penalty without better calibration.
```

Immediate next direction:

- keep Stage45 broad contract prior as the stable bridge baseline;
- keep Stage49 sequence verifier as a separate reliability signal;
- develop an offline-calibrated sequence gate that distinguishes false negatives
  from genuinely actor-incompatible edges;
- use Stage51 strict path-safety as a safety ablation, not as the final method.
