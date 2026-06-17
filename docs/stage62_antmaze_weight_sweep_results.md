# Stage62 AntMaze Weight-Sweep Results

Last updated: 2026-06-16

## Offline Boundary

Stage62 remains within the offline RL setting.

- Contract scorers and actor-conditioned labels are trained only from fixed
  offline datasets and frozen GAS artifacts.
- Graph patching changes planner-time edge costs only.
- Environment rollouts are used only for reporting.

## Code Added

New code:

- `scripts/stage62_launch_official_contract_eval.py`

Purpose:

- generic official-GAS evaluation launcher;
- supports `original`, `rows:`-based scored-row aggregation, and direct
  `edge:`-based patch evaluation;
- allows matched comparisons across AntMaze/Humanoid official artifacts on one
  GPU;
- tolerates concurrent `latest` symlink races when multiple launchers run in
  parallel.

## Motivation

Stage61 on humanoid showed that:

- strong planner penalties such as `risk_weight=0.25` can wash out an otherwise
  useful offline graph repair signal;
- a milder penalty near `0.10` is more stable.

Stage62 tests whether the same calibration rule transfers to official AntMaze
artifacts.

## Giant Navigate: 10-Episode Sweep

Run roots:

```text
runs_stage62_official_contract_eval_gpu3/giant_nav_w0p10_10ep
runs_stage62_official_contract_eval_gpu3/giant_nav_hybrid_w0p15_10ep
```

Variants:

- `original`
- `stage45_base`: Stage45 global contract scorer
- `stage48_actor_conditioned`: Stage48 actor-conditioned scorer
- `stage49_hybrid`: Stage45 prior + Stage49 sequence verifier hybrid

Results:

| variant | weight | success |
| --- | ---: | ---: |
| original | n/a | 0.78 |
| stage45 base | 0.10 | 0.84 |
| stage48 actor-conditioned | 0.10 | 0.78 |
| stage49 hybrid | 0.10 | 0.84 |
| stage49 hybrid | 0.15 | 0.84 |

Reading:

- mild penalty clearly outperforms the matched original in this short sweep;
- Stage48 actor-conditioned labels do not help here;
- Stage49 hybrid is competitive with the simpler Stage45 base, but not better
  yet.

## Giant Navigate: 20-Episode Confirmation

Run root:

```text
runs_stage62_official_contract_eval_gpu3/giant_nav_w0p10_20ep
```

Results:

| variant | weight | success | mean length |
| --- | ---: | ---: | ---: |
| original | n/a | 0.77 | 760.62 |
| stage45 base | 0.10 | 0.78 | 755.00 |
| stage49 hybrid | 0.10 | 0.78 | 767.27 |

Task-wise:

| variant | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.50 | 1.00 | 0.75 | 0.80 | 0.80 |
| stage45 base | 0.35 | 0.85 | 0.95 | 0.95 | 0.80 |
| stage49 hybrid | 0.45 | 0.90 | 0.85 | 1.00 | 0.70 |

Interpretation:

- the `0.84` 10-episode result was optimistic;
- after confirmation, both Stage45 base and Stage49 hybrid keep only a small
  `+0.01` gain over the matched original;
- Stage49 hybrid redistributes performance across tasks rather than delivering
  a clean uniform gain;
- Stage49 hybrid is therefore not yet stronger than Stage45 base on official
  giant-navigate.

## Large Explore: 10-Episode Sweep

Run root:

```text
runs_stage62_official_contract_eval_gpu3/large_explore_w0p10_10ep
```

Results:

| variant | weight | success |
| --- | ---: | ---: |
| original | n/a | 0.90 |
| stage45 base | 0.10 | 0.92 |
| stage48 actor-conditioned | 0.10 | 0.92 |
| stage49 hybrid | 0.10 | 0.92 |

Short-sweep reading:

- mild penalty gives a small positive signal on a high-baseline environment;
- all three patched variants behave almost identically at this horizon.

## Large Explore: 20-Episode Confirmation

Run root:

```text
runs_stage62_official_contract_eval_gpu3/large_explore_w0p10_20ep
```

Results:

| variant | weight | success | mean length |
| --- | ---: | ---: | ---: |
| original | n/a | 0.94 | 419.59 |
| stage45 base | 0.10 | 0.94 | 410.90 |
| stage49 hybrid | 0.10 | 0.94 | 411.22 |

Interpretation:

- the 10-episode gain vanishes under 20-episode confirmation;
- mild penalty does not hurt, but the environment is already close to ceiling;
- large-explore is not the best place to demonstrate algorithmic advantage.

## Main Takeaways

1. The Stage61 calibration lesson transfers: strong penalty was too aggressive,
   and `risk_weight=0.10` is much safer.
2. On official AntMaze, mild penalty avoids regression and can keep small gains.
3. Stage49 hybrid is still not clearly better than the simpler Stage45 base on
   AntMaze official artifacts.
4. Stage48 actor-conditioned labels remain weak on AntMaze in closed loop,
   despite their offline diagnostic value.
5. Short 10-episode gains are still too noisy; matched 20-episode confirmation
   changes the story materially.

## Current AntMaze Position

The best current claim is conservative:

```text
offline contract-based graph patching with mild risk weight
can produce small but repeatable gains on lower-baseline AntMaze settings
without harming already-strong settings.
```

What we still cannot claim:

- that the sequence-level hybrid is already the dominant AntMaze method;
- that actor-conditioned labels alone improve AntMaze execution;
- that large-explore provides strong headroom for the paper headline.

## Recommended Next Step

The next improvement should not be another flat global weight sweep.

The results point to:

1. adaptive penalty composition instead of one global `risk_weight`;
2. task/edge-local calibration using contract uncertainty, sequence evidence,
   and actor disagreement;
3. deeper analysis of giant-navigate task tradeoffs, because Stage49 hybrid
   improves some tasks while harming others.
