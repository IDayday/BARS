# Round 005 Completed Interim Conclusion

Generated: 2026-05-21 15:14:38 Asia/Shanghai.

Status: superseded by `reports/round_005_final_conclusion.md` after all 9 jobs completed.

## Scope

Evidence class: `E4_FULL_BUDGET_TRAINED_METHOD` for rows with completed `eval.csv`.

Gate context: GAS baseline reproduction only. This report does not interpret BARS failure modes, oracle headroom, p_bridge, boundary, or integrated BARS results.

## Completion State

As of this report, 7 of 9 Round 005 jobs have completed evaluation:

| env | completed seeds | pending seeds |
| --- | --- | --- |
| antmaze-giant-stitch-v0 | 0, 2 | 1 |
| antmaze-large-explore-v0 | 0, 1, 2 | none |
| scene-play-v0 | 0, 2 | 1 |

## Completed Scores

Public lower bound is `public_mean_pp - max(2 * public_std_pp, 5.0)`.

| env | completed seed scores pp | n | completed mean pp | public mean pp | lower bound pp | current interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| antmaze-giant-stitch-v0 | 84.4, 88.4 | 2 | 86.4 | 88.3 | 81.1 | partial result passes lower bound but is below public mean; wait for seed1 before final 3-seed conclusion |
| antmaze-large-explore-v0 | 74.0, 99.6, 94.4 | 3 | 89.3 | 94.2 | 88.2 | complete 3-seed result passes lower bound narrowly but fails strict public mean; high seed variance |
| scene-play-v0 | 60.8, 50.8 | 2 | 55.8 | 73.6 | 57.6 | partial result is below lower bound; strict public mean is already mathematically unreachable for the final 3-seed mean because even a 100.0 seed1 gives 70.5 |

## Evidence Files

Completed eval CSVs:

- `artifacts/gas_selftrain_round005/antmaze-giant-stitch-v0/seed0/policy/round005_selftrain_antmaze-giant-stitch_seed0/antmaze-giant-stitch-v0_sd000__2026-05-21_12-35-44/eval.csv`
- `artifacts/gas_selftrain_round005/antmaze-giant-stitch-v0/seed2/policy/round005_selftrain_antmaze-giant-stitch_seed2/antmaze-giant-stitch-v0_sd002__2026-05-21_12-35-28/eval.csv`
- `artifacts/gas_selftrain_round005/antmaze-large-explore-v0/seed0/policy/round005_selftrain_antmaze-large-explore_seed0/antmaze-large-explore-v0_sd000__2026-05-21_12-32-21/eval.csv`
- `artifacts/gas_selftrain_round005/antmaze-large-explore-v0/seed1/policy/round005_selftrain_antmaze-large-explore_seed1/antmaze-large-explore-v0_sd001__2026-05-21_12-09-05/eval.csv`
- `artifacts/gas_selftrain_round005/antmaze-large-explore-v0/seed2/policy/round005_selftrain_antmaze-large-explore_seed2/antmaze-large-explore-v0_sd002__2026-05-21_12-35-42/eval.csv`
- `artifacts/gas_selftrain_round005/scene-play-v0/seed0/policy/round005_selftrain_scene-play_seed0/scene-play-v0_sd000__2026-05-21_12-37-40/eval.csv`
- `artifacts/gas_selftrain_round005/scene-play-v0/seed2/policy/round005_selftrain_scene-play_seed2/scene-play-v0_sd002__2026-05-21_12-37-23/eval.csv`

Status files:

- `runs_round005_gas_selftrain/*/seed*/status.json`

## Interim Conclusion

We can make one complete environment-level conclusion now:

- `antmaze-large-explore-v0`: Round 005 full-budget from-scratch 3-seed training does not reach the public reported mean, but it does pass the repo lower-bound certification threshold narrowly.

We can also make one bounded mathematical conclusion:

- `scene-play-v0`: Round 005 final 3-seed mean cannot reach the public reported mean anymore, regardless of pending seed1. It may still pass or fail the lower-bound gate depending on seed1; seed1 must score at least 61.2 pp for the final 3-seed mean to reach the 57.6 pp lower bound.

We should not make a final all-environment Round 005 conclusion until `antmaze-giant-stitch-v0` seed1 and `scene-play-v0` seed1 complete evaluation.

This interim result still does not establish public-report bias, because Round 003 official Hugging Face checkpoints reproduced public-quality scores locally under the same evaluation loop.
