# Round 005 GAS 3-Seed Self-Training Final Conclusion

Generated: 2026-05-21 15:59:33 Asia/Shanghai.

## Scope

Evidence class: `E4_FULL_BUDGET_TRAINED_METHOD`.

Gate context: GAS baseline reproduction only. This report does not interpret BARS failure modes, oracle headroom, p_bridge, boundary, or integrated BARS results.

Protocol:

- Envs: `antmaze-giant-stitch-v0`, `antmaze-large-explore-v0`, `scene-play-v0`.
- Seeds: `0,1,2`.
- Training: TDR 1,000,000 steps + policy 1,000,000 steps.
- Official weights used: false.
- Artifact root: `artifacts/gas_selftrain_round005`.
- Run root: `runs_round005_gas_selftrain`.
- Evaluation: GAS `evaluate_gas.py`, `eval_episodes=49` + `eval_video_episodes=1` = 50 rollouts/task.

All 9 jobs completed evaluation.

## Main Result

Round 005 does not reproduce the strict public reported mean on any of the three environments.

Under the repo's public lower-bound gate, all three environments pass, but `antmaze-large-explore-v0` and `scene-play-v0` pass with small margins and show substantial seed variability.

| env | seed scores pp | 3-seed mean pp | sample std pp | public mean pp | public std pp | lower bound pp | vs public mean pp | vs lower bound pp | strict public mean | lower-bound gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| antmaze-giant-stitch-v0 | 84.4, 86.8, 88.4 | 86.5 | 2.0 | 88.3 | 3.6 | 81.1 | -1.8 | +5.4 | FAIL | PASS |
| antmaze-large-explore-v0 | 74.0, 99.6, 94.4 | 89.3 | 13.5 | 94.2 | 3.0 | 88.2 | -4.9 | +1.1 | FAIL | PASS |
| scene-play-v0 | 60.8, 66.4, 50.8 | 59.3 | 7.9 | 73.6 | 8.0 | 57.6 | -14.3 | +1.7 | FAIL | PASS |

## Comparison To Official Checkpoint Evaluation

Round 003 official Hugging Face checkpoints evaluated locally under the same GAS evaluation loop reached public-quality performance:

| env | Round 003 official checkpoint pp | Round 005 self-train mean pp | gap pp |
| --- | ---: | ---: | ---: |
| antmaze-giant-stitch-v0 | 92.0 | 86.5 | -5.5 |
| antmaze-large-explore-v0 | 96.8 | 89.3 | -7.5 |
| scene-play-v0 | 79.6 | 59.3 | -20.3 |

This means the local evaluator and official artifacts can reproduce public-quality scores, while local full-budget from-scratch training does not match those official artifacts.

## Reproducibility And Checkpoint-Selection Note

Round 005 should be treated as more representative evidence for the algorithm's expected from-scratch performance under the recorded local protocol than a single official checkpoint evaluation.

The official checkpoint is an important artifact-level baseline: it proves that a high-performing set of weights exists and that our local evaluator can reproduce public-quality scores. However, it does not by itself establish that the reported table reflects an unbiased average over ordinary training seeds. A single released checkpoint may correspond to a favorable random seed, favorable checkpoint selection, unreported training-condition details, or other selection effects.

Round 005 repeats full-budget training over three seeds per environment and shows materially lower means than the official checkpoint on all three environments:

- `antmaze-giant-stitch-v0`: 3-seed mean `86.5pp` vs official checkpoint `92.0pp`.
- `antmaze-large-explore-v0`: 3-seed mean `89.3pp` vs official checkpoint `96.8pp`.
- `scene-play-v0`: 3-seed mean `59.3pp` vs official checkpoint `79.6pp`.

This is especially important for `scene-play-v0`, where the from-scratch 3-seed mean is `20.3pp` below the official checkpoint and `14.3pp` below the public reported mean.

The appropriate recorded interpretation is:

- The official checkpoint result remains valid as an artifact-specific result.
- The Round 005 3-seed rerun is stronger evidence for reproducible algorithm performance under the available source, README hyperparameters, local dataset, and local runtime.
- The gap supports a serious reproducibility concern and a plausible checkpoint/seed-selection effect.
- It does not strictly prove intentional report bias without additional evidence about the authors' seed policy, checkpoint selection procedure, exact runtime stack, and all training seeds.

## Task-Level Notes

Descriptive only:

- `antmaze-large-explore-v0` has a low seed0 (`74.0pp`) while seed1/seed2 are high (`99.6pp`, `94.4pp`), indicating high training variance in this run.
- `scene-play-v0` remains consistently below public mean across all three seeds. The weakest task-level scores are on `task4_put_in_drawer` and `task5_rearrange_hard`.
- `antmaze-giant-stitch-v0` is close to public mean but still below it on the 3-seed mean.

## Conclusion

Round 005 supports the following claims:

1. Strict reported-mean reproduction from scratch failed on all three tested GAS environments.
2. Lower-bound baseline certification passes on all three environments, but `large-explore` and `scene-play` have narrow margins.
3. The main unresolved issue is training reproducibility / missing training-condition sensitivity, not evaluation mismatch: official full-budget checkpoints pass locally, but full-budget from-scratch training under the available source/config does not reach official checkpoint quality.
4. The repeated 3-seed from-scratch result should be reported as the more credible estimate of expected algorithm performance under our reproducible protocol than the single released official checkpoint.
5. The official checkpoint may reflect a favorable seed or checkpoint-selection effect; this is a serious reproducibility concern that should be stated explicitly.

Round 005 does not by itself prove intentional public-report bias, because official checkpoints evaluated locally still reproduce public-quality performance. It does, however, weaken the interpretation that the public reported number is an easily reproducible ordinary-seed performance level.

## Evidence Files

Primary launch and audit reports:

- `reports/round_005_gas_selftrain_3seed_launch.md`
- `reports/round_005_gas_selftrain_3seed_jobs.tsv`
- `reports/round_005_gas_selftrain_direct_scene_jobs.tsv`
- `reports/round_004_gas_selftrain_code_audit.md`
- `reports/round_003_baseline_certification.md`
- `reports/round_003_gas_official_eval.csv`

Completed status files:

- `runs_round005_gas_selftrain/antmaze-giant-stitch-v0/seed0/status.json`
- `runs_round005_gas_selftrain/antmaze-giant-stitch-v0/seed1/status.json`
- `runs_round005_gas_selftrain/antmaze-giant-stitch-v0/seed2/status.json`
- `runs_round005_gas_selftrain/antmaze-large-explore-v0/seed0/status.json`
- `runs_round005_gas_selftrain/antmaze-large-explore-v0/seed1/status.json`
- `runs_round005_gas_selftrain/antmaze-large-explore-v0/seed2/status.json`
- `runs_round005_gas_selftrain/scene-play-v0/seed0/status.json`
- `runs_round005_gas_selftrain/scene-play-v0/seed1/status.json`
- `runs_round005_gas_selftrain/scene-play-v0/seed2/status.json`

Completed eval CSVs:

- `artifacts/gas_selftrain_round005/antmaze-giant-stitch-v0/seed0/policy/round005_selftrain_antmaze-giant-stitch_seed0/antmaze-giant-stitch-v0_sd000__2026-05-21_12-35-44/eval.csv`
- `artifacts/gas_selftrain_round005/antmaze-giant-stitch-v0/seed1/policy/round005_selftrain_antmaze-giant-stitch_seed1/antmaze-giant-stitch-v0_sd001__2026-05-21_13-03-58/eval.csv`
- `artifacts/gas_selftrain_round005/antmaze-giant-stitch-v0/seed2/policy/round005_selftrain_antmaze-giant-stitch_seed2/antmaze-giant-stitch-v0_sd002__2026-05-21_12-35-28/eval.csv`
- `artifacts/gas_selftrain_round005/antmaze-large-explore-v0/seed0/policy/round005_selftrain_antmaze-large-explore_seed0/antmaze-large-explore-v0_sd000__2026-05-21_12-32-21/eval.csv`
- `artifacts/gas_selftrain_round005/antmaze-large-explore-v0/seed1/policy/round005_selftrain_antmaze-large-explore_seed1/antmaze-large-explore-v0_sd001__2026-05-21_12-09-05/eval.csv`
- `artifacts/gas_selftrain_round005/antmaze-large-explore-v0/seed2/policy/round005_selftrain_antmaze-large-explore_seed2/antmaze-large-explore-v0_sd002__2026-05-21_12-35-42/eval.csv`
- `artifacts/gas_selftrain_round005/scene-play-v0/seed0/policy/round005_selftrain_scene-play_seed0/scene-play-v0_sd000__2026-05-21_12-37-40/eval.csv`
- `artifacts/gas_selftrain_round005/scene-play-v0/seed1/policy/round005_selftrain_scene-play_seed1/scene-play-v0_sd001__2026-05-21_13-06-05/eval.csv`
- `artifacts/gas_selftrain_round005/scene-play-v0/seed2/policy/round005_selftrain_scene-play_seed2/scene-play-v0_sd002__2026-05-21_12-37-23/eval.csv`
