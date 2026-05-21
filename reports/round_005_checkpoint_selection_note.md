# Round 005 Checkpoint Selection / Reproducibility Note

Generated: 2026-05-21 Asia/Shanghai.

## Claim

Round 005 provides evidence that the released official GAS checkpoints may not represent ordinary expected from-scratch training performance under the available public source and README hyperparameters.

This should be recorded as a reproducibility concern and a plausible checkpoint/seed-selection effect.

## Evidence

Round 003 official checkpoint evaluation passed locally under the same GAS evaluator:

| env | official checkpoint pp | public mean pp |
| --- | ---: | ---: |
| antmaze-giant-stitch-v0 | 92.0 | 88.3 |
| antmaze-large-explore-v0 | 96.8 | 94.2 |
| scene-play-v0 | 79.6 | 73.6 |

Round 005 full-budget from-scratch 3-seed rerun produced lower means:

| env | seed scores pp | 3-seed mean pp | gap vs official checkpoint pp | gap vs public mean pp |
| --- | --- | ---: | ---: | ---: |
| antmaze-giant-stitch-v0 | 84.4, 86.8, 88.4 | 86.5 | -5.5 | -1.8 |
| antmaze-large-explore-v0 | 74.0, 99.6, 94.4 | 89.3 | -7.5 | -4.9 |
| scene-play-v0 | 60.8, 66.4, 50.8 | 59.3 | -20.3 | -14.3 |

## Interpretation

The official checkpoint result is valid as an artifact-level result: it demonstrates that high-performing weights exist and that our local evaluator can reproduce them.

However, a single released checkpoint does not establish that the reported result reflects ordinary expected training performance across random seeds. It may correspond to a favorable seed, a favorable checkpoint, undisclosed training details, or another selection effect.

Round 005 is therefore the more credible estimate of reproducible from-scratch GAS performance under our recorded protocol. The gap between Round 005 and the official checkpoint should be explicitly stated in any report.

## Boundary

This note does not claim intentional report bias. It records that the available evidence supports a serious reproducibility concern and a plausible checkpoint/seed-selection effect.

Primary evidence file: `reports/round_005_final_conclusion.md`.
