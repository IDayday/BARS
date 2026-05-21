# Round 004 GAS Self-Train Eval Summary

Generated from Round 004 self-trained GAS artifacts after all three jobs completed.

## Gate Context

- Baseline-first rule: this report evaluates GAS baseline reproduction only; it does not interpret BARS failure modes, oracle headroom, p_bridge, boundary, or integrated BARS results.
- Evidence class for current self-trained rows: `E4_FULL_BUDGET_TRAINED_METHOD`.
- Public target evidence class: `E1_BASELINE_REGISTRY`, sourced from `reports/round_002_baseline_registry.csv` / `research_state/baseline_cards/*__GAS.json`.
- Public source: GAS ICML 2025 / OpenReview PDF, Table 1 (`https://openreview.net/pdf?id=73EwiOrN8W`).
- Certification threshold used by the repo gate: `lower_bound_pp = public_mean_pp - max(2 * public_std_pp, 5.0)`.
- Current run used local full-budget training from scratch, not official Hugging Face weights.
- Evaluation protocol: `evaluate_gas.py`, seed 0, 5 task ids, `eval_episodes=49` plus `eval_video_episodes=1` = 50 rollouts per task.

## Summary

Strict reported mean result: Round 004 reaches the reported mean on 1 of 3 environments.

Baseline lower-bound gate result: Round 004 passes the public lower-bound gate on 2 of 3 environments; `scene-play-v0` fails public reproduction under this self-trained checkpoint.

| env | self-train score pp | public mean pp | lower bound pp | vs mean pp | vs lower pp | reported mean reached | gate status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| antmaze-giant-stitch-v0 | 86.4 | 88.3 | 81.1 | -1.9 | +5.3 | no | PASS_BASELINE_CERTIFICATION |
| antmaze-large-explore-v0 | 99.2 | 94.2 | 88.2 | +5.0 | +11.0 | yes | PASS_BASELINE_CERTIFICATION |
| scene-play-v0 | 48.8 | 73.6 | 57.6 | -24.8 | -8.8 | no | FAIL_PUBLIC_REPRODUCTION |

## Task-Level Scores

Task-level scores are descriptive only; no causal failure interpretation is made here.

| env | task scores pp |
| --- | --- |
| antmaze-giant-stitch-v0 | task1=72.0; task2=98.0; task3=82.0; task4=90.0; task5=90.0 |
| antmaze-large-explore-v0 | task1=98.0; task2=98.0; task3=100.0; task4=100.0; task5=100.0 |
| scene-play-v0 | task1_open=80.0; task2_unlock_and_lock=74.0; task3_rearrange_medium=42.0; task4_put_in_drawer=16.0; task5_rearrange_hard=32.0 |

## Evidence Files

- Current eval CSVs:
  - `artifacts/gas_selftrain_round004/antmaze-giant-stitch-v0/seed0/policy/round004_selftrain_antmaze-giant-stitch_seed0/antmaze-giant-stitch-v0_sd000__2026-05-21_08-09-27/eval.csv`
  - `artifacts/gas_selftrain_round004/antmaze-large-explore-v0/seed0/policy/round004_selftrain_antmaze-large-explore_seed0/antmaze-large-explore-v0_sd000__2026-05-21_08-15-06/eval.csv`
  - `artifacts/gas_selftrain_round004/scene-play-v0/seed0/policy/round004_selftrain_scene-play_seed0/scene-play-v0_sd000__2026-05-21_08-09-49/eval.csv`
- Run completion statuses:
  - `runs_round004_gas_selftrain/antmaze-giant-stitch-v0/seed0/status.json`
  - `runs_round004_gas_selftrain/antmaze-large-explore-v0/seed0/status.json`
  - `runs_round004_gas_selftrain/scene-play-v0/seed0/status.json`
- Public targets:
  - `reports/round_002_baseline_registry.csv`
  - `research_state/baseline_cards/antmaze-giant-stitch-v0__GAS.json`
  - `research_state/baseline_cards/antmaze-large-explore-v0__GAS.json`
  - `research_state/baseline_cards/scene-play-v0__GAS.json`

## Boundary

This round does not certify the self-trained `scene-play-v0` GAS baseline. Do not use the `scene-play-v0` self-trained checkpoint for downstream BARS scientific interpretation unless the baseline is rerun/fixed and passes the public lower-bound gate, or an official certified checkpoint is used instead.
