# Round 004 GAS Self-Train Code Audit

Generated: 2026-05-21 Asia/Shanghai.

## Scope

Question: whether Round 004 full-budget GAS self-training results are sufficient evidence that the reported GAS performance is biased, given that the run mostly uses upstream GAS source.

Gate context: baseline certification precedes scientific interpretation. This report only audits GAS baseline/source/protocol reproduction. It does not interpret BARS failure modes, oracle headroom, p_bridge, boundary, or integrated BARS results.

Evidence classes:

- Official artifact evaluation: `E1_OFFICIAL_FULL_BUDGET_ARTIFACT_EVAL`.
- Round 004 local self-training: `E4_FULL_BUDGET_TRAINED_METHOD`.
- Source/protocol audit: `E0_PROTOCOL_AUDIT`.

Primary report files:

- `reports/round_003_baseline_certification.md`
- `reports/round_003_gas_official_eval.csv`
- `reports/round_004_gas_selftrain_eval_summary.md`
- `reports/round_004_gas_selftrain_eval_summary.csv`

## Bottom Line

Current evidence does not justify the claim that the public reported GAS result is biased or unreachable.

The strongest counter-evidence is Round 003: official Hugging Face GAS checkpoints evaluated locally with the same `evaluate_gas.py` protocol pass all three certified environments, including `scene-play-v0`:

| env | official checkpoint eval pp | public mean pp | lower bound pp | status |
| --- | ---: | ---: | ---: | --- |
| antmaze-giant-stitch-v0 | 92.0 | 88.3 | 81.1 | PASS |
| antmaze-large-explore-v0 | 96.8 | 94.2 | 88.2 | PASS |
| scene-play-v0 | 79.6 | 73.6 | 57.6 | PASS |

Round 004 local full-budget from-scratch training instead produced:

| env | self-train eval pp | public mean pp | lower bound pp | status |
| --- | ---: | ---: | ---: | --- |
| antmaze-giant-stitch-v0 | 86.4 | 88.3 | 81.1 | PASS lower-bound, below mean |
| antmaze-large-explore-v0 | 99.2 | 94.2 | 88.2 | PASS, above mean |
| scene-play-v0 | 48.8 | 73.6 | 57.6 | FAIL lower-bound |

Therefore the defensible statement is: Round 004 did not reproduce the `scene-play-v0` public-quality baseline from scratch in this local run. Because official artifacts do reproduce the reported-quality result under local evaluation, this is currently a training-reproducibility gap, not evidence of report bias.

## Source Audit

Official source reference checked at audit time:

- GitHub: `https://github.com/qortmdgh4141/GAS`
- `git ls-remote` HEAD: `c9e590fcd6f082de677d332a84e44a1a631da5c5`

Files compared against upstream `main`:

- `pretrain_tdr.py`
- `train_policy.py`
- `evaluate_gas.py`
- `construct_graph.py`
- `O_utils/env_utils.py`
- `O_utils/evaluation.py`
- `O_utils/datasets.py`
- `M_utils/agents/gas.py`
- `M_utils/flax_utils.py`

Material differences from upstream:

| file | local difference | likely result impact |
| --- | --- | --- |
| `O_utils/env_utils.py` | Adds BARS dataset helper, `OGBENCH_DATASET_DIR`, and passes `dataset_dir` to `ogbench.make_env_and_datasets`. | Can affect which local dataset files are used; dataset identity must be recorded before making reproducibility claims. |
| `pretrain_tdr.py` | Adds `--resume_tdr_path` and resumes from serialized agent state. | Non-upstream training path when used. Can change stochastic trajectory after interrupted runs. |
| `train_policy.py` | Adds `--resume_policy_path` and resumes from serialized agent state. | Non-upstream training path when used. It was not used in Round 004 policy training. |
| `pretrain_tdr.py`, `train_policy.py`, `evaluate_gas.py`, `construct_graph.py` | Uses `os.environ.setdefault('MUJOCO_GL', 'egl')` instead of unconditional `egl`; wandb import is routed through local log utils. | Operational/compatibility change; unlikely to be algorithmic for state-based training/eval. |

No diff was found for the core GAS agent/loss implementation in `M_utils/agents/gas.py`, dataset wrapper logic in `O_utils/datasets.py`, evaluation logic in `O_utils/evaluation.py`, or checkpoint serialization in `M_utils/flax_utils.py` against upstream `main`.

## Protocol Audit

The official README commands for the three Round 004 environments specify:

| env | train steps | discount | tdr_expectile | alpha | batch size | p_aug | way_steps | te_threshold | eval rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| antmaze-giant-stitch-v0 | 1,000,000 | 0.995 | 0.999 | 1.0 | 1024 | 0.0 | 8 | 0.99 | 49 + 1 |
| antmaze-large-explore-v0 | 1,000,000 | 0.99 | 0.999 | 0.01 | 1024 | 0.0 | 8 | 0.99 | 49 + 1 |
| scene-play-v0 | 1,000,000 | 0.99 | 0.999 | 1.0 | 1024 | 0.0 | 48 | 0.99 | 49 + 1 |

Round 004 launcher matches these documented flags, uses seed 0, and evaluates 50 rollouts per task via `eval_episodes=49` plus `eval_video_episodes=1`.

Important deviation: all three TDR runs used local resume logic after interruption:

| env | initial TDR log range | resumed TDR log range | policy log range |
| --- | --- | --- | --- |
| antmaze-giant-stitch-v0 | 5k..945k | 905k..1,000k from `params_900000.pkl` | 5k..1,000k |
| antmaze-large-explore-v0 | 5k..950k | 905k..1,000k from `params_900000.pkl` | 5k..1,000k |
| scene-play-v0 | 5k..985k | 905k..1,000k from `params_900000.pkl` | 5k..1,000k |

`M_utils/flax_utils.py` serializes/restores the full agent state dictionary, so the resume is not merely parameters-only. Still, it is not the clean uninterrupted upstream README path, and it discards unsaved progress after 900k from the interrupted process.

## Dataset / Runtime Audit

Runtime package versions:

| package | version |
| --- | --- |
| ogbench | 1.1.5 |
| jax | 0.4.25 |
| jaxlib | 0.4.25 |
| flax | 0.8.5 |
| optax | 0.2.3 |
| gymnasium | 1.1.1 |
| mujoco | 3.1.6 |
| numpy | 1.26.4 |

Round 004 dataset root: `/root/remote/datasets/ogbench`.

Dataset fingerprints:

| file | sha256 |
| --- | --- |
| `antmaze-giant-stitch-v0.npz` | `c697fa4ebd0442d66839e1e4a57092657195e52e8905155903d4d21363eb886e` |
| `antmaze-giant-stitch-v0-val.npz` | `b4d0862d31d1ac014b60deaa732a78bddba0c547489373863ec2b8e30c7edb9d` |
| `antmaze-large-explore-v0.npz` | `8c6303259a60db6bfceb742e2a18016507325614472c10648aee798b580e880b` |
| `antmaze-large-explore-v0-val.npz` | `80fac2570437e83666318e2383cdecfdb4198aceb3d007bd205ae7ccc469a7ce` |
| `scene-play-v0.npz` | `66625e0cd9f2fcf92f5988f4d1bedeb2aa8b1e748316e507e89f4a9afb87c341` |
| `scene-play-v0-val.npz` | `6700c42a1602fa4a9457414395613d9d8ad6350181662a79751ca02dc0423cb2` |

These files are valid zip/npz files per the Round 003 dataset inventory, but this audit does not yet compare them against an upstream published hash manifest.

## Descriptive Artifact Diagnostics

Official vs self-trained keygraph sizes:

| env | official base nodes | selftrain base nodes | official edges | selftrain edges |
| --- | ---: | ---: | ---: | ---: |
| antmaze-giant-stitch-v0 | 1932 | 1985 | 31986 | 33768 |
| antmaze-large-explore-v0 | 2509 | 2514 | 34616 | 33964 |
| scene-play-v0 | 731 | 702 | 62354 | 54066 |

This is descriptive only. It is not a causal failure interpretation.

## Conclusion

Round 004 does not reach report-level performance on `scene-play-v0`, but the local run is not a decisive refutation of the reported GAS result because:

- official full-budget checkpoints pass locally under the same evaluation loop;
- the local source has BARS dataset/resume patches;
- the failed run used an interrupted/resumed TDR training path;
- the run is one seed only;
- dataset identity is recorded but not matched to an upstream hash manifest;
- stochastic JAX training can differ across hardware/runtime versions even with matched flags.

The strongest current claim is: `scene-play-v0` requires a clean source-faithful from-scratch rerun before using Round 004 self-trained artifacts as certified baseline evidence.

Recommended next evidence step: rerun `scene-play-v0` from scratch with `FORCE=1`, `WAIT_FOR_GPU_FREE=1`, no resume, isolated GPU allocation, upstream GAS source checked out at `c9e590fcd6f082de677d332a84e44a1a631da5c5`, the same dataset fingerprints recorded above, and at least three seeds before considering any claim about training reproducibility or reporting bias.
