# Round 006 GAS Config Alignment Audit

Generated: 2026-05-24T01:40+08:00.

## Scope

This audit checks whether `scripts/round006_gas_dynamic_orchestrator.py` launches full-budget GAS training with the same command-line hyperparameters as the official GAS README command templates.

Evidence class: `E4_FULL_BUDGET_TRAINED_METHOD` pending completed checkpoints/evaluation.

This is a baseline-only audit. It does not interpret BARS failure modes, oracle headroom, p_bridge, boundary behavior, or integrated BARS results.

## Primary Evidence

- Official upstream README checked from `https://raw.githubusercontent.com/qortmdgh4141/GAS/main/README.md`.
- Local README: `external_src/GAS/README.md`.
- `cmp` result: upstream README and local README are identical.
- Official agent config checked from `https://raw.githubusercontent.com/qortmdgh4141/GAS/main/M_utils/agents/gas.py`.
- Local agent config: `external_src/GAS/M_utils/agents/gas.py`.
- `cmp` result: upstream `gas.py` and local `gas.py` are identical.
- Programmatic comparison between all Round006 target env configs and README command templates found `0` hyperparameter mismatches.

## Matched Parameters

Round006 matches the official README for all 24 OGBench target environments on:

- `train_steps`
- `log_interval`
- `save_interval`
- `te_threshold`
- `eval_on_cpu`
- `eval_episodes`
- `eval_video_episodes`
- `eval_final_goal_threshold`
- `agent_config.encoder`
- `agent_config.discount`
- `agent_config.tdr_expectile`
- `agent_config.alpha`
- `agent_config.batch_size`
- `agent_config.p_aug`
- `agent_config.way_steps`

The matched environment groups are:

- State AntMaze and Scene: `1,000,000` TDR steps and `1,000,000` policy steps.
- HumanoidMaze additional results: `1,000,000` TDR steps and `1,000,000` policy steps.
- Visual OGBench: `500,000` TDR steps and `500,000` policy steps.

## Current Live Evidence

Example active command from `runs_round006_gas_dynamic/antmaze-giant-navigate-v0/seed42/pretrain_tdr.log`:

```text
python pretrain_tdr.py --train_steps 1000000 --log_interval 5000 --save_interval 100000 --env_name antmaze-giant-navigate-v0 --seed 42 --gpu 0 --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
```

Example persisted `flags.json` files record `resume_tdr_path: null` for the currently launched fresh jobs, so the active first batch is from-scratch rather than resumed.

## Intentional Local Compatibility Differences

The local GAS source has compatibility changes relative to upstream official scripts. These are engineering/runtime changes, not GAS hyperparameter or loss/network changes:

- WandB import is routed through `O_utils.log_utils` so logging can run with `WANDB_DISABLED=true`.
- `MUJOCO_GL` uses `setdefault` instead of unconditional `egl`, allowing cluster/runtime override.
- Optional `resume_tdr_path` and `resume_policy_path` flags were added; current fresh jobs have null resume paths.
- `O_utils/env_utils.py` accepts `OGBENCH_DATASET_DIR` and uses the BARS OGBench downloader before calling `ogbench.make_env_and_datasets(..., compact_dataset=False)`.
- `K_utils/graph_builder.py` has the local call-signature compatibility fix already tracked in `third_party/gas_stage22.patch`.

These differences should be disclosed in final reproduction notes. None changes the README-matched GAS hyperparameters.

## Runtime Environment Note

The local `gcrlo` environment reports:

- `jax 0.4.25`
- `jaxlib 0.4.25`
- `mujoco 3.1.6`
- `ogbench 1.1.5`
- `flax 0.8.5`
- `optax 0.2.3`

The official README states Python 3.9, MuJoCo 3.1.6, and JAX >= 0.4.26 (CUDA 12 build). MuJoCo matches; JAX/JAXlib are one minor version below the README requirement and should be recorded as environment-version drift unless the environment is upgraded for a later strict rerun.

## Decision

Round006 training hyperparameters and step budgets are aligned with official GAS README command templates for all targeted OGBench environments. No parameter correction is required for the currently running queue.

