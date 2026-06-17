# Stage44 Humanoid/Visual GAS Retrain Plan

## Decision

Use local `/mnt/project` datasets for all HumanoidMaze and Visual OGBench work.
Do not let OGBench auto-download data into `~/.ogbench/data`.

The correct default dataset root is:

```bash
/mnt/project/offlinerl_datasets/ogbench
```

`~/.ogbench/data` should contain symlinks to the project files because official
GAS calls OGBench without an explicit `dataset_dir` in several paths.

## Official Configuration Facts

The official GAS README gives the four-stage pipeline:

1. `pretrain_tdr.py`
2. `construct_graph.py`
3. `train_policy.py`
4. `evaluate_gas.py`

It also gives exact command-line flags for state and pixel environments.  The
local registry `configs/stage32_official_gas_protocol_registry.json` mirrors
those settings and is the source used by the launcher.

OGBench officially exposes HumanoidMaze and pixel observations through
`ogbench.make_env_and_datasets`; each goal-conditioned environment has task IDs
1-5 and returns `info["success"]` during evaluation.

## Local Preflight

Verified in the `gcrlo` environment with
`PYTHONPATH=/mnt/project/BARS/external_src/tmd-release`:

| env | observation | action |
| --- | ---: | ---: |
| `humanoidmaze-large-navigate-v0` | `(69,)` | `(21,)` |
| `visual-antmaze-large-explore-v0` | `(64, 64, 3)` | `(8,)` |
| `visual-scene-play-v0` | `(64, 64, 3)` | `(5,)` |

The EGL teardown warning seen during this lightweight preflight is not a dataset
or environment construction failure.

## First Local Retrain Set

Run one official-config local version first:

| env | reason | official-config summary |
| --- | --- | --- |
| `humanoidmaze-large-navigate-v0` | Humanoid checkpoint is not listed in official GAS public checkpoints; old local score is weak. | state encoder, 1M steps, `way_steps=32`, `alpha=0.1` |
| `humanoidmaze-large-stitch-v0` | Stitch/composition humanoid counterpart. | state encoder, 1M steps, `way_steps=32`, `alpha=0.1` |
| `visual-antmaze-large-explore-v0` | Low published GAS score leaves headroom; current Stage32 artifacts missing. | `impala_small`, 500k steps, batch 256, `p_aug=0.5`, `alpha=0.01` |
| `visual-scene-play-v0` | Visual manipulation stress test; current Stage32 artifacts missing. | `impala_small`, 500k steps, batch 256, `p_aug=0.5`, `way_steps=24` |

Visual GAS public checkpoints are listed by the official README, so they can
also be downloaded later for a public-checkpoint comparison.  This stage still
trains local artifacts so the baseline and BARS variants can share the same
data path, metadata, and protocol audit.

## Launcher

Use:

```bash
bash scripts/stage44_launch_humanoid_visual_retrain.sh
```

Default behavior:

- unique artifact root under `artifacts/gas_ogbench_stage44_humanoid_visual_retrain_*`;
- unique run root under `runs_stage44_humanoid_visual_retrain/*`;
- `DOWNLOAD_DATASETS=0`;
- default envs:
  `humanoidmaze-large-navigate-v0,humanoidmaze-large-stitch-v0,visual-antmaze-large-explore-v0,visual-scene-play-v0`;
- default seed: `0`;
- default GPUs: `0,2,3,4`, avoiding the currently busy GPU 1.

The launcher uses `setsid` by default and writes:

- `commands.log`
- `logs/launcher.outer.log`
- `launcher.pid`
- `stage35_full_gas_training_status.csv`

## Monitoring

```bash
OUT_ROOT=runs_stage44_humanoid_visual_retrain/latest
cat "${OUT_ROOT}/launcher.pid"
tail -n 120 "${OUT_ROOT}/logs/launcher.outer.log"
tail -n 40 "${OUT_ROOT}/stage35_full_gas_training_status.csv"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

After training, run the Stage32 inventory on the new artifact root before using
the results as a GAS baseline.

## Interpretation

- Humanoid retraining is required because public GAS README checkpoints do not
  list HumanoidMaze, and the old local Stage32 scores were far below the
  official target range.
- Visual environments are open and public GAS checkpoints exist, but local
  artifacts are currently missing.  Local retraining gives a reproducible
  baseline; official checkpoint download remains a separate verification route.
- No BARS-vs-GAS claim should use these retrains until protocol inventory marks
  checkpoint, keygraph, flags, dataset embeddings, and manifest as valid.
