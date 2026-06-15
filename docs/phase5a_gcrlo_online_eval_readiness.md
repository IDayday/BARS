# Phase 5A gcrlo Online Evaluation Readiness

Phase 5A checks whether the local `gcrlo` conda environment can construct
OGBench environments for closed-loop evaluation.

## External References

- OGBench repository: https://github.com/seohongpark/ogbench
- OGBench project page: https://seohong.me/projects/ogbench/
- Gymnasium MuJoCo docs: https://gymnasium.farama.org/environments/mujoco/
- MuJoCo Python bindings: https://mujoco.readthedocs.io/en/stable/python.html

OGBench exposes Gymnasium-style environments through
`ogbench.make_env_and_datasets(...)`. That means online closed-loop evaluation
is possible when the Python environment can import OGBench, Gymnasium, and the
MuJoCo-related backends.

## gcrlo Dependency State

The active shell is still `navsim`, but online OGBench work should use:

```bash
conda run -n gcrlo ...
```

`gcrlo` already had `gymnasium`, `gym`, `mujoco`, and `Cython`. It did not have
`ogbench` installed as a package, but this repository has a local OGBench source
tree under `external_src/tmd-release`, so the required runtime command is:

```bash
export PYTHONPATH=/mnt/project/BARS/external_src/tmd-release
export OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench
export MUJOCO_GL=egl
```

Scene additionally required `dm_control`, which was installed into `gcrlo`:

```bash
conda run -n gcrlo python -m pip install dm-control==1.0.27
```

This upgraded `mujoco` in `gcrlo` to `3.9.0`.

`d4rl` remains missing, but it is optional for the current OGBench AntMaze/Scene
preflight and should not block Phase 3/5 natural-start rollout work.

## Preflight Results

Commands:

```bash
MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/check_phase3_env.py \
  --dataset_name antmaze-large-stitch-v0 \
  --dataset_dir /mnt/project/offlinerl_datasets/ogbench \
  --output_dir results/phase3/env_preflight_gcrlo

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/check_phase3_env.py \
  --dataset_name scene-play-v0 \
  --dataset_dir /mnt/project/offlinerl_datasets/ogbench \
  --output_dir results/phase3/env_preflight_gcrlo
```

Results:

| dataset | env status | dataset loaded | reset probe | missing required packages |
| --- | --- | --- | --- | --- |
| `antmaze-large-stitch-v0` | `env_available` | yes | `reset_unsupported` | none |
| `scene-play-v0` | `env_available` | yes | `reset_unsupported` | none |

The reset probe failures are semantic, not dependency failures:

- AntMaze: `set_state` expects `qpos, qvel`, but the offline dataset arrays used
  by the probe expose only observations/actions/next observations/terminals.
- Scene: `set_state` expects `qpos, qvel, button_states`; the probe does not
  have a reliable full simulator state for arbitrary dataset observations.

Therefore, edge-level arbitrary reset rollout should remain disabled for now.

## Closed-Loop Smoke

Both environments can run natural-start `reset/step`:

| dataset | observation shape | action shape | one-step status |
| --- | ---: | ---: | --- |
| `antmaze-large-stitch-v0` | `(29,)` | `(8,)` | `env.reset(seed=0)` and `env.step(action)` succeeded |
| `scene-play-v0` | `(40,)` | `(5,)` | `env.reset(seed=0)` and `env.step(action)` succeeded |

## Recommendation

The blocker has moved from `env_unavailable` to `reset_unsupported`.

Phase 5B now provides a minimal natural-start closed-loop smoke test that uses
`info["goal"]` from `env.reset(...)` and trained GCBC checkpoints. The smoke
runs completed in `gcrlo` for AntMaze and Scene, but direct edge-BC GCBC did not
solve either task in 2 episodes x 100 steps. This only validates the online
evaluation interface; it does not validate the graph algorithm.

Phase 5C adds hierarchical support-graph natural-start rollout. It can plan and
execute partial option paths, but the current policies still do not solve the
tasks. AntMaze requires full support-bank fallback and repeatedly replans; Scene
plans a short path but does not reliably complete the first option.

Next evaluation should improve the hierarchical natural-start protocol:

1. `env.reset()`.
2. Infer current cluster from observation.
3. Plan with the repaired support graph and Phase 4O-selected method.
4. Execute GCBC toward the next option goal.
5. Switch options when the destination cluster is reached.
6. Record task success, path completion, failure reason, and replanning count.

Reset-to-state edge execution can be revisited only if future instrumentation
stores exact MuJoCo state references (`qpos`, `qvel`, and Scene button states)
for sampled starts/goals.
