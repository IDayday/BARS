# tmd-test Protocol

## Branch commit

- Original required branch: `stage25`
- User-authorized branch for this run: `stage25-protocol-oracle-drift`
- Experiment name: `tmd-test` / `tmd_test`

## Envs

- Smoke: `antmaze-medium-stitch-v0`
- Pilot target: `antmaze-medium-stitch-v0`, `antmaze-medium-navigate-v0` when env-specific TMD checkpoints are available.

## Seeds

- Smoke: `0`
- Pilot: `0`

## Evaluation episodes

- Smoke: `2`
- Pilot: `50`, optionally `100` if resources allow.

## Fallback mode

- Default and current mode: `fallback=none`.

## Success definition

- Online environment success is read from `info.success`, `info.goal_achieved`, `info.is_success`, or `info.episode.success` when present.

## task_id schedule

- Smoke uses task `1`.
- Pilot should expand to all available OGBench task ids after smoke.

## Goal source

- Graph virtual targets use terminal observations from local OGBench validation data under `/mnt/project/offlinerl_datasets/ogbench`.
- Online evaluation uses `env.reset(options={"task_id": task_id})` goal information.

## Checkpoint source

- Official TMD implementation: `external_src/tmd-release`.
- Smoke checkpoint: `artifacts/tmd_gas_script_smoke/exp_tmd/smoke_tmd_actor/antmaze-medium-stitch-v0_sd000__2026-05-21_17-53-08/params_1000.pkl`.

## Artifact source

- Datasets: `/mnt/project/offlinerl_datasets/ogbench`.
- Outputs: `artifacts/tmd_test`, `runs_tmd_test`, and `reports/tmd_test_*`.
