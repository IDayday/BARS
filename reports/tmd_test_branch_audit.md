# tmd-test Branch Audit

## Git state
- original required branch: stage25
- user-authorized branch for this run: stage25-protocol-oracle-drift
- commit: 595930c1dc8fe0680b182976572b7a43ef5f3aeb
- latest commit: 595930c Publish Round006 lightweight results
- dirty state:

```text
 M commands/round_006_gas_dynamic_launch.sh
 M external_src/GAS/D_utils/d4rl_env_utils.py
 M reports/round_006_gas_dynamic_jobs.tsv
 M reports/round_006_gas_dynamic_launch.md
 M reports/round_006_ogbench_download_status.tsv
 M rounds/round_006/gas_dynamic_jobs.tsv
 M rounds/round_006/ogbench_download_status.tsv
 M scripts/round006_gas_dynamic_orchestrator.py
 M scripts/setup_gas_repo.sh
?? bars/tmd_test/
?? configs/tmd_test/
?? external_src/tmd-release/
?? reports/tmd_test_branch_audit.md
?? reports/tmd_test_decisions.md
?? reports/tmd_test_eval_summary.md
?? reports/tmd_test_graph_diagnostics.md
?? reports/tmd_test_protocol.md
?? rounds/round_006/gas_dynamic_remaining_jobs.tsv
?? scripts/tmd_test_analyze.py
?? scripts/tmd_test_audit.sh
?? scripts/tmd_test_construct_graph.sh
?? scripts/tmd_test_eval.sh
?? scripts/tmd_test_pilot.sh
?? scripts/tmd_test_prepare_tmd.sh
```

## Directory overview

```text
.
./.cache
./.cache/jax_cuda12_3
./.cache/jax_cuda12_3/jaxlib
./.cache/jax_cuda12_3/jaxlib-0.4.25+cuda12.cudnn89.dist-info
./.cache/jax_cuda12_3/nvidia
./.cache/jax_cuda12_3/nvidia_cublas_cu12-12.3.4.1.dist-info
./.cache/jax_cuda12_3/nvidia_cuda_cupti_cu12-12.3.101.dist-info
./.cache/jax_cuda12_3/nvidia_cuda_nvcc_cu12-12.3.107.dist-info
./.cache/jax_cuda12_3/nvidia_cuda_nvrtc_cu12-12.9.86.dist-info
./.cache/jax_cuda12_3/nvidia_cuda_runtime_cu12-12.3.101.dist-info
./.cache/jax_cuda12_3/nvidia_cudnn_cu12-8.9.7.29.dist-info
./.cache/jax_cuda12_3/nvidia_cufft_cu12-11.0.12.1.dist-info
./.cache/jax_cuda12_3/nvidia_cusolver_cu12-11.5.4.101.dist-info
./.cache/jax_cuda12_3/nvidia_cusparse_cu12-12.2.0.103.dist-info
./.cache/jax_cuda12_3/nvidia_nccl_cu12-2.19.3.dist-info
./.cache/jax_cuda12_3/nvidia_nvjitlink_cu12-12.3.101.dist-info
./.cache/wheels
./.git
./.git/branches
./.git/hooks
./.git/info
./.git/logs
./.git/logs/refs
./.git/objects
./.git/objects/01
./.git/objects/0b
./.git/objects/0d
./.git/objects/0e
./.git/objects/0f
./.git/objects/10
./.git/objects/12
./.git/objects/14
./.git/objects/15
./.git/objects/1c
./.git/objects/1f
./.git/objects/20
./.git/objects/22
./.git/objects/23
./.git/objects/28
./.git/objects/29
./.git/objects/2e
./.git/objects/36
./.git/objects/39
./.git/objects/3b
./.git/objects/44
./.git/objects/47
./.git/objects/48
./.git/objects/49
./.git/objects/4a
./.git/objects/4d
./.git/objects/4e
./.git/objects/4f
./.git/objects/56
./.git/objects/59
./.git/objects/63
./.git/objects/64
./.git/objects/65
./.git/objects/69
./.git/objects/70
./.git/objects/73
./.git/objects/75
./.git/objects/76
./.git/objects/77
./.git/objects/78
./.git/objects/80
./.git/objects/82
./.git/objects/88
./.git/objects/89
./.git/objects/8a
./.git/objects/8f
./.git/objects/91
./.git/objects/95
./.git/objects/9b
./.git/objects/9d
./.git/objects/9f
./.git/objects/a4
./.git/objects/ae
./.git/objects/b3
./.git/objects/b4
./.git/objects/bd
./.git/objects/c7
./.git/objects/c8
./.git/objects/c9
./.git/objects/cb
./.git/objects/d1
./.git/objects/d2
./.git/objects/da
./.git/objects/dc
./.git/objects/e4
./.git/objects/e7
./.git/objects/e8
./.git/objects/e9
./.git/objects/ea
./.git/objects/ef
./.git/objects/f2
./.git/objects/f5
./.git/objects/fe
./.git/objects/info
./.git/objects/pack
./.git/refs
./.git/refs/heads
./.git/refs/remotes
./.git/refs/tags
./.vscode
./_data
./_data/d4rl
./_data/ogbench
./artifacts
./artifacts/detach_test_163613
./artifacts/gas_official_full_20260521
./artifacts/gas_official_full_20260521/antmaze-giant-navigate-v0
./artifacts/gas_official_full_20260521/antmaze-giant-stitch-v0
./artifacts/gas_official_full_20260521/antmaze-large-explore-v0
./artifacts/gas_official_full_20260521/antmaze-large-navigate-v0
./artifacts/gas_official_full_20260521/antmaze-large-stitch-v0
./artifacts/gas_official_full_20260521/antmaze-medium-explore-v0
./artifacts/gas_official_full_20260521/antmaze-medium-navigate-v0
./artifacts/gas_official_full_20260521/antmaze-medium-stitch-v0
./artifacts/gas_official_full_20260521/kitchen-partial-v0
./artifacts/gas_official_full_20260521/scene-play-v0
./artifacts/gas_official_full_20260521/visual-antmaze-giant-navigate-v0
./artifacts/gas_official_full_20260521/visual-antmaze-giant-stitch-v0
./artifacts/gas_official_full_20260521/visual-antmaze-large-explore-v0
./artifacts/gas_official_full_20260521/visual-antmaze-large-navigate-v0
./artifacts/gas_official_full_20260521/visual-antmaze-large-stitch-v0
./artifacts/gas_official_full_20260521/visual-antmaze-medium-explore-v0
./artifacts/gas_official_full_20260521/visual-antmaze-medium-navigate-v0
./artifacts/gas_official_full_20260521/visual-antmaze-medium-stitch-v0
./artifacts/gas_official_full_20260521/visual-scene-play-v0
./artifacts/gas_ogbench_offline_full_20260522_165138
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-giant-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-giant-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-large-explore-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-large-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-large-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-medium-explore-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-medium-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-medium-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-teleport-explore-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-teleport-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/antmaze-teleport-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/humanoidmaze-large-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/humanoidmaze-large-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/humanoidmaze-medium-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/humanoidmaze-medium-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/pointmaze-giant-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/pointmaze-giant-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/pointmaze-large-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/pointmaze-large-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/pointmaze-medium-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/pointmaze-medium-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/pointmaze-teleport-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/pointmaze-teleport-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/scene-play-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/visual-antmaze-giant-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/visual-antmaze-giant-stitch-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/visual-antmaze-large-explore-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/visual-antmaze-large-navigate-v0
./artifacts/gas_ogbench_offline_full_20260522_165138/visual-antmaze-large-stitch-v0
```

## Existing GAS/BARS/TMD-related files

```text
bars/__init__.py
bars/__pycache__/__init__.cpython-310.pyc
bars/__pycache__/__init__.cpython-39.pyc
bars/cli.py
bars/common/__init__.py
bars/common/artifacts.py
bars/common/checkpoint.py
bars/common/config.py
bars/common/device.py
bars/common/logging.py
bars/common/profile.py
bars/common/progress.py
bars/common/seed.py
bars/common/stopper.py
bars/data/__init__.py
bars/data/__pycache__/__init__.cpython-39.pyc
bars/data/__pycache__/d4rl_dataset.cpython-39.pyc
bars/data/__pycache__/normalization.cpython-39.pyc
bars/data/__pycache__/trajectories.cpython-39.pyc
bars/data/d4rl_dataset.py
bars/data/normalization.py
bars/data/ogbench_dataset.py
bars/data/toy_dataset.py
bars/data/trajectories.py
bars/eval/__init__.py
bars/eval/edge_rollout_diag.py
bars/eval/rollout.py
bars/experiments/__init__.py
bars/experiments/pipeline.py
bars/external/__init__.py
bars/external/__pycache__/__init__.cpython-39.pyc
bars/external/__pycache__/gas_artifacts.cpython-313.pyc
bars/external/__pycache__/gas_artifacts.cpython-39.pyc
bars/external/__pycache__/gas_backbone.cpython-39.pyc
bars/external/__pycache__/gas_prepare.cpython-39.pyc
bars/external/gas_artifacts.py
bars/external/gas_backbone.py
bars/external/gas_compat.py
bars/external/gas_prepare.py
bars/gas_bars/__init__.py
bars/gas_bars/bars_v3_planner.py
bars/gas_bars/boundary.py
bars/gas_bars/boundary_feasibility.py
bars/gas_bars/bridge_boundary.py
bars/gas_bars/bridge_dataset.py
bars/gas_bars/bridge_graph.py
bars/gas_bars/bridge_verifier.py
bars/gas_bars/cache.py
bars/gas_bars/diagnostics.py
bars/gas_bars/edge_execution.py
bars/gas_bars/evaluate.py
bars/gas_bars/failure_atlas.py
bars/gas_bars/failure_slice.py
bars/gas_bars/fallback_causal.py
bars/gas_bars/graph_table.py
bars/gas_bars/oracle_bridge.py
bars/gas_bars/planner.py
bars/gas_bars/reachability_dataset.py
bars/gas_bars/reachability_model.py
bars/gas_bars/risk_calibration.py
bars/gas_bars/score_edges.py
bars/gas_bars/selected_edge_diagnostics.py
bars/gas_bars/stage22r_common.py
bars/graph/__init__.py
bars/graph/ann.py
bars/graph/boundary.py
bars/graph/diagnostics.py
bars/graph/edges.py
bars/graph/nodes.py
bars/graph/planner.py
bars/graph/support.py
bars/graph/types.py
bars/models/__init__.py
bars/models/bars_iql.py
bars/models/dataset_embedding.py
bars/models/external_policy.py
bars/models/mlp.py
bars/models/policy.py
bars/models/reachability.py
bars/models/tdr.py
bars/sched/__init__.py
bars/sched/gpu.py
bars/sched/jobctl.py
bars/tmd_test/__init__.py
bars/tmd_test/__pycache__/__init__.cpython-310.pyc
bars/tmd_test/__pycache__/calibration.cpython-310.pyc
bars/tmd_test/__pycache__/construct_graph.cpython-310.pyc
bars/tmd_test/__pycache__/diagnostics.cpython-310.pyc
bars/tmd_test/__pycache__/evaluate_tmd_graph.cpython-310.pyc
bars/tmd_test/__pycache__/io.cpython-310.pyc
bars/tmd_test/__pycache__/keygraph_tmd.cpython-310.pyc
bars/tmd_test/__pycache__/keynodes_tmd.cpython-310.pyc
bars/tmd_test/__pycache__/repr_provider.cpython-310.pyc
bars/tmd_test/__pycache__/tmd_agent_adapter.cpython-310.pyc
bars/tmd_test/calibration.py
bars/tmd_test/construct_graph.py
bars/tmd_test/diagnostics.py
bars/tmd_test/evaluate_tmd_graph.py
bars/tmd_test/io.py
bars/tmd_test/keygraph_tmd.py
bars/tmd_test/keynodes_tmd.py
bars/tmd_test/repr_provider.py
bars/tmd_test/tmd_agent_adapter.py
bars/training/__init__.py
bars/training/bars_iql_train.py
bars/training/goal_sampler.py
bars/training/policy_train.py
bars/training/reach_policy_train.py
bars/training/reach_train.py
bars/training/tdr_train.py
configs/route_b/gas_antmaze-medium-play-v2_seed0_adapter_template.json
configs/routeb/d4rl_antmaze_gas_keygraph_import.json
configs/routeb/d4rl_antmaze_gas_official_backbone.json
configs/routeb/d4rl_antmaze_gas_te_bars.json
configs/routeb/d4rl_antmaze_stage20_bars.json
configs/routeb/d4rl_antmaze_stage20_gas_same_backbone.json
configs/routeb/d4rl_antmaze_stage20_hiql_policy.json
configs/routeb/d4rl_antmaze_stage21_full_bars.json
configs/routeb/ogbench_antmaze_gas_bars.json
configs/routeb/ogbench_antmaze_stage20_gas_same_backbone.json
configs/routeb/ogbench_antmaze_stage21_full_bars.json
configs/stage22/budget_calibration_sweep.json
configs/stage22/confirm_ogbench_antmaze.json
configs/stage22/d4rl_protocol_repair.json
configs/stage22/full_bars_gas_same_backbone.json
configs/stage22/pilot_ogbench_medium.json
configs/stage23_boundary_reentry.json
configs/stage23_d4rl_protocol_repair.json
configs/stage23_key_claim_reachability.json
configs/stage24_oracle_scan.json
configs/stage24_reachability_confirm.json
configs/stage25_local_drift_v2.json
configs/stage25_oracle_scan_matrix.json
configs/stage25_reachability_closing.json
configs/sweeps/d4rl_routeb_gas_official_medium.json
configs/sweeps/d4rl_routeb_gas_te_medium.json
configs/sweeps/d4rl_routeb_switch_gas_keygraph_medium.json
configs/sweeps/d4rl_routeb_switch_gas_te_medium.json
configs/sweeps/d4rl_stage20_budget_sweep.json
configs/sweeps/d4rl_stage20_hiql_official_weights.json
configs/sweeps/d4rl_stage20_protocol_fix_medium100.json
configs/sweeps/d4rl_stage20_routeb_gas_same_backbone.json
configs/sweeps/d4rl_stage20_routeb_hiql_policy.json
configs/sweeps/d4rl_stage21_full_bars.json
configs/sweeps/d4rl_stage2a_reachability_lambda.json
configs/sweeps/ogbench_stage20_gas_same_backbone.json
configs/sweeps/ogbench_stage21_full_bars.json
configs/sweeps/ogbench_stage21_gas_bars.json
configs/tmd_test/tmd_test_antmaze_medium.json
configs/tmd_test/tmd_test_quick.json
external_src/GAS/D_utils/__pycache__/d4rl_env_utils.cpython-39.pyc
external_src/GAS/D_utils/__pycache__/kitchen_utils.cpython-39.pyc
external_src/GAS/D_utils/d4rl_env_utils.py
external_src/GAS/D_utils/kitchen_utils.py
external_src/GAS/K_utils/__pycache__/graph_builder.cpython-39.pyc
external_src/GAS/K_utils/__pycache__/keygraph_utils.cpython-39.pyc
external_src/GAS/K_utils/__pycache__/keynodes_utils.cpython-39.pyc
external_src/GAS/K_utils/graph_builder.py
external_src/GAS/K_utils/keygraph_utils.py
external_src/GAS/K_utils/keynodes_utils.py
external_src/GAS/LICENSE
external_src/GAS/M_utils/__pycache__/encoders.cpython-39.pyc
external_src/GAS/M_utils/__pycache__/flax_utils.cpython-39.pyc
external_src/GAS/M_utils/__pycache__/networks.cpython-39.pyc
external_src/GAS/M_utils/agents/__init__.py
external_src/GAS/M_utils/agents/gas.py
external_src/GAS/M_utils/encoders.py
external_src/GAS/M_utils/flax_utils.py
external_src/GAS/M_utils/networks.py
external_src/GAS/O_utils/__pycache__/datasets.cpython-39.pyc
external_src/GAS/O_utils/__pycache__/env_utils.cpython-39.pyc
external_src/GAS/O_utils/__pycache__/evaluation.cpython-39.pyc
external_src/GAS/O_utils/__pycache__/log_utils.cpython-39.pyc
external_src/GAS/O_utils/datasets.py
external_src/GAS/O_utils/env_utils.py
external_src/GAS/O_utils/evaluation.py
external_src/GAS/O_utils/log_utils.py
external_src/GAS/README.md
external_src/GAS/construct_graph.py
external_src/GAS/evaluate_gas.py
external_src/GAS/pretrain_tdr.py
external_src/GAS/requirements.txt
external_src/GAS/train_policy.py
external_src/GAS_TMD/D_utils/__pycache__/d4rl_env_utils.cpython-310.pyc
external_src/GAS_TMD/D_utils/__pycache__/d4rl_env_utils.cpython-39.pyc
external_src/GAS_TMD/D_utils/__pycache__/kitchen_utils.cpython-310.pyc
external_src/GAS_TMD/D_utils/__pycache__/kitchen_utils.cpython-39.pyc
external_src/GAS_TMD/K_utils/__pycache__/graph_builder.cpython-310.pyc
external_src/GAS_TMD/K_utils/__pycache__/graph_builder.cpython-39.pyc
external_src/GAS_TMD/K_utils/__pycache__/keygraph_tmd_utils.cpython-310.pyc
external_src/GAS_TMD/K_utils/__pycache__/keygraph_tmd_utils.cpython-39.pyc
external_src/GAS_TMD/K_utils/__pycache__/keygraph_utils.cpython-310.pyc
external_src/GAS_TMD/K_utils/__pycache__/keygraph_utils.cpython-39.pyc
external_src/GAS_TMD/K_utils/__pycache__/keynodes_tmd_utils.cpython-310.pyc
external_src/GAS_TMD/K_utils/__pycache__/keynodes_tmd_utils.cpython-39.pyc
external_src/GAS_TMD/K_utils/__pycache__/keynodes_utils.cpython-310.pyc
external_src/GAS_TMD/K_utils/__pycache__/keynodes_utils.cpython-39.pyc
external_src/GAS_TMD/M_utils/__pycache__/encoders.cpython-310.pyc
external_src/GAS_TMD/M_utils/__pycache__/encoders.cpython-39.pyc
external_src/GAS_TMD/M_utils/__pycache__/flax_utils.cpython-310.pyc
external_src/GAS_TMD/M_utils/__pycache__/flax_utils.cpython-39.pyc
external_src/GAS_TMD/M_utils/__pycache__/networks.cpython-310.pyc
external_src/GAS_TMD/M_utils/__pycache__/networks.cpython-39.pyc
external_src/GAS_TMD/O_utils/__pycache__/datasets.cpython-310.pyc
external_src/GAS_TMD/O_utils/__pycache__/datasets.cpython-39.pyc
external_src/GAS_TMD/O_utils/__pycache__/env_utils.cpython-310.pyc
external_src/GAS_TMD/O_utils/__pycache__/env_utils.cpython-39.pyc
external_src/GAS_TMD/O_utils/__pycache__/evaluation.cpython-310.pyc
external_src/GAS_TMD/O_utils/__pycache__/evaluation.cpython-39.pyc
external_src/GAS_TMD/O_utils/__pycache__/log_utils.cpython-310.pyc
external_src/GAS_TMD/O_utils/__pycache__/log_utils.cpython-39.pyc
external_src/GAS_TMD/O_utils/__pycache__/tmd_gas_datasets.cpython-310.pyc
external_src/GAS_TMD/O_utils/__pycache__/tmd_gas_datasets.cpython-39.pyc
external_src/GAS_TMD/R_utils/__pycache__/__init__.cpython-310.pyc
external_src/GAS_TMD/R_utils/__pycache__/__init__.cpython-39.pyc
external_src/GAS_TMD/R_utils/__pycache__/calibration.cpython-310.pyc
external_src/GAS_TMD/R_utils/__pycache__/calibration.cpython-39.pyc
external_src/GAS_TMD/R_utils/__pycache__/checkpoint_utils.cpython-310.pyc
external_src/GAS_TMD/R_utils/__pycache__/checkpoint_utils.cpython-39.pyc
external_src/GAS_TMD/R_utils/__pycache__/dataset_utils.cpython-310.pyc
external_src/GAS_TMD/R_utils/__pycache__/dataset_utils.cpython-39.pyc
external_src/GAS_TMD/R_utils/__pycache__/json_utils.cpython-310.pyc
external_src/GAS_TMD/R_utils/__pycache__/json_utils.cpython-39.pyc
external_src/GAS_TMD/R_utils/__pycache__/path_selection.cpython-310.pyc
external_src/GAS_TMD/R_utils/__pycache__/path_selection.cpython-39.pyc
external_src/GAS_TMD/R_utils/__pycache__/repr_provider.cpython-310.pyc
external_src/GAS_TMD/R_utils/__pycache__/repr_provider.cpython-39.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/__init__.cpython-310.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/__init__.cpython-39.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_agent.cpython-310.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_agent.cpython-39.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_datasets.cpython-310.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_datasets.cpython-39.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_encoders.cpython-310.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_encoders.cpython-39.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_flax_utils.cpython-310.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_flax_utils.cpython-39.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_networks.cpython-310.pyc
external_src/GAS_TMD/TMD_utils/__pycache__/tmd_networks.cpython-39.pyc
external_src/GAS_TMD/__pycache__/construct_graph.cpython-310.pyc
external_src/GAS_TMD/__pycache__/construct_graph.cpython-39.pyc
external_src/GAS_TMD/__pycache__/construct_graph_tmd.cpython-310.pyc
external_src/GAS_TMD/__pycache__/construct_graph_tmd.cpython-39.pyc
external_src/GAS_TMD/__pycache__/evaluate_gas.cpython-310.pyc
external_src/GAS_TMD/__pycache__/evaluate_gas.cpython-39.pyc
external_src/GAS_TMD/__pycache__/evaluate_gas_tmd.cpython-310.pyc
external_src/GAS_TMD/__pycache__/evaluate_gas_tmd.cpython-39.pyc
external_src/GAS_TMD/__pycache__/pretrain_tdr.cpython-310.pyc
external_src/GAS_TMD/__pycache__/pretrain_tdr.cpython-39.pyc
external_src/GAS_TMD/__pycache__/pretrain_tmd.cpython-310.pyc
external_src/GAS_TMD/__pycache__/pretrain_tmd.cpython-39.pyc
external_src/GAS_TMD/__pycache__/train_policy.cpython-310.pyc
external_src/GAS_TMD/__pycache__/train_policy.cpython-39.pyc
external_src/GAS_TMD/__pycache__/train_policy_tmd_low.cpython-310.pyc
external_src/GAS_TMD/__pycache__/train_policy_tmd_low.cpython-39.pyc
external_src/tmd-release/.git/HEAD
external_src/tmd-release/.git/config
external_src/tmd-release/.git/description
external_src/tmd-release/.git/hooks/applypatch-msg.sample
external_src/tmd-release/.git/hooks/commit-msg.sample
external_src/tmd-release/.git/hooks/fsmonitor-watchman.sample
external_src/tmd-release/.git/hooks/post-update.sample
external_src/tmd-release/.git/hooks/pre-applypatch.sample
external_src/tmd-release/.git/hooks/pre-commit.sample
external_src/tmd-release/.git/hooks/pre-merge-commit.sample
external_src/tmd-release/.git/hooks/pre-push.sample
external_src/tmd-release/.git/hooks/pre-rebase.sample
external_src/tmd-release/.git/hooks/pre-receive.sample
external_src/tmd-release/.git/hooks/prepare-commit-msg.sample
external_src/tmd-release/.git/hooks/push-to-checkout.sample
external_src/tmd-release/.git/hooks/update.sample
external_src/tmd-release/.git/index
external_src/tmd-release/.git/info/exclude
external_src/tmd-release/.git/logs/HEAD
external_src/tmd-release/.git/packed-refs
external_src/tmd-release/.gitignore
external_src/tmd-release/CHANGELOG.md
external_src/tmd-release/LICENSE
external_src/tmd-release/README.md
external_src/tmd-release/assets/env_teaser.png
external_src/tmd-release/assets/ogbench.svg
external_src/tmd-release/data_gen_scripts/commands.sh
external_src/tmd-release/data_gen_scripts/generate_antsoccer.py
external_src/tmd-release/data_gen_scripts/generate_locomaze.py
external_src/tmd-release/data_gen_scripts/generate_manipspace.py
external_src/tmd-release/data_gen_scripts/generate_powderworld.py
external_src/tmd-release/data_gen_scripts/main_sac.py
external_src/tmd-release/data_gen_scripts/online_env_utils.py
external_src/tmd-release/data_gen_scripts/viz_utils.py
external_src/tmd-release/impls/agents/__init__.py
external_src/tmd-release/impls/agents/cmd.py
external_src/tmd-release/impls/agents/crl.py
external_src/tmd-release/impls/agents/gcbc.py
external_src/tmd-release/impls/agents/gciql.py
external_src/tmd-release/impls/agents/gcivl.py
external_src/tmd-release/impls/agents/hiql.py
external_src/tmd-release/impls/agents/qrl.py
external_src/tmd-release/impls/agents/sac.py
external_src/tmd-release/impls/agents/tmd.py
external_src/tmd-release/impls/hyperparameters.sh
external_src/tmd-release/impls/main.py
external_src/tmd-release/impls/requirements.txt
external_src/tmd-release/impls/utils/__init__.py
external_src/tmd-release/impls/utils/datasets.py
external_src/tmd-release/impls/utils/encoders.py
external_src/tmd-release/impls/utils/env_utils.py
external_src/tmd-release/impls/utils/evaluation.py
external_src/tmd-release/impls/utils/flax_utils.py
external_src/tmd-release/impls/utils/log_utils.py
external_src/tmd-release/impls/utils/networks.py
external_src/tmd-release/ogbench/__init__.py
external_src/tmd-release/ogbench/__pycache__/__init__.cpython-310.pyc
external_src/tmd-release/ogbench/__pycache__/relabel_utils.cpython-310.pyc
external_src/tmd-release/ogbench/__pycache__/utils.cpython-310.pyc
external_src/tmd-release/ogbench/locomaze/__init__.py
external_src/tmd-release/ogbench/locomaze/ant.py
external_src/tmd-release/ogbench/locomaze/humanoid.py
external_src/tmd-release/ogbench/locomaze/maze.py
external_src/tmd-release/ogbench/locomaze/point.py
external_src/tmd-release/ogbench/manipspace/__init__.py
```

## Existing scripts/configs/reports that can be reused

- bars/external and bars/gas_bars contain existing GAS/BARS adapters and evaluators.
- external_src/GAS is the vendored GAS implementation.
- external_src/tmd-release is the official TMD release cloned for this run.
- /mnt/project/offlinerl_datasets/ogbench is the local OGBench dataset source.

## Files that must not be used as evidence
- main-branch reports/results are stale and excluded.

## Planned tmd-test additions

- bars/tmd_test/*
- configs/tmd_test/*
- scripts/tmd_test_*
- reports/tmd_test_*
- artifacts/tmd_test/*
- runs_tmd_test/*
