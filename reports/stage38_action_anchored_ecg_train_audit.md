# Stage38 Action-Anchored ECG Train Audit

- branch: `codex/cage-mvp`
- starting commit: `d8663ff`
- scope: action-anchored ECG train/eval loop, no humanoid/teleport expansion, no large online benchmark.

## Current State

Stage37 的主要问题不是完全没有 planner signal，而是 planner signal 依赖大量 `knn_bridge_candidate`；该类边不是真实观测转移，不能直接作为可信执行边。Stage37 同时显示从 phi-only contract examples 反向匹配 action 的路线失败，`action_supervision_rate=0`，因此 policy alignment dataset 不能用于 BC。

本轮不再把 phi-only samples 匹配 action 作为主路线。主路线改为从 raw/offline trajectory 的 `observation/action/next_observation` 重建 action-anchored contract samples，并重新形成可训练的合同模型、低层 adapter、action-anchored ECG graph 和 planner score。

## Dataset Availability

已发现可用 raw/offline 数据路径：

- `/root/.ogbench/data/antmaze-giant-navigate-v0.npz`
- `/root/.ogbench/data/antmaze-giant-stitch-v0.npz`
- `artifacts/stage27_gas/datasets/antmaze-giant-navigate-v0/gas_seed44/dataset.npz`
- `artifacts/stage27_gas/datasets/antmaze-giant-stitch-v0/gas_seed44/dataset.npz`

Stage27 派生数据包含 `observations/actions/next_observations/traj_ids/tdr_emb`，可以从 offline trajectory 直接重建 action-supervised positive samples。该路线不依赖 Stage37 的旧 contract examples。

## Stage38 Deliverable Plan

1. 构建 `results/cage_ecg/action_anchored_dataset/action_contracts.{jsonl,npz}`。
2. 训练 `results/cage_ecg/action_anchored_models/contract/model.pt`。
3. 训练 `results/cage_ecg/action_anchored_models/policy_adapter/model.pt`。
4. 构建 `results/cage_ecg/action_anchored_graph_v2/contract_graph.json`，主边只允许真实 offline/action-anchored positive/final positive。
5. 训练/校准 `results/cage_ecg/action_anchored_models/planner_score/weights.json`。
6. 通过显式 ECG variants 接入 evaluator，并在 gate 通过后运行最小 AntMaze smoke/minipilot。

## Blockers

当前没有 raw dataset blocker。如果后续 GAS evaluator 运行失败，应按依赖、checkpoint、runtime model path 或环境注册错误分类，不允许把失败解释成算法结果。
