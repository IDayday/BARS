# CAGE-v0.2 仓库审计

日期：2026-06-06

## 仓库状态

- 仓库根目录：`/mnt/project/BARS`
- 当前分支：`codex/cage-mvp`
- 当前基准提交：`8b4b0c86bd1a224c704f4e81f0d7e2286bd9e24e`
- GAS evaluator：`external_src/GAS/evaluate_gas.py`
- GAS evaluation hook：`external_src/GAS/O_utils/evaluation.py`
- CAGE package：`external_src/GAS/cage/`

## 已有 CAGE 功能

- `--use_cage` 默认关闭；关闭时 GAS 原始路径、policy action 公式、checkpoint loading 不改变。
- CAGE-MVP 已有子目标 commitment、stall/drift monitor、local recovery、final-goal phase、JSONL trace。
- Repair-0 已有 `cage_trace_only`、`cage_safe_full`、churn guard、fallback-to-GAS、recovery lockout。
- CLP1 已有 StateRef、segment contract capture、branchable probe、contract oracle analysis、closed-loop contract dataset builder。

## 已有合同数据和模型功能

- 合同数据存在：
  - `results/cage_clp1/datasets/closed_loop_contracts.jsonl`
  - `results/cage_clp1/datasets_candidate/closed_loop_contracts.jsonl`
- CLP1 oracle summary 存在：
  - `results/cage_clp1/oracle/contract_oracle_summary.md`
- 旧合同模型摘要存在：
  - `results/cage_clp1/models/contract_model_eval_metrics.json`
  - `results/cage_clp1/models/contract_model.json`
- 本轮新增 held-out split 和 v0.2 合同模型输出：
  - `results/cage_v02_contract/splits/`
  - `results/cage_v02_contract/models/contract_model.json`
  - `results/cage_v02_contract/models/eval_metrics.json`

## 当前缺失项

- 合同模型仍是轻量线性 JSON 模型，不是最终 learned reachability 或 risk-aware graph planner。
- 现有合同数据只有小规模 branchable probe，且很多 candidate target 没有有效 exact rollout 特征；不能作为论文级 SOTA 证据。
- `cage_reachability` 和 `cage_risk_path` 仍不支持，未伪造 ablation。
- 远端服务器训练/OGBench 依赖没有在本轮验证。

## 可直接运行的实验

- 静态检查和单测。
- 离线合同数据 split、训练和评估。
- 本地 `gcrlo` 环境下基于已有 artifact 的 AntMaze 小样本 smoke/minipilot。

## 阻塞项

- BLOCKED：minipilot 的 humanoid 部分被 disk quota 阻塞。exact StateRef/contract debug raw JSONL 过大，导致 `OSError: [Errno 122] Disk quota exceeded`。
- BLOCKED：humanoid GAS job 在写 contract trace 时失败，不是算法逻辑失败；后续 humanoid CAGE variant 未运行。
- BLOCKED：当前小样本 minipilot 中 `cage_contract_commit` 在两个 AntMaze 环境均明显回退，因此不允许进入大规模 benchmark。
