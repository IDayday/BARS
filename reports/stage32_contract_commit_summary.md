# Stage32 CAGE-v0.2 Contract Commit 总结

## 本轮目标

实现并验证 `cage_contract_commit`：合同校准的承诺执行算法。目标是让子目标切换、final-goal phase、recovery 和 replan 都受闭环执行合同约束，并默认安全回退，不破坏官方 GAS 默认行为。

## 代码改动摘要

- 新增 `cage_contract_commit` variant。
- 新增合同模型在线 scorer 和合同 gate。
- 合同 gate 失败时优先保持当前 committed target，否则 fallback 到 GAS 原始子目标。
- v0.2 默认启用 churn guard，默认关闭不确定 recovery。
- 增加 held-out contract dataset split 脚本。
- 扩展合同模型训练/评估脚本，输出 AUROC、Brier、calibration、selective curve 和 d_phi baseline。
- 增加 CAGE debug trace 合同字段和 StateRef 摘要字段。
- 更新 command builder、manifest、aggregation 和 contract capture smoke 脚本。

## 新增/修改文件

- `external_src/GAS/cage/config.py`
- `external_src/GAS/cage/state_machine.py`
- `external_src/GAS/cage/contract_model.py`
- `external_src/GAS/cage/closed_loop_contracts.py`
- `external_src/GAS/evaluate_gas.py`
- `external_src/GAS/O_utils/evaluation.py`
- `scripts/build_cage_eval_command.py`
- `scripts/cage_experiment_manifest.py`
- `scripts/aggregate_cage_experiments.py`
- `scripts/run_contract_capture_smoke.py`
- `scripts/build_contract_dataset_splits.py`
- `train_cage_contract.py`
- `evaluate_cage_contract.py`
- `tests/test_cage_contract_gate.py`
- `tests/test_contract_dataset_splits.py`
- `tests/test_cage_contract_commit.py`
- `docs/cage_v02_repo_audit.md`
- `docs/cage_v02_contract_commit_design.md`
- `docs/cage_sota_plan.md`
- `reports/stage32_contract_commit_smoke.md`
- `reports/stage32_contract_commit_minipilot.md`

## 验证命令

```bash
python -m py_compile external_src/GAS/evaluate_gas.py external_src/GAS/O_utils/evaluation.py external_src/GAS/cage/*.py scripts/build_contract_dataset_splits.py train_cage_contract.py evaluate_cage_contract.py scripts/build_cage_eval_command.py scripts/cage_experiment_manifest.py scripts/aggregate_cage_experiments.py scripts/run_contract_capture_smoke.py
pytest tests/test_cage_experiment_manifest.py tests/test_cage_aggregation.py -q
pytest tests/test_cage_state_machine.py tests/test_cage_subgoal_selector.py -q
pytest tests/test_cage_trace_only.py tests/test_cage_churn_guard.py tests/test_cage_contract_gate.py tests/test_contract_dataset_splits.py tests/test_cage_contract_commit.py -q
```

结果：全部返回 0；共 29 个 pytest 用例通过。

## 合同数据和模型

```bash
python scripts/build_contract_dataset_splits.py \
  --input_jsonl results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --out_dir results/cage_v02_contract/splits \
  --seed 0 \
  --min_examples 100

python train_cage_contract.py \
  --train_path results/cage_v02_contract/splits/train.jsonl \
  --val_path results/cage_v02_contract/splits/val.jsonl \
  --test_path results/cage_v02_contract/splits/test.jsonl \
  --out_model results/cage_v02_contract/models/contract_model.json \
  --out_metrics results/cage_v02_contract/models/eval_metrics.json \
  --out_report results/cage_v02_contract/models/eval_report.md \
  --epochs 300 \
  --lr 0.05
```

Split status：`ok`，总记录 9216。合同模型 test 有效特征样本 192；结果可用于 smoke，但不能用于 SOTA 结论。

关键 test 指标：

- hit AUROC：1.000；d_phi baseline：0.997
- contract-positive AUROC：0.954；d_phi baseline：0.873
- negative-progress AUROC：0.719；d_phi baseline：0.339

## Smoke 结果

见：`reports/stage32_contract_commit_smoke.md`

- 5 个 job 全部返回 0。
- GAS 命令未包含 `--use_cage`。
- `cage_contract_commit` 命令包含 `--use_cage --cage_contract_commit`。
- `cage_contract_commit` 在 smoke 中将 global replan 从 5 降到 0，但成功率样本全为 0，不可做性能结论。

## Minipilot 结果

见：`reports/stage32_contract_commit_minipilot.md`

- AntMaze navigate/stitch 的 10 个 job 成功。
- Humanoid GAS 在写 exact StateRef contract trace 时因 disk quota 失败，后续 humanoid CAGE variant 未运行。
- 已清理本轮 minipilot 的大型 raw segment/debug traces；保留 compact summary、`eval.csv` 和 stdout/stderr。
- AntMaze 中 trace-only 与 GAS 成功率一致。
- `cage_contract_commit` 成功率明显回退：
  - antmaze-nav：GAS 0.64，contract_commit 0.36
  - antmaze-stitch：GAS 0.80，contract_commit 0.00
- `cage_contract_commit` replan 降到 0，但 gate 过保守：
  - antmaze-nav reject rate 0.517
  - antmaze-stitch reject rate 0.998

## Gate 状态

| Gate | 状态 | 依据 |
|---|---|---|
| trace-only parity | PASS on AntMaze, BLOCKED on Humanoid | AntMaze 两 env 成功率与 GAS 一致；humanoid 因 quota 未完成 |
| churn reduction | PASS on AntMaze | contract_commit global replan 为 0，低于 safe_full 约 5.6 |
| success safety | FAIL | AntMaze contract_commit 明显低于 GAS |
| failure-dense improvement | BLOCKED | humanoid 未完成；AntMaze teleport 未运行 |
| contract validity | INCONCLUSIVE | held-out negative-progress 优于 d_phi baseline，但部署 gate 过保守导致成功率回退 |

## 失败样本分析

当前失败不是 graph no-path，也不是 replan storm。`cage_contract_commit` 主要失败模式是合同 gate 过度拒绝候选目标，导致过多保持/回退和 stall，尤其在 antmaze-stitch 中 gate reject rate 接近 1。模型可以预测负进展，但在线阈值和 target-mode/domain 泛化还不足。

## 是否进入更大规模 benchmark

不建议。当前 minipilot 已经在标准 AntMaze 上出现明显成功率回退。下一阶段应先修合同数据质量、threshold preregistration 和 online gate calibration，再重新跑 staged minipilot。

## 下一步命令

先跑一个无 exact StateRef 大 trace 的轻量重测，验证合同 gate 是否仍然过保守：

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/cage_experiment_manifest.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_v02_contract_commit/recheck_antmaze_no_debug \
  --envs antmaze-giant-navigate-v0 antmaze-giant-stitch-v0 \
  --seeds 42 \
  --variants gas cage_trace_only cage_safe_full cage_contract_commit \
  --episodes_per_goal 5 \
  --goals_per_env 5 \
  --manifest_path results/cage_v02_contract_commit/recheck_antmaze_no_debug/manifests/recheck_manifest.jsonl \
  --strict_paths \
  --cage_contract_model_path /mnt/project/BARS/results/cage_v02_contract/models/contract_model.json

/root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py \
  --manifest_path results/cage_v02_contract_commit/recheck_antmaze_no_debug/manifests/recheck_manifest.jsonl \
  --max_jobs 8
```
