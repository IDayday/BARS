# Stage33 CAGE-v0.3 Contract-Rank 审计

日期：2026-06-06

## 审计结论

Stage32 工程链路已通过。`cage_contract_commit`、合同模型训练/评估、held-out split、manifest/run/aggregation、debug trace 和 smoke/minipilot 报告均已落地，且 Stage32 轻量测试全部返回 0。

但 Stage32 的实验结论不能支持进入大规模 benchmark。合同模型离线有信号，但 test 有效特征样本只有 192，不能作为 SOTA 证据。模型在 test 上表现为：

| label | AUROC | d_phi baseline AUROC | Brier |
|---|---:|---:|---:|
| hit | 1.000 | 0.997 | 0.020 |
| contract_positive | 0.954 | 0.873 | 0.079 |
| negative_progress | 0.719 | 0.339 | 0.222 |

## Stage32 主要失败

`cage_contract_commit` 在 AntMaze 中压低了 CAGE-induced replan，但 success safety 失败：

| env | GAS success | cage_contract_commit success | contract_commit replans | gate reject rate |
|---|---:|---:|---:|---:|
| antmaze-giant-navigate-v0 | 0.64 | 0.36 | 0.00 | 0.517 |
| antmaze-giant-stitch-v0 | 0.80 | 0.00 | 0.00 | 0.998 |

其中 antmaze-stitch gate reject rate 为 0.998，说明 hard gate 过度保守。当前失败不是 no-path、不是 replan storm、不是 trace-only 破坏，而是硬合同门控导致保持/回退/stall。`cage_contract_commit` 消除了重规划风暴，但把过多候选目标拒绝掉，形成“安全局部循环”：segment reach 可以升高，但最终成功率下降。

## 必须推进 CAGE-v0.3

下一步不能启动大规模 benchmark，必须先做 gate calibration 和 `cage_contract_rank`：

1. GAS 原始目标必须进入候选集，作为保守基准。
2. 不再用固定 LCB 阈值拒绝大部分目标。
3. 仅对 extreme negative-progress risk 做 hard reject。
4. 对候选目标做合同排序、coverage 控制和 GAS replacement discipline。
5. 如果 non-GAS 目标没有明显优于 GAS，优先执行 GAS 原始目标。
6. recovery 默认仍关闭，除非 recovery candidate 合同明确通过且显式启用。
7. 实验必须先从 no-debug recheck、AntMaze smoke、AntMaze minipilot 开始。

## 阻塞项

- Humanoid Stage32 失败来自 exact StateRef/debug trace 写入触发 `OSError: [Errno 122] Disk quota exceeded`；Stage33 必须使用 light trace 或 no-debug 模式。
- 合同模型 test 有效样本只有 192；合同模型可用于诊断和 smoke，但不能用于 SOTA 声明。
- AntMaze success safety 未恢复前，不应运行 humanoid/teleport 大规模实验。
