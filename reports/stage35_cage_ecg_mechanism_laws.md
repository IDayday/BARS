# Stage35 CAGE-ECG Mechanism Laws

本报告只给出机制规律和离线证据，不声明统计显著性。样本数过少处标记 INCONCLUSIVE。

| law | status | value | n | evidence |
|---|---|---:|---:|---|
| segment_target_reach_rate vs success | OBSERVED | -0.8735 | 18 | Pearson correlation over compact deployment rows; n=18. |
| mean_segment_progress vs success | OBSERVED | 0.9341 | 18 | Pearson correlation over compact deployment rows; n=18. |
| final_goal_on_rate vs success | OBSERVED | 0.9804 | 18 | Pearson correlation over compact deployment rows; n=18. |
| stall_count vs success | OBSERVED | -0.8286 | 18 | Pearson correlation over compact deployment rows; n=18. |
| intervention_rate vs success | OBSERVED | 0.1277 | 10 | Pearson correlation over compact deployment rows; n=10. |
| source_gas_rate vs success | INCONCLUSIVE | -0.0043 | 6 | Pearson correlation over compact deployment rows; n=6. |
| source_cage_rate vs success | INCONCLUSIVE | 0.6601 | 6 | Pearson correlation over compact deployment rows; n=6. |
| source_committed_rate vs success | INCONCLUSIVE | -0.3266 | 5 | Pearson correlation over compact deployment rows; n=5. |
| committed_usage_rate vs success | INCONCLUSIVE | -0.5082 | 3 | Pearson correlation over compact deployment rows; n=3. |
| local_safe_loop proxy | OBSERVED | 3 | 22 | Proxy: segment reach >=0.30, stall >=15, success <=0.50, final_goal_on_rate <=0.70. |
| local executability is insufficient | OBSERVED | NA | 22 | Stage33 rank had high segment reach relative to trace-only but lower success; Stage34 intervention reduced committed use yet still failed success safety. |
| intervention is insufficient | OBSERVED | NA | 12 | Stage34 contract_intervene improved over contract_rank but remained below GAS on both AntMaze envs. |
| need contract graph and policy alignment | OBSERVED | NA | 12 | GP0/CLP1 show q_train support alone is insufficient; closed-loop contractibility and final-goal/farther path target quality are bottlenecks. |
| contract model evidence strength | INCONCLUSIVE | 192 | NA | Held-out contract model was useful for smoke but test feature count is small; no SOTA claim. |
| split coverage | ok | 9216 | NA | Contract split summary is included to preserve dataset audit context. |

## 结论

Stage32-34 的失败演化说明：只做在线 gate/rank/intervention 不足以解决图路径、执行合同和任务推进之间的不一致。下一步需要显式构建执行 funnel node、合同 edge、边界兼容 contract，并用这些对象驱动离线合同路径规划和图诱导策略对齐数据集。
