# CAGE-ECG: Execution Contract Graph Framework

## 问题定义

图离线强化学习方法通常先在表示空间构建可搜索图，再把图路径拆成子目标交给低层 goal-conditioned policy。GAS 的关键假设是：TDR graph path 可以作为可拼接的子目标序列。但 Stage32-34 和 GP0/CLP1 的结果说明，表示空间路径存在并不等价于闭环可执行路径。

## 当前实验规律

1. graph path exists 不等于 execution succeeds。
2. coarse q_train support 不足以解释成功或失败。
3. local executable 不等于 task-progress；`cage_contract_rank` 可提高 segment reach，却仍低于 GAS。
4. hard gate 会过度拒绝；Stage32 stitch reject rate 接近 1。
5. soft rank 会过度选择局部安全目标；Stage33 committed source rate 过高。
6. GAS-anchored intervention 更安全，但 Stage34 仍低于 GAS，说明单步干预不足。

## 核心对象

### Executable Funnel Node

一个 funnel node 表示策略在 phi 空间中可稳定进入/离开的局部区域：

- center phi；
- radius；
- support count；
- entry/exit contract score；
- 来源 keygraph/dataset/trace 节点。

### Executable Contract Edge

一个 edge 表示从 funnel A 到 funnel B 的闭环执行合同：

- `contract_lcb`；
- `predicted_hit`；
- `predicted_negative_progress`；
- uncertainty；
- q_train support；
- bottleneck score。

### Boundary Compatibility Contract

两个 edge 单独可执行不代表可拼接。boundary contract 建模 `edge_i -> edge_j` 的兼容性：

- compatibility score；
- boundary risk；
- observed transition count。

### Risk-Constrained Contract Path

路径规划不再只最短 TDR 距离，而是约束：

- 最小 edge contract；
- 累积 negative risk；
- uncertainty；
- boundary risk；
- path length；
- bottleneck edge。

### Graph-Induced Policy Alignment

合同图可以导出策略对齐数据：

- high-contract positive targets；
- low-contract / high-negative hard targets；
- final-goal targets；
- recovery targets；
- boundary transition targets。

若没有 action supervision，不能把 hard goals 当 BC positive，只能用于 ranking、contrastive、conservative filtering 或未来数据采集。

## 算法伪代码

```text
input: frozen GAS artifacts, closed-loop contract dataset, optional contract model

build funnel nodes from observed phi endpoints
for every closed-loop probed pair:
    score contract model on (phi_s, phi_g, target metadata)
    create ContractEdge(src_funnel, dst_funnel)
    set bottleneck_score = min(contract_lcb, 1 - predicted_negative_progress)

for adjacent observed segments:
    estimate BoundaryContract(edge_i, edge_j)

for planning query:
    run candidate planners:
        shortest_by_dphi
        max_contract_path
        risk_constrained_path
        bottleneck_robust_path
        progress_contract_path
    report path_min_contract, negative risk, boundary risk, bottleneck

for policy alignment:
    export high-contract edges as positive contract examples
    export low-contract/high-risk edges as hard negative/ranking examples
    mark action supervision availability
```

## 与相关方法的区别

| method | 主要对象 | ECG 区别 |
|---|---|---|
| GAS | TDR keygraph path | ECG 显式建模闭环执行合同和边界拼接风险 |
| TTGS | trajectory graph search | ECG 不只看可搜索轨迹边，还看 policy-conditioned contract |
| HIQL/HILP | high-level value / latent policy | ECG 是可插拔 contract graph 层，可与更强 backbone 结合 |
| LAVL | value/representation planning | ECG 明确输出 funnel/edge/boundary contract dataset |
| CAGE-v0.4 | GAS anchored intervention | ECG 不再只做执行时 wrapper，而是重建合同图和 policy alignment 数据 |

## 为什么不是 GAS 补丁

CAGE-ECG 改变的是规划对象：从表示图路径升级为执行合同图路径。它仍可使用 GAS 的 keygraph/TDR/policy 作为输入，但输出的是 contract-aware graph、planner audit 和 policy alignment dataset，而不是单纯在线 target selection heuristic。

## 进入 online benchmark 的 gate

1. 合同图成功构建并能识别 low-contract/high-risk edges。
2. 至少一种 contract planner 产生不同于 shortest_by_dphi 的可解释路径。
3. policy alignment dataset 明确 action supervision rate，不能误用 hard unlabeled examples。
4. 离线机制规律解释 Stage32-34 的失败演化。
5. 只允许先做 limited AntMaze smoke；humanoid/teleport 和 SOTA benchmark 仍需后续 gate。

## Stage36: Transition Graph and Action-Supervised Alignment

Stage35 的 planner audit 退化为 direct edge，原因是合同图主要由独立 probe pair 构成，缺少可拼接的 transition edge 和足够密集的 boundary compatibility。即使每条 direct edge 有合同分数，planner 也没有多跳选择空间，无法证明 risk-constrained planner 会不同于 shortest_by_dphi。

Stage36 因此补三类对象：

1. transition-augmented contract graph：从同一 source segment、相邻 path_position、segment capture 的 start/end/final/path phi 和 KNN bridge 中补候选边。
2. final/recovery contract augmentation：final-goal edge 缺失会限制任务终点推进分析；recovery edge 太少会使 recovery 结论 underpowered。
3. action-supervised hard-positive mining：policy alignment 的硬门槛是 action supervision。没有 action 的 hard goals 不能作为 BC 正样本，只能用于 ranking、contrastive、conservative filtering 或未来数据采集。

下一步 ECG policy training 的必要条件：

- transition graph 能产生多跳合同路径；
- final/recovery coverage 至少部分通过；
- positive action-supervised examples 数量大于 0；
- 离线 planner 显示不同于 shortest 的 path，并在 min_contract 或 negative risk 上有改善。

若这些条件不满足，不能进入 online benchmark，也不能宣称 policy alignment training 可行。
