# Phase 2.1 验证工作整理

本文整理当前仓库中已完成并留有产物的 Phase 2.1 验证工作。Phase 2/2.1 的范围仍然是
support-only compressed option graph baseline：只基于离线轨迹和 Phase 1 cluster/support/bottleneck
诊断构造 data-supported directed option graph，并评估图层覆盖、节点选择、option edge 支持量、
compatibility、bottleneck removal 和非支持图 baseline。

明确不属于 Phase 2.1 已验证范围的内容：

- 不训练策略。
- 不实现或验证 TDR/TMD/MQE latent model。
- 不做环境 rollout 或 closed-loop online success 验证。
- 不把 kNN/random baseline 当作可执行边证据。

## 代码与配置

- Phase 2 包：`phase2/`
- 主入口：`scripts/run_phase2_support_graph.py`
- 合成测试：`tests/test_phase2_synthetic.py`
- Sweep 配置：
  - `configs/phase2_sweep_antmaze_stitch.yaml`
  - `configs/phase2_sweep_scene.yaml`
- 单点配置：
  - `configs/phase2_antmaze_stitch.yaml`
  - `configs/phase2_scene.yaml`
  - `configs/phase2_antmaze_navigate.yaml`

本轮新增/加固的关键指标包括：

- strict coverage 的 `all_num_queries`、`strict_query_selection_rate`、`strict_coverage_over_all`。
- virtual query 的 `num_virtual_edges_used`、`num_real_option_edges_used`、`virtual_edge_ratio`。
- option edge 的 unique starts/terminations/start-goal pairs/episodes 和 raw/certified reverse support。
- cost 的 `support_unit=segments|unique_starts|episodes`。
- compatibility 的 bridge support 数量、bridge episodes、bridge horizon 和 support rate。
- aggregate 级 coverage、compatibility、edge scale、bottleneck removal delta 和 sweep plots。

## 复验命令

已运行：

```bash
pytest -q tests/test_phase2_synthetic.py
python scripts/run_phase2_support_graph.py --config configs/phase2_sweep_antmaze_stitch.yaml --sweep
python scripts/run_phase2_support_graph.py --config configs/phase2_sweep_scene.yaml --sweep
python scripts/run_phase2_support_graph.py --config configs/phase2_antmaze_navigate.yaml
```

测试结果：`13 passed`。

正式 sweep 产物：

- `results/phase2/antmaze_large_stitch/aggregate_summary.csv`：36 rows，H = 5/10/25，budgets = 40/80/120，methods = density/bottleneck/core_plus_bottleneck/all。
- `results/phase2/scene_play/aggregate_summary.csv`：36 rows，H = 5/10/25，budgets = 64/128/192，methods = density/bottleneck/core_plus_bottleneck/all。
- dataset-level plots：
  - `coverage_vs_budget_by_method.png`
  - `virtual_coverage_vs_budget_by_method.png`
  - `compatibility_vs_budget_by_method.png`
  - `num_edges_vs_budget_by_method.png`
  - `bottleneck_removal_delta_by_method.png`

可选单点也已完成：

- `results/phase2/antmaze_large_navigate/aggregate_summary.csv`

## Sweep 结果

### antmaze-large-stitch-v0

压缩方法均值，不含 all upper bound：

| method | strict coverage over all | virtual coverage | strict compatible rate | mean option edges | bottleneck removal delta coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| density | 0.247 | 0.464 | 0.700 | 364 | 0.094 |
| bottleneck | 0.216 | 0.528 | 0.668 | 464 | 0.269 |
| core_plus_bottleneck | 0.234 | 0.478 | 0.674 | 414 | 0.161 |

`all` upper bound 的均值为 strict coverage over all 0.992、virtual coverage 0.998、strict compatible rate
0.633、mean option edges 1,922。它证明离线支持图本身覆盖很强，但不是压缩预算下的公平节点选择。

结论：

- 相同预算下，`density` 通常给出最高 strict coverage over all 和最高 strict compatibility。
- `bottleneck` 给出最高 virtual coverage，并且 bottleneck removal delta 最大，说明它更偏向保留桥接/连接性节点。
- `core_plus_bottleneck` 居中：coverage 不总是最高，但比纯 density 更保留瓶颈移除敏感性。

### scene-play-v0

压缩方法均值，不含 all upper bound：

| method | strict coverage over all | virtual coverage | strict compatible rate | mean option edges | bottleneck removal delta coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| density | 0.072 | 0.886 | 0.361 | 1,465 | 0.039 |
| bottleneck | 0.070 | 0.966 | 0.350 | 2,095 | 0.041 |
| core_plus_bottleneck | 0.083 | 0.954 | 0.333 | 1,934 | 0.179 |

`all` upper bound 的均值为 strict coverage over all 0.950、virtual coverage 1.000、strict compatible rate
0.299、mean option edges 14,298。

结论：

- `bottleneck` 与 `core_plus_bottleneck` 在 virtual coverage 上明显优于 density。
- `core_plus_bottleneck` 的 strict coverage over all 和 bottleneck removal delta 更高，说明混合选择更能保留关键桥接结构。
- `density` 的 strict compatibility 最高，但 coverage 较低，体现了覆盖和严格可拼接性之间的 trade-off。

### antmaze-large-navigate-v0 单点

`core_plus_bottleneck`，H=10，budget=80：

| metric | value |
| --- | ---: |
| option edges | 444 |
| edge segments | 462,062 |
| strict query selection rate | 0.140 |
| strict coverage over all | 0.080 |
| virtual path coverage | 0.330 |
| strict compatible rate | 0.611 |
| bottleneck removal delta coverage | 0.143 |

这只是额外单点诊断，不替代 stitch/scene 的 sweep 结论。

## Coverage 与 Compatibility Trade-off

sweep 显示 coverage 提升通常伴随 compatibility 下降。`all` upper bound 拥有最多边和最高覆盖，但
strict compatible rate 低于多数压缩 density 图：

- antmaze stitch：`all` strict compatible rate 均值 0.633，density 为 0.700。
- scene play：`all` strict compatible rate 均值 0.299，density 为 0.361。

virtual coverage 也需要结合 virtual edge usage 解读。scene-play 中 virtual query 覆盖很高，但 mean
virtual edge ratio 在 compressed methods 中约 0.695 到 0.796，`all` 约 0.896。这说明很多 query 依赖
临时 start/goal support connectors，而不是完全由 real option edges 串起。

## Baseline 解释

kNN/random baseline 的边不要求离线 support，因此只能作为 connectivity illusion 对照。Phase 2.1 中
baseline cost 已改成 `hop_count_cost`，不再和 support graph 的 horizon/support-aware path cost 直接比较。

平均 unsupported edge rate：

| dataset | baseline | unsupported edge rate | strict path coverage |
| --- | --- | ---: | ---: |
| antmaze stitch | kNN_same_nodes | 0.239 | 0.545 |
| antmaze stitch | kNN_geometry_same_nodes | 0.217 | 0.557 |
| antmaze stitch | random_same_edge_budget | 0.942 | 0.892 |
| scene play | kNN_same_nodes | 0.534 | 0.717 |
| scene play | random_same_edge_budget | 0.880 | 0.889 |

random baseline 覆盖高但 unsupported edge rate 极高，不能作为可执行路径证据。kNN baseline 在 scene-play
上也有超过一半边没有 support，因此不能替代 data-supported option edges。

## 当前结论

Phase 2.1 可以支撑的结论：

- 已完成从单点 `core_plus_bottleneck` 到多 H、多 node budget、多 node selection 的 sweep。
- data-supported option graph 能稳定生成，并输出 edge support scale、unique support、cost 和 compatibility。
- `all` upper bound 验证了离线 support graph 的覆盖上限，压缩方法之间存在 coverage/compatibility/edge-scale trade-off。
- bottleneck removal 会降低覆盖或提高路径成本，尤其在 antmaze bottleneck/core 和 scene core 上更明显。
- kNN/random 的高覆盖常来自 unsupported edges，是图连通性假象，不是 option executability 证据。

仍不能声明的内容：

- 不能声明 option edges 已被策略执行验证。
- 不能声明 Phase 2 图能直接提升 online success。
- 不能声明 TDR/TMD/MQE 或 learned policy 贡献已完成。
- policy executability 必须留到 Phase 3 通过策略和环境验证。
