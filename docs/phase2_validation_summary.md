# Phase 2 验证工作整理

本文整理当前仓库中已经完成并留有产物的 Phase 2 相关验证工作。Phase 2 的范围是
support-only compressed option graph baseline：只基于离线轨迹支持构造压缩有向 option
graph，并做路径覆盖、兼容性、瓶颈移除和基线对比诊断。

明确不属于 Phase 2 已验证范围的内容：

- 不训练策略。
- 不实现或验证 TDR/TMD/MQE latent model。
- 不做环境 closed-loop rollout 或在线执行成功率验证。
- 不把 kNN/random baseline 当作可执行边证据，只作为图连通性对照。

## 代码与配置落点

- Phase 2 包：`phase2/`
- 主入口：`scripts/run_phase2_support_graph.py`
- 合成单元测试：`tests/test_phase2_synthetic.py`
- 已配置数据集：
  - `configs/phase2_antmaze_stitch.yaml`
  - `configs/phase2_scene.yaml`
  - `configs/phase2_antmaze_navigate.yaml`

当前结果目录中已有完整产物的是 `antmaze-large-stitch-v0` 和 `scene-play-v0`；
`antmaze-large-navigate-v0` 当前只看到配置，未看到对应 `results/phase2/antmaze_large_navigate`
完成产物。

## 验证类型

### 1. 合成单元测试

当前已验证：

- 只有真实离线 trajectory segment 会变成 option edge。
- self-loop 默认不会作为高层 option edge。
- density、bottleneck、core_plus_bottleneck 三类节点选择预算逻辑正确。
- virtual query 可以通过 support_N 临时连接 start/goal 到 selected graph。
- option 之间的 termination/initiation mismatch 可被 strict compatibility 捕获。
- 移除高瓶颈节点会降低桥接图路径覆盖。

复验命令：

```bash
pytest -q tests/test_phase2_synthetic.py
```

本轮复验结果：`6 passed in 1.67s`。

### 2. 冒烟验证

产物目录：`results/phase2_smoke/antmaze_large_stitch/core_plus_bottleneck_budget10_H5/`

目的：验证 Phase 2 pipeline 在小样本上能完成数据加载、聚类、节点选择、option edge 构造、
图构建、路径评估、兼容性评估、瓶颈移除和图表输出。

关键参数与数据规模：

- 数据集：`antmaze-large-stitch-v0`
- transitions：1,000
- H：5
- node budget：10
- node selection：`core_plus_bottleneck`

关键结果：

| 指标 | 值 |
| --- | ---: |
| selected nodes | 10 |
| option edges | 12 |
| edge segments | 470 |
| graph reachable pair ratio sampled | 0.2000 |
| strict path coverage | 0.0000 |
| virtual path coverage | 0.0000 |
| cluster compatible rate | 1.0000 |
| strict compatible rate | 0.5000 |

解读：这是 pipeline/产物冒烟，不是性能结论。小样本和极小 node budget 下没有覆盖验证路径是预期风险。

### 3. `antmaze-large-stitch-v0` 正式离线诊断

产物目录：`results/phase2/antmaze_large_stitch/core_plus_bottleneck_budget80_H10/`

关键参数与数据规模：

- transitions：200,000
- observation dim：29
- action dim：8
- terminal flags：1,000
- cluster method：`grid_xy`
- n_clusters：400
- H：10
- min_support：3
- min_episodes：2
- node budget：80
- node selection：`core_plus_bottleneck`

核心结果：

| 指标 | 值 |
| --- | ---: |
| selected nodes | 80 |
| option edges | 312 |
| edge segments | 348,947 |
| mean option median_h | 8.0689 |
| mean segments per edge | 1,118.4199 |
| graph reachable pair ratio sampled | 0.8694 |
| strict path coverage | 116 / 144 = 0.8056 |
| virtual path coverage | 296 / 500 = 0.5920 |
| cluster compatible rate | 1.0000 |
| strict compatible rate | 0.6845 |

瓶颈移除验证：

| 条件 | path coverage | mean path cost |
| --- | ---: | ---: |
| before | 0.8056 | 58.9913 |
| after top 10% bottleneck removal | 0.6181 | 65.6790 |

解读：离线支持图在 selected-node strict 查询上有较高路径覆盖；移除瓶颈节点后覆盖明显下降，说明
bottleneck score 对图连通性有实际解释力。strict compatibility 约 68.45%，表明相邻 option 的严格时序拼接
不是全覆盖，后续如果要进入可执行性阶段仍需 rollout 或 policy-level 验证。

对照基线：

| baseline | edges | unsupported edge rate | strict path coverage |
| --- | ---: | ---: | ---: |
| kNN same nodes | 312 | 0.1378 | 0.7708 |
| random same edge budget | 312 | 0.9647 | 0.9236 |

解读：random baseline 覆盖高但 unsupported edge rate 极高，不能作为可执行路径证据；kNN baseline
也有非零 unsupported edge rate，Phase 2 主结论应优先使用 data-supported option edges。

### 4. `scene-play-v0` 正式离线诊断

产物目录：`results/phase2/scene_play/core_plus_bottleneck_budget128_H10/`

关键参数与数据规模：

- transitions：200,000
- observation dim：40
- action dim：5
- terminal flags：200
- cluster method：`kmeans`
- n_clusters：512
- H：10
- min_support：3
- min_episodes：2
- node budget：128
- node selection：`core_plus_bottleneck`

核心结果：

| 指标 | 值 |
| --- | ---: |
| selected nodes | 128 |
| option edges | 1,089 |
| edge segments | 181,079 |
| mean option median_h | 6.9780 |
| mean segments per edge | 166.2801 |
| graph reachable pair ratio sampled | 0.9386 |
| strict path coverage | 17 / 17 = 1.0000 |
| virtual path coverage | 99 / 100 = 0.9900 |
| cluster compatible rate | 1.0000 |
| strict compatible rate | 0.3348 |

瓶颈移除验证：

| 条件 | path coverage | mean path cost |
| --- | ---: | ---: |
| before | 1.0000 | 16.8452 |
| after top 10% bottleneck removal | 0.8824 | 21.6384 |

解读：scene-play 的离线路径覆盖很高，瓶颈移除后覆盖和路径成本都恶化，支持瓶颈诊断有效。
但 strict compatibility 只有约 33.48%，说明压缩 option 图的 cluster-level 连通性强于严格可拼接性。

对照基线：

| baseline | edges | unsupported edge rate | strict path coverage |
| --- | ---: | ---: | ---: |
| kNN same nodes | 527 | 0.4763 | 0.5294 |
| random same edge budget | 1,089 | 0.9100 | 1.0000 |

解读：random baseline 仍然主要是无支持边带来的图连通性假象；kNN baseline 在 scene-play 上的
unsupported edge rate 更高，不应替代 Phase 2 的离线支持边。

## 已生成产物清单

每个完整 Phase 2 run 都生成以下主要产物：

- `dataset_summary.json`：数据维度、transition 数、terminal 数。
- `cluster_density.csv`：cluster 占用与密度。
- `bottleneck_scores_H*.csv`：crossing、betweenness、removal impact 合成瓶颈分数。
- `path_queries.csv`：路径覆盖查询。
- `selected_nodes.csv`：按 selection method 选中的压缩图节点。
- `option_edges.csv`：离线支持的有向 option edges。
- `edge_segments.npz`：每条 option edge 对应的支持轨迹片段。
- `self_loop_summary.csv`：selected cluster 内部 self-loop 支持摘要。
- `graph_summary.csv`：图连通性、SCC/WCC、采样可达率。
- `path_coverage.csv`、`strict_paths.csv`、`virtual_paths.csv`：strict/virtual 查询覆盖结果。
- `compatibility_summary.csv`、`path_compatibility.csv`：相邻 option 严格可拼接性诊断。
- `bottleneck_utility.csv`：移除 top bottleneck 后的覆盖变化。
- `baseline_summary.csv`：kNN/random 对照图的覆盖和 unsupported edge rate。
- `metrics_summary.json`、`aggregate_summary.csv`：关键指标聚合。
- `plots/`：selected node、edge horizon、path coverage、bottleneck removal 图。

## 当前结论

Phase 2 已完成的是离线支持图层面的验证：代码路径有合成测试覆盖，pipeline 有 smoke 产物，
`antmaze-large-stitch-v0` 和 `scene-play-v0` 有 20 万 transition 规模的正式离线诊断结果。

可以支撑的结论：

- Phase 2 能从离线轨迹构造 data-supported compressed option graph。
- `core_plus_bottleneck` 节点选择在两个已完成数据集上都能形成非平凡可达图。
- bottleneck removal 会降低路径覆盖，说明瓶颈分数对压缩图连通性有解释力。
- compatibility 诊断能揭示 cluster-level 连通与严格轨迹拼接之间的差距。
- kNN/random baseline 可作为连通性对照，但由于 unsupported edge rate 高，不能作为可执行路径证据。

尚不能支撑的结论：

- 不能声明 option edges 已被策略执行验证。
- 不能声明 Phase 2 图能直接提升 online success。
- 不能声明 TDR/TMD/MQE 或 learned policy 相关贡献已完成。
- 不能声明 `antmaze-large-navigate-v0` 已完成 Phase 2 结果验证。
