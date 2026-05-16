# Stage21: Full-BARS v2 and GAS+BARS

This patch adds two separate directions.

## 1. GAS+BARS same-backbone

GAS+BARS is a controlled attribution experiment.  It keeps the official GAS
keygraph, dataset embeddings, and low-level policy, then replaces only the path
selection/scoring layer with BARS reachability and boundary-aware planning.

Run example:

```bash
cd /root/remote/project/BARS
export GAS_REPO_PATH=/root/remote/project/BARS/external_src/GAS
export GAS_OGBENCH_ARTIFACT_ROOT=/root/remote/project/BARS/external_artifacts/gas_ogbench
export GAS_OGBENCH_POLICY_CKPT_ROOT=/root/remote/project/BARS/external_artifacts/gas_ogbench_policy_ckpts

DRY_RUN=1 MODE=gas_bars GPUS=0,1,2,3,4,5,6 \
  LOG_ROOT=runs_stage21_gas_bars \
  ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0,antmaze-large-navigate-v0,antmaze-large-stitch-v0 \
  SEEDS=0,1,2 EPISODES=100 \
  bash scripts/run_stage21_full_bars.sh
```

Required artifact layout:

```text
$GAS_OGBENCH_ARTIFACT_ROOT/$ENV/seed$SEED/keygraph.pkl
$GAS_OGBENCH_ARTIFACT_ROOT/$ENV/seed$SEED/dataset_embeddings.npy
$GAS_OGBENCH_ARTIFACT_ROOT/$ENV/seed$SEED/node_indices.npy      # optional
$GAS_OGBENCH_POLICY_CKPT_ROOT/$ENV/seed$SEED/policy.pkl
```

## 2. Full-BARS v2

Full-BARS v2 is an independent algorithm.  It does not depend on HIQL or GAS
policies.  It trains:

1. BARS TDR representation.
2. BARS-IQL low-level policy using a graph-aware goal distribution.
3. Policy-conditioned reachability scorer.
4. BARS bottleneck-temporal graph.
5. Boundary compatibility and BARS path planner.

Run example:

```bash
cd /root/remote/project/BARS
DRY_RUN=1 MODE=d4rl_full_bars GPUS=0,1,2,3,4,5,6 \
  LOG_ROOT=runs_stage21_full_bars_d4rl \
  ENVS=antmaze-medium-play-v2,antmaze-medium-diverse-v2,antmaze-large-play-v2,antmaze-large-diverse-v2 \
  SEEDS=0,1,2 EPISODES=100 \
  bash scripts/run_stage21_full_bars.sh
```

For OGBench state-based antmaze tasks:

```bash
DRY_RUN=1 MODE=ogbench_full_bars GPUS=0,1,2,3,4,5,6 \
  LOG_ROOT=runs_stage21_full_bars_ogbench \
  SEEDS=0,1,2 EPISODES=100 \
  bash scripts/run_stage21_full_bars.sh
```

## Important variants

- `full_bars`: current recommended BARS planner.
- `reachability`: no boundary term, useful strong ablation.
- `constrained_bars`: budgeted planner; currently diagnostic unless tuned.
- `gas`: GAS shortest-path baseline on imported GAS graph.
- `gas_bars`: boundary-aware BARS planner on imported GAS graph.

## Hard-edge mining

If edge rollout diagnostics are enabled, failed high-score edges can be mined and
fed into the next BARS-IQL / reachability run:

```bash
python scripts/mine_stage21_hard_edges.py --run-dir <RUN_DIR> --p-exec-min 0.5
```

This writes `<RUN_DIR>/cache/hard_edges.npz`.  Set
`policy.hard_edge_cache` or warmstart the cache into a subsequent run.
