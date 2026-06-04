# Stage30 Official GAS Layered Diagnosis

Status: IMPLEMENTED_SMOKE_VALIDATED.

Scope:
- This is an official-GAS-first diagnosis pipeline.
- BARS_BASE, Stage28, Stage29 support graph, bridge-friendly K1, and BARS planner are not used as GAS evidence.
- BARS code is used only as a non-invasive logging, parsing, probing, and analysis scaffold.
- No official GAS graph, planner, policy, subgoal selection, or action outputs are modified.

Implemented:
- Protocol lock rows for instrumentation, keygraph audit, edge probe, and taxonomy.
- Non-invasive official GAS episode tracing with per-task official-style actor RNG lifecycle.
- Diagnostic-only official keygraph edge/node audit.
- Official policy edge execution probe with stratified categories.
- Official-GAS-only failure taxonomy with by-seed/by-env Wilson confidence intervals.
- Parallel layered runner: `scripts/stage30_run_layered_diagnosis.sh`.

Smoke validation:
- Command: `OUT_ROOT=runs_stage30_official_gas/layered_smoke_stage30_diag ENVS=antmaze-medium-navigate-v0 SEEDS=44 TASK_IDS=1 EPISODES=1 EDGES_PER_CATEGORY=2 GPU=cpu EVAL_ON_CPU=1 MAX_JOBS=1 RECOVER_DATASET_INDICES=1 NODE_MAP_TOLERANCE=1e-5 conda run -n gcrlo bash scripts/stage30_run_layered_diagnosis.sh`
- Exact instrumentation success rate: 1.0000 over 1 smoke episode.
- Exact keygraph audit: 12,332 official edges, 927 nodes, 54 path-used edges.
- Exact edge probe: INSUFFICIENT_SAMPLE because official keygraph nodes did not exact-map to dataset states in this smoke.
- Relaxed execution-interface smoke: `NODE_MAP_TOLERANCE=1.0` produced valid set_state/action rollout rows, while `exact_embedding_match=0` and same/cross/dt stayed unavailable.

Key files:
- `scripts/stage30_official_gas_instrument.py`
- `scripts/stage30_official_gas_keygraph_audit.py`
- `scripts/stage30_official_gas_edge_probe.py`
- `scripts/stage30_official_gas_analyze.py`
- `scripts/stage30_run_layered_diagnosis.sh`
- `runs_stage30_official_gas/layered_smoke_stage30_diag/`
- `runs_stage30_official_gas/edge_probe_relaxed_smoke_stage30_diag/`

Full official diagnosis command:

```bash
OUT_ROOT=runs_stage30_official_gas/layered_official_gas_diag \
ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0,antmaze-giant-navigate-v0,antmaze-giant-stitch-v0 \
SEEDS=44,45,46 \
TASK_IDS=1 \
EPISODES=49 \
EDGES_PER_CATEGORY=200 \
GPU=cpu \
EVAL_ON_CPU=1 \
MAX_JOBS=4 \
RECOVER_DATASET_INDICES=1 \
NODE_MAP_TOLERANCE=1e-5 \
conda run -n gcrlo bash scripts/stage30_run_layered_diagnosis.sh
```

Interpretation guard:
- Do not assign a concrete GAS failure mode unless the official episode trace, official path edge trace, and official edge probe evidence support it.
- Do not infer same-trajectory, cross-trajectory, or dt from non-exact node mappings.
- Do not implement algorithm changes until the full official GAS diagnosis identifies a stable dominant failure mode.
