# Stage20 SOTA Alignment Patch

This patch is built against the uploaded BARS.zip snapshot.

## Main code changes

1. `bars/graph/planner.py`
   - Adds `constrained_bars` / `budget_bars` planner variants.
   - Implements label-setting line-graph search for the BARS objective:
     minimize temporal/progress cost subject to an execution budget on reachability risk and boundary mismatch.

2. `bars/eval/rollout.py`
   - Fixes `direct_goal_after_k`: no-path episodes no longer terminate before the failure streak can reach k.
   - Adds `direct_goal_after_progress`, `direct_goal_after_k_or_progress`, and `direct_goal_after_k_and_progress`.
   - Logs graph progress, fallback attribution, final-goal distance improvement, and execution-budget fields.

3. `bars/graph/edges.py`
   - Preserves same-trajectory temporal support edges during top-k pruning by default.
   - Adds optional p_exec filtering and graph connectivity diagnostics.

4. Route-B same-backbone support
   - `bars/experiments/pipeline.py` can load official external dataset embeddings instead of silently saving a random zero-step local TDR.
   - `bars/external/gas_compat.py` aligns external raw dataset embeddings to BARS compact trajectory indices and supports raw-to-compact node index conversion.
   - `bars/models/external_policy.py` expands environment variables in external paths and exposes `embed()`.
   - `routeb_gas_policy_adapter.py` exposes GAS `get_phi` for online graph lookup.

## New configs

- `configs/routeb/d4rl_antmaze_stage20_bars.json`
- `configs/routeb/d4rl_antmaze_stage20_gas_same_backbone.json`
- `configs/routeb/d4rl_antmaze_stage20_hiql_policy.json`

## New scripts

- `scripts/make_stage20_sota_sweeps.py`
- `scripts/run_stage20_sota_alignment.sh`

## Minimal commands

```bash
python -m py_compile \
  bars/graph/planner.py \
  bars/eval/rollout.py \
  bars/graph/edges.py \
  bars/models/external_policy.py \
  bars/models/dataset_embedding.py \
  bars/external/gas_compat.py \
  bars/experiments/pipeline.py \
  scripts/make_stage20_sota_sweeps.py
```

Protocol-fix sweep:

```bash
MODE=protocol GPUS=0,1,2,3 LOG_ROOT=runs_stage20_protocol_fix \
  bash scripts/run_stage20_sota_alignment.sh
```

Budget sweep:

```bash
MODE=budget GPUS=0,1,2,3 LOG_ROOT=runs_stage20_budget \
  bash scripts/run_stage20_sota_alignment.sh
```

GAS same-backbone sweep:

```bash
MODE=gas GPUS=0,1,2,3 LOG_ROOT=runs_stage20_gas \
  GAS_ARTIFACT_ROOT=/path/to/gas_artifacts \
  GAS_REPO_PATH=/path/to/GAS \
  GAS_POLICY_CKPT_ROOT=/path/to/gas_policy_ckpts \
  bash scripts/run_stage20_sota_alignment.sh
```

HIQL low-level sweep:

```bash
MODE=hiql GPUS=0,1,2,3 LOG_ROOT=runs_stage20_hiql \
  HIQL_REPO_PATH=/path/to/HIQL \
  HIQL_POLICY_CKPT_ROOT=/path/to/hiql_ckpts \
  bash scripts/run_stage20_sota_alignment.sh
```
