# Stage23 Initial Audit

Updated: 2026-05-19

## Repository And GAS Source

- The repo contains a full `external_src/GAS` checkout with the official upstream remote `https://github.com/qortmdgh4141/GAS` at commit `c9e590fcd6f082de677d332a84e44a1a631da5c5`.
- The GAS checkout is not pristine. Local patches modify `evaluate_gas.py`, `pretrain_tdr.py`, `train_policy.py`, `construct_graph.py`, `K_utils/graph_builder.py`, and `O_utils/log_utils.py`.
- The local GAS patches are mostly execution-environment compatibility changes: WandB shim/TensorBoard logging, `MUJOCO_GL` default handling, and a `setup_task_env` call repair in graph construction.
- Because the tree is dirty, Stage23 route A must distinguish pristine official GAS from locally patched official GAS. Current scripts record the commit and dirty state in `stage23_protocol_audit.md`.

## GAS Adapter Status

- `bars/external/gas_backbone.py` directly loads official GAS agent, TDR, policy, environment, dataset, and keygraph classes.
- `bars/gas_bars/evaluate.py` does not call official `evaluate_with_graph` for execution. It reimplements the evaluation loop, replanning, virtual start/goal connectors, fallback triggers, and planner variants around the GAS backbone.
- Therefore the A/B/C reproduction matrix is mandatory:
  - A: official GAS eval with official/pretrained checkpoint if available.
  - B: official GAS eval with our trained checkpoint.
  - C: BARS adapter eval with the same checkpoint.

## Current Artifacts

- `antmaze-medium-navigate-v0` and `antmaze-medium-stitch-v0` have complete local GAS artifacts, but the checkpoints are `params_100000.pkl`, not full official `params_1000000.pkl`.
- `antmaze-large-explore-v0` has HuggingFace official artifacts at `params_1000000.pkl`.
- The official README lists pretrained HuggingFace artifacts for giant navigate, giant stitch, large explore, scene, kitchen, and visual variants. It does not list pretrained medium navigate/stitch checkpoints.

## Stage22 Versus GAS Paper / Protocol

- Stage22 medium same-backbone baseline is healthy but not yet an official reproduction:
  - `gas_shortest` no-fallback success: medium-navigate `0.89`, medium-stitch `0.86` in the latest 100-episode summary.
  - Earlier Stage22 pilot also reported medium baselines around `0.86-0.92`.
- This is encouraging for the adapter but cannot be treated as GAS-paper reproduction because:
  - medium artifacts are 100k quick-trained checkpoints rather than official 1M checkpoints;
  - route C uses the BARS adapter loop, not official `evaluate_gas.py`;
  - official paper/README commands use `eval_episodes=49` per task and `eval_final_goal_threshold=2`, while Stage22 adapter runs 100 aggregate episodes with custom replanning behavior.

## p_exec Saturation Evidence

Stage22 reachability diagnostics show strong global AUROC but saturated selected-edge probabilities:

- medium-navigate: `val_auroc=0.941`, `val_auprc=0.954`, `p_exec_mean=0.9976`, `p_exec_q50=0.999997`, `p_exec_q90=1.0`.
- medium-stitch: `val_auroc=0.916`, `val_auprc=0.941`, `p_exec_mean=0.9961`, `p_exec_q50=0.999995`, `p_exec_q90=1.0`.

This supports the Stage23 decision that all-edge `p_exec` is not enough evidence for reachability on selected GAS paths.

## Boundary 100% Reject Initial Diagnosis

- Stage22 finalized summary shows `gas_boundary_budget` at budgets `2.0` and `3.0` has `success=0`, `steps=0`, `no_path_rate=1.0`, `budget_reject_rate=1.0` on both medium envs.
- Boundary summaries themselves are dense: medium-navigate `coverage=15.78`, medium-stitch `coverage=23.24`, supported pair rate around `0.96`, median psi around `0.84`.
- Initial diagnosis: the issue is not lack of boundary-score coverage. It is risk-scale/objective misuse: boundary cost is accumulated over all consecutive local edges, so healthy long local GAS paths exceed budget and collapse to no-path.
- Stage23 should restrict boundary to bridge junctions only.

## Protocol Risks To Audit

- `eval horizon`: current adapter defaults to `max_steps=1000`; official env horizon must be read from env spec and compared.
- `goal source`: adapter requires `info['goal']` and refuses random fallback, which is good, but planner-goal versus env-goal hashes need logging.
- `task_id`: official eval iterates all `env.task_infos`; adapter aggregate scheduling cycles task ids by episode. Distribution must be recorded.
- `success threshold`: official OGBench wrapper maps normalized return close to `100` into success; adapter reads success from env info. This needs route A/C comparison.
- `H_TD / way_steps`: GAS AntMaze uses `way_steps=8`; Stage22 subgoal threshold defaults are much looser (`8.0` in phi distance), which may hide path execution mismatch.
- `checkpoint step`: medium current artifacts are quick `100000` step checkpoints, not full reproduction.
- `render_goal/options`: official reset uses `options=dict(task_id=task_id, render_goal=should_render)`; adapter uses the same shape but needs logging.
- `steps field`: Stage22 adapter `steps` is env step count in the debug loop, while official `episode.length` comes from the environment monitor.

## Immediate Blockers

1. Reproduction is HOLD until official-eval route B and adapter route C are compared on the same medium checkpoint, and full 1M training is either run or the 100k gap is explicitly accepted as quick sanity only.
2. Route A on medium cannot use official pretrained HF artifacts because the official README does not list medium pretrained checkpoints.
3. Local `external_src/GAS` is dirty; a pristine clone or explicit patch manifest is needed for clean official-route claims.
4. Bridge/oracle conclusions require harder or official-pretrained envs with full artifacts; currently only `antmaze-large-explore-v0` has a 1M official artifact locally.
5. Edge execution oracle may be limited by arbitrary reset support. If `set_state` fails, results must be marked weak proxy and not used as oracle evidence.
