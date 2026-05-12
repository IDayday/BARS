# BARS Stage 1.6 code update notes

This update addresses the open engineering/research issues raised after Stage 1 v2.

## Implemented

1. **Support-based portal-mode boundary overlap**
   - `bars/graph/boundary.py` now supports `boundary.method=support_modes`.
   - It samples same-trajectory support segments, maps them to graph edges, and records edge-specific departure/arrival mode histograms.
   - `psi(e_i,e_j)` is histogram overlap between arrival modes of the previous edge and departure modes of the next edge at the shared graph node.
   - Direction smoothness remains only as fallback for unsupported edges.

2. **Optional real edge rollout diagnostics**
   - `bars/eval/edge_rollout_diag.py` implements best-effort edge execution rollouts when reset-to-dataset-state is supported.
   - It logs `edge_rollout_auc`, `edge_rollout_auprc`, `success_rate`, and per-edge rollout rows.
   - Disabled by default; enable with `diagnostics.edge_rollout_enabled=true`.

3. **Optional FAISS / GPU ANN path**
   - `bars/graph/ann.py` introduces `KNNIndex`, using FAISS CPU/GPU if installed and falling back to sklearn otherwise.
   - Nodes, edges, support mapping, boundary and path diagnostics now use this wrapper where relevant.

4. **Full-stage graph/profile reporting**
   - `bars/common/profile.py` writes `logs/profile.csv` with timings for load, training, embedding, node selection, edge build, boundary build and diagnostics.
   - `scripts/analyze_bars_results.py` now outputs `*_profile_summary.csv` and a report section.

5. **PU reachability training**
   - `bars/training/reach_train.py` adds `reachability.loss_mode=pu`.
   - Same-trajectory short-future pairs are positives, far same-trajectory/latent-near unsupported pairs are hard negatives, cross-trajectory pairs become unlabeled instead of dominating as negatives.

6. **Experiment configs/scripts**
   - `configs/d4rl_antmaze_stage16.json`
   - `configs/sweeps/d4rl_stage16_sanity4.json`
   - `configs/sweeps/d4rl_stage16_full12.json`
   - `configs/sweeps/d4rl_stage16_edge_rollout_quick.json`
   - `scripts/launch_stage16.sh`
   - `scripts/profile_stage16.sh`

## Not guaranteed yet

- Edge rollout reset-to-state depends on D4RL/Gym/MuJoCo wrapper support. If unsupported, the diagnostic logs `reset_unavailable` and skips rather than producing fake labels.
- FAISS acceleration requires installing FAISS. Without it, `KNNIndex` falls back to sklearn and logs `ann_backend=sklearn`.
- Online D4RL final evaluation is still a planned experiment, not a completed result.
