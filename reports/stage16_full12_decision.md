# Stage16 full12 decision

Decision: GO_EDGE_ROLLOUT

Run status:
- completed: 12/12
- failed: 0/12
- running: 0/12

Graph timing4:
- Initial warm-start attempt with FAISS GPU failed 4/4 at `select_nodes` due to `cublas failed (13)`.
- After switching to CPU ANN (`ann.backend=auto`, `ann.use_gpu=false`) without changing graph/boundary budgets, timing4 completed 4/4.
- graph-related total time per run: 5.504 to 5.742 sec.
- This is orders of magnitude below the Stage 1 v2 2-3 hour graph-build black-box and comfortably below the continue thresholds.

Full12 graph/profile:
- graph-related total time per run: 5.477 to 5.698 sec.
- Typical graph phase times:
  - `select_nodes`: about 1.53 to 1.57 sec
  - `build_edges`: about 0.74 to 0.83 sec
  - `build_boundary`: about 3.28 to 3.41 sec including support-modes construction
- Diagnostics are now the dominant per-run cost, not graph construction.

Support-modes boundary:
- `supported_pair_rate`: about 0.2286 to 0.2417 depending on env.
- `psi_mean`: about 0.1888 to 0.2022.
- `psi_p50`: about 0.1061 to 0.1147.
- `psi_p90`: about 0.5145 to 0.5519.
- `supported_edge_arr_rate` / `supported_edge_dep_rate`: about 0.4174 to 0.4341.
- Boundary signal is active and non-trivial; this is no longer a direction-only placeholder result.

PU / balanced edge diagnostics:
- Exact edge proxy AUC by env: about 0.7411 to 0.8855.
- Exact edge proxy AUPRC by env: about 0.0059 to 0.0236.
- Balanced edge AUC by env: about 0.7717 to 0.7831.
- Balanced edge AUPRC by env: about 0.7564 to 0.7802.
- `supported_edge_rate`: about 0.4650 to 0.4807.
- `selected_supported_rate`: about 0.4482 on large envs and 0.5230 to 0.5397 on medium envs.
- `selected_hard_neg_proxy_rate`: about 0.0949 to 0.1590.
- `score_supported_mean` remains well above `score_hard_neg_proxy_mean`, while `score_unlabeled_bridge_mean` stays close to supported edges rather than hard-negative proxies.
- `cross_traj_selected_rate` stays high in exact proxy diagnostics (about 0.9708 to 0.9842), so the next validation step should be edge rollout rather than treating cross-trajectory edges as errors.

Path risk-cost Pareto:
- `found=1.0` across envs, variants, and all lambda values in this diagnostic set.
- As `lambda_risk` increases, `full_bars` consistently reduces `total_risk` while increasing `total_cost` and `total_boundary`, producing a clean Pareto tradeoff.
- Example, `antmaze-medium-play-v2` full_bars:
  - lambda 0.0: risk 1.4949, cost 1.4023, boundary 0.4682
  - lambda 1.0: risk 1.0100, cost 1.5247, boundary 0.6310
  - lambda 3.0: risk 0.8701, cost 1.7024, boundary 0.8174
- The same monotonic pattern holds on medium-diverse, large-play, and large-diverse.
- Compared with `shortest`, `reachability` and `full_bars` both reduce risk as lambda rises; `full_bars` adds a boundary penalty term and traces a distinct cost-risk-boundary frontier rather than collapsing to shortest-path behavior.

Recommendation:
- Do not auto-start online eval in this turn.
- Next step should be edge rollout diagnostics on selected env/seed slices to verify that the high cross-trajectory selection rate corresponds to executable bridge edges.
- If edge rollout looks healthy, then proceed to a small quick online eval.
