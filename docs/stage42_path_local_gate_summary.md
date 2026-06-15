# Stage42 Path-Local Support Gate

Stage42 tested the next refinement after Stage41.  Instead of mixing whole-task
cached path tables, it gates each individual `(task_id, source_node)` path
change.  A candidate support-risk path is accepted only when it improves local
offline support and stays close to the original GAS geometric cost.

## Implementation

New script:

- `scripts/stage42_path_local_gated_keygraph.py`

The script keeps the official GAS actor and node space unchanged.  It edits only
`task_paths_dict` and `task_paths_dist_dict`.  Candidate path costs are evaluated
on the original GAS graph, not on the risk-weighted graph.  By default, accepted
paths write their original-graph cost back to `task_paths_dist_dict`, preventing
support-risk weights from distorting GAS nearest-source selection.

The implementation also avoids deep-copying JAX arrays in keygraph objects, so
offline keygraph editing can run with `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`
even when GPUs are busy.

## Variants

| method | candidate paths | max original cost ratio | selected paths | path change rate |
| --- | --- | ---: | ---: | ---: |
| `gate_w5_cost102` | `w=5` | `1.02` | 37 | `0.010` |
| `gate_w5_cost105` | `w=5` | `1.05` | 109 | `0.030` |
| `gate_w5_cost108` | `w=5` | `1.08` | 118 | `0.032` |
| `gate_multi_cost105` | `w=2,5,10,20` | `1.05` | 124 | `0.034` |
| `gate_multi_cost108` | `w=2,5,10,20` | `1.08` | 212 | `0.058` |

## Offline Path Metrics

| method | unsupported edge fraction | mean same-traj support | mean path edges |
| --- | ---: | ---: | ---: |
| original | `0.973` | `11.240` | `2.262` |
| global `w=5` | `0.963` | `16.367` | `2.259` |
| `gate_w5_cost108` | `0.963` | `16.526` | `2.262` |
| `gate_multi_cost105` | `0.962` | `16.375` | `2.262` |
| `gate_multi_cost108` | `0.954` | `18.861` | `2.265` |

`gate_multi_cost108` is the strongest graph-side variant: it reaches roughly
the same path-change rate as global `w=5` while improving the support metrics
more.

## Closed-Loop Results

Scene ep50 closed-loop GAS results:

| method | success | mean length |
| --- | ---: | ---: |
| original | `0.756` | `343.468` |
| global `w=5` | `0.764` | `348.896` |
| `gate_w5_cost108` | `0.732` | `355.588` |
| `gate_multi_cost105` | `0.724` | `361.548` |
| `gate_multi_cost108` | `0.728` | `361.404` |

Per-task results show the same pattern.  The gated variants improve offline
support metrics on every task, but degrade most task success rates.  Task 5
stays high, while task 1-4 lose enough performance to erase any graph-side
benefit.

## Interpretation

This is an important negative result.  More local and cost-normalized support
gating still does not make Scene better.  The support-only objective is selecting
paths that look safer in the dataset but are not better matched to the existing
GAS low-level actor.

The useful rule is now sharper:

- Offline graph support is necessary evidence, but not sufficient for changing
  an execution path.
- Path-local gating is better engineered than whole-task mixing, but it still
  fails without policy/execution compatibility.
- The next algorithmic step should score path changes using actor-aware signals:
  closed-loop trace outcomes, subgoal progress, local intervention success, or a
  learned compatibility/contract model trained from execution traces.
- For Scene, support-risk should act as a veto against obvious graph shortcuts,
  not as the primary path-selection objective.

Result files:

- `runs_stage42_path_local_gate/scene-play-v0/seed0/stage42_gate_eval_summary.csv`
- `runs_stage42_path_local_gate/scene-play-v0/seed0/stage42_selection_by_task.csv`
- `runs_stage42_path_local_gate/scene-play-v0/seed0/stage42_path_metrics_by_task.csv`
- `runs_stage42_path_local_gate/scene-play-v0/seed0/stage42_gate_summary.json`
- `runs_stage42_path_local_gate/scene-play-v0/seed0/path_audit_gate_sweep/path_summary.csv`
- `runs_stage42_path_local_gate/scene-play-v0/seed0/path_audit_gate_sweep/path_diff_summary.csv`
