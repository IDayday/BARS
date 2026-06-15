# Stage41 Task-Conditioned Support Routing

Stage41 tested whether Scene's weak Stage40 gain comes from using one global
support-risk weight for all tasks.  The intervention keeps the official GAS
actor and node embeddings unchanged, and only mixes GAS per-task cached shortest
path tables from different support-risk keygraphs.

## Setup

- Dataset: `scene-play-v0`
- Seed: `0`
- Policy: official GAS `params_1000000.pkl`
- Base keygraph: official GAS keygraph
- Support-risk keygraph: Stage40 calibrated hybrid support, mainly `w=5`
- No policy training and no node-space conversion

Implemented tool:

- `scripts/stage41_mix_task_keygraph_paths.py`

The tool copies `task_paths_dict[task_id]` and `task_paths_dist_dict[task_id]`
from selected method keygraphs into a base keygraph.  This isolates
task-conditioned graph routing from low-level policy changes.

## Variants

| method | task routing rule |
| --- | --- |
| `global_w5` | all tasks use Stage40 `hybrid_w5p00_forward` cached paths |
| `taskmix_low_support_only` | tasks 1-3 use original GAS; tasks 4-5 use `w=5` |
| `taskmix_safe_w5` | tasks 1 and 3 use original GAS; tasks 2, 4, 5 use `w=5` |
| `taskmix_oracle_diagnostic` | post-hoc diagnostic mix from Stage40 ep50 per-task winners |

`taskmix_oracle_diagnostic` is not a deployable algorithm; it is an upper-bound
diagnostic for whether per-task routing has obvious headroom.

## Results

Closed-loop Scene success:

| method | episodes/task | overall success | mean length |
| --- | ---: | ---: | ---: |
| original | 50 | `0.756` | `343.468` |
| global `w=5` | 50 | `0.764` | `348.896` |
| taskmix low-support-only | 50 | `0.764` | `341.896` |
| taskmix safe `w=5` | 50 | `0.724` | `363.472` |
| taskmix oracle diagnostic | 50 | `0.748` | `344.704` |
| original | 200 | `0.741` | `355.008` |
| global `w=5` | 200 | `0.754` | `347.454` |
| taskmix low-support-only | 200 | `0.748` | `352.252` |

Path audit:

| method | unsupported edge fraction | mean same-traj support | path change rate |
| --- | ---: | ---: | ---: |
| original | `0.973` | `11.240` | `0.000` |
| global `w=5` | `0.963` | `16.367` | `0.059` |
| taskmix low-support-only | `0.968` | `12.416` | `0.033` |
| taskmix safe `w=5` | `0.966` | `13.278` | `0.042` |
| taskmix oracle diagnostic | `0.963` | `14.356` | `0.068` |

## Interpretation

Task-conditioned cached-path mixing did not produce a robust Scene improvement.
The best ep200 value remains the simple global `w=5` keygraph (`0.754`), while
`taskmix_low_support_only` reaches `0.748`.  Relative to original GAS, global
`w=5` gains `+0.013` over 1000 total episodes, but the approximate unpaired
standard error of the difference is about `0.019`, so this is not a strong
success-rate claim.

The result narrows the useful algorithmic direction:

- Support-risk graph metrics can improve without reliably improving Scene
  success.
- Simple task-level route selection is too coarse; it cannot repair mismatches
  between graph path support and the low-level policy's actual controllability.
- The next version should score path changes at execution time using policy
  compatibility, progress, and cost-scale-normalized risk, rather than replacing
  whole task caches.
- Scene likely needs policy-aware or closed-loop-aware routing, not only
  data-support-aware graph construction.

Result files:

- `runs_stage41_task_conditioned_support/scene-play-v0/seed0/stage41_taskmix_eval_summary.csv`
- `runs_stage41_task_conditioned_support/scene-play-v0/seed0/stage41_taskmix_eval_with_path_metrics.csv`
- `runs_stage41_task_conditioned_support/scene-play-v0/seed0/stage41_taskmix_summary.json`
- `runs_stage41_task_conditioned_support/scene-play-v0/seed0/path_audit_taskmix/path_summary.csv`
- `runs_stage41_task_conditioned_support/scene-play-v0/seed0/path_audit_taskmix/path_diff_summary.csv`
