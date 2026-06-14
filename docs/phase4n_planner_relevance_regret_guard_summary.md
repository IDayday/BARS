# Phase 4N Planner-Relevance Regret Guard

Phase 4N is a small reset-free offline supervised follow-up to Phase 4M. It
keeps the same repaired support-certified Scene H10 graph and the same GCBC
training setup, but sweeps weaker planner-relevant loss-weight strengths to
reduce the overall validation-MSE regret observed in the original `s04`
setting.

No environment rollout is used. These are offline supervised action-fitting
metrics, not option execution success.

## Setup

- Dataset/run: Scene `core_plus_bottleneck_budget192_H10`
- Baseline: `augmented_loss_support_bottleneck_s03`
- Existing Phase 4M strong setting: `planner_relevant_repair_s04`
- New guarded settings:
  - `planner_relevant_repair_s02`: planner relevance `0.2`, hard repair `0.1`,
    loss clip `[0.75, 1.9]`
  - `planner_relevant_repair_s01`: planner relevance `0.1`, hard repair `0.05`,
    loss clip `[0.8, 1.7]`
- Seeds: `[0, 1]`
- Steps: `3000`

The related-method assumptions are unchanged from Phase 4M: this is clipped
supervised loss weighting inspired by goal-conditioned supervised learning,
long-tail/class-balanced weighting, and priority signals. It is not TD-error
replay and it does not introduce unsupported graph edges.

## Results

| method | final val MSE ratio | direct repair MSE ratio | planner-used repair MSE ratio | policy support ratio |
| --- | ---: | ---: | ---: | ---: |
| `planner_relevant_repair_s01` | 1.006709 | 1.000639 | 0.994857 | 0.998925 |
| `planner_relevant_repair_s02` | 1.005790 | 0.988326 | 0.960478 | 1.002015 |
| `planner_relevant_repair_s04` | 1.008751 | 0.993475 | 0.962466 | 1.000197 |

`planner_relevant_repair_s02` is the best guarded Scene H10 choice in this
sweep. It keeps the planner-used repair-edge MSE improvement from the strong
`s04` setting, improves direct repair-edge MSE more than `s04`, and reduces
overall validation-MSE regret from about `0.875%` to about `0.579%`.

The very weak `s01` setting reduces planner-used repair MSE only mildly and
does not improve direct repair-edge MSE, so it is too weak for the intended
target.

## Interpretation

The Scene H10 trade-off is not eliminated, but it is better controlled. The
current practical rule is:

- Use `planner_relevant_repair_s02` as the guarded Scene H10 setting.
- Treat `s04` as an aggressive setting when planner-used repair-edge MSE is the
  only target.
- Do not make planner-relevant weighting a global default until Scene H25,
  longer training, and closed-loop evaluation are available.

Artifacts:

- `configs/phase4m_planner_relevant_loss_weighting_scene_H10_B192_regret_sweep_3000.yaml`
- `results/phase4m/scene_play/core_plus_bottleneck_budget192_H10_3000/`
- `results/phase4m_training/scene_play/core_plus_bottleneck_budget192_H10_3000/`
