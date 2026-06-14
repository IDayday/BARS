# Phase 4H Stronger Scene GCBC Direct Repair Validation Design

Phase 4H tests whether the Scene Phase 4G direct repair-edge evidence depends
on the weak 200-step smoke GCBC model. It trains a longer Scene GCBC model on
the same Phase 2 support-certified edge dataset, then reruns the Phase 4G direct
repair-edge action-MSE evaluation and planner comparison.

## Scope

- Dataset: `scene-play-v0`.
- Graph: `core_plus_bottleneck_budget192_H5` repaired with
  `all_budget192_H5` support-bank edges.
- Training: goal-conditioned behavior cloning on observed edge segments.
- Evaluation: direct repair-edge supervised action MSE, calibrated repair-edge
  certification, and compatibility-aware planning metrics.
- No environment rollout.
- No arbitrary reset assumption.
- No TDR/TMD/MQE implementation.

## Related Work Checked

- Goal-Conditioned Supervised Learning (GCSL):
  https://arxiv.org/abs/1912.06088
- RvS: offline reinforcement learning via supervised learning:
  https://arxiv.org/abs/2112.10751
- GCSL reference implementation:
  https://github.com/dibyaghosh/gcsl

These works support using goal-conditioned supervised learning as a simple,
stable offline baseline. They do not justify interpreting held-out action MSE
as closed-loop option execution success.

## Experimental Change

The previous Scene Phase 4G result used the available 200-step smoke model from
the Phase 3D sampling ablation. Phase 4H adds:

- `configs/phase4h_scene_gcbc_uniform_transition_H5_B192_10000.yaml`
- `configs/phase4h_direct_repair_policy_scene_uniform_transition_H5_B192_10000.yaml`
- `configs/phase4h_scene_repair_validation_uniform_transition_H5_B192_10000.yaml`
- `scripts/run_phase4h_scene_gcbc_repair_validation.py`

The driver reuses existing scripts:

```bash
python scripts/train_phase3_gcbc.py --config configs/phase4h_scene_gcbc_uniform_transition_H5_B192_10000.yaml
python scripts/run_phase4g_direct_repair_policy.py --config configs/phase4h_direct_repair_policy_scene_uniform_transition_H5_B192_10000.yaml
```

It then writes candidate-minus-baseline deltas against the 200-step Phase 4G
Scene baseline.

## Interpretation Rules

- Lower direct repair-edge MSE is stronger supervised policy-fitting evidence.
- Higher direct repair certified rate is stronger offline proxy evidence.
- Planner coverage and compatibility metrics remain graph-level/path-level
  offline metrics.
- None of these metrics are online success until Phase 3C or Phase 3F rollout
  can run in an environment-available setup.
