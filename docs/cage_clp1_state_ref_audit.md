# CAGE-CLP1 StateRef Audit

Audit output:
- JSON: `results/cage_clp1/state_ref_audit/state_ref_capability.json`
- Markdown: `results/cage_clp1/state_ref_audit/state_ref_capability.md`

Summary:

| env_name | recommended mode | restore passed | max obs error |
| --- | --- | --- | ---: |
| antmaze-giant-navigate-v0 | exact_mujoco_state | true | 0 |
| antmaze-giant-stitch-v0 | exact_mujoco_state | true | 0 |
| humanoidmaze-large-navigate-v0 | exact_mujoco_state | true | 0 |

All three CLP1 target environments expose `qpos/qvel` and `env.unwrapped.set_state(...)`. The audit performed reset, random rollout, StateRef capture, additional random rollout, restore, and observation comparison.

Decision: exact branchable closed-loop probes are permitted for these runtime rollout states.
