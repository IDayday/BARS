# CAGE-CLP1 StateRef

`StateRef` is the branchable simulator-state handle used by CAGE-CLP1 closed-loop probes.

It stores compact serializable fields only:
- `obs`, `phi`, optional `goal_obs`, optional `goal_phi`
- MuJoCo `qpos` and `qvel` when available
- rollout metadata: `env_name`, `seed`, `episode_idx`, `step_idx`, `source`, `source_variant`
- `reset_mode`: `exact_mujoco_state`, `exact_dataset_state`, `obs_only_not_exact`, or `unsupported`

Rules:
- Observation-only records are never treated as exact.
- `restore_state_ref(...)` raises unless qpos/qvel and `env.unwrapped.set_state(...)` are available, unless an explicit approximate mode is later implemented.
- CLP1 probes must skip records where `is_exact_state_ref(...)` is false.

Primary API:
- `capture_state_ref(env, obs=None, phi=None, metadata=None)`
- `restore_state_ref(env, state_ref, allow_approximate=False)`
- `is_exact_state_ref(state_ref)`
- `serialize_state_ref(state_ref)`
- `deserialize_state_ref(record)`
- `compare_state_ref_restore(env, state_ref, encoder=None)`

This is an instrumentation layer only. It does not modify GAS planning, TDR, keygraph construction, low-level policy training, or policy action computation.
