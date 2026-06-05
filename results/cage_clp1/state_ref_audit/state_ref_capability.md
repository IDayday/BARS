# CAGE-CLP1 StateRef Capability Audit

| env_name | can_construct | reset_seed_works | observation_shape | action_shape | qpos_qvel_available | has_unwrapped_set_state | has_sim_set_state | has_mujoco_data_qpos_qvel | restore_test_passed | max_abs_obs_error | dataset_or_env_exposes_state_ids | recommended_state_ref_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-giant-navigate-v0 | True | True | [29] | [8] | True | True | False | True | True | 0 | False | exact_mujoco_state |
| antmaze-giant-stitch-v0 | True | True | [29] | [8] | True | True | False | True | True | 0 | False | exact_mujoco_state |
| humanoidmaze-large-navigate-v0 | True | True | [69] | [21] | True | True | False | True | True | 0 | False | exact_mujoco_state |

## Per-Env Notes

### antmaze-giant-navigate-v0

- Exact restore test passed.

### antmaze-giant-stitch-v0

- Exact restore test passed.

### humanoidmaze-large-navigate-v0

- Exact restore test passed.

