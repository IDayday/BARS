# CAGE-CLP0 StateRef

`StateRef` is the reset contract used by closed-loop policy probes.

Exact reset requires:

- MuJoCo `qpos`
- MuJoCo `qvel`
- `env.unwrapped.set_state(qpos, qvel)`

Observation-only records are explicitly marked `observation_only_not_exact` and are not treated as probeable unless a caller opts into approximate behavior. CLP0 does not use approximate reset for reported closed-loop contracts.

Main helpers:

- `make_state_ref_from_env(env, obs=None, phi=None, metadata=None)`
- `restore_env_from_state_ref(env, state_ref)`
- `state_ref_is_exact(state_ref)`
- `serialize_state_ref(state_ref)`
- `deserialize_state_ref(record)`

For OGBench locomaze:

- AntMaze observations equal `qpos || qvel`, so dataset observations can be converted to exact StateRefs.
- HumanoidMaze observations do not equal `qpos || qvel`; existing dataset-only q_G pairs are not exact-reset capable.

This distinction is intentional. CLP0 must not infer closed-loop success from phi-pair support or observation-only reset.
