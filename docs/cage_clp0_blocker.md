# CAGE-CLP0 Blocker

HumanoidMaze closed-loop probes from existing q_G artifacts are blocked.

Reason:

- Runtime `humanoidmaze-large-navigate-v0` exposes `env.unwrapped.set_state(qpos, qvel)` and MuJoCo `data.qpos/qvel`.
- Existing OGBench dataset rows do not contain raw `qpos/qvel`.
- Humanoid observation shape is 69, while `qpos || qvel` has shape 55.
- Therefore observation-only dataset rows cannot be restored exactly.

Do not approximate humanoid closed-loop success from phi-space support or observation-only reset.

Required future instrumentation:

1. During GAS/CAGE rollout, call `make_state_ref_from_env(...)` at every selected subgoal transition.
2. Store `state_ref_s` with exact `qpos/qvel`.
3. Store `phi_g`, target type, path position, recovery/final flags.
4. Re-run CLP0 probes only on rows where `state_ref_s.exact_reset=true`.
