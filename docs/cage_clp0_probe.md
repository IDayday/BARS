# CAGE-CLP0 Probe Protocol

Purpose: estimate the frozen low-level policy execution contract for graph-planned state-goal pairs.

Definitions:

- `R_pi(s,g)`: hit probability within horizon `H`.
- `Delta_pi(s,g)`: `d_phi(s,g) - d_phi(s_H,g)`.
- `normalized_progress`: `Delta_pi / d_phi(s,g)`.

Probe execution:

1. Read q_G pairs with `state_ref_s`.
2. Keep only pairs with `probeable=true` and exact reset.
3. Restore the environment with `qpos/qvel`.
4. Use the same GAS low-level policy checkpoint.
5. Compute the same skill interface as `evaluate_gas.py`:
   `skill = (phi_g - phi_obs) / norm(phi_g - phi_obs)`.
6. Roll out the frozen policy for `H` steps.
7. Record hit, progress, action norms, termination flags, and reset mode.

No CAGE target selection is used in the probe. The probe estimates raw low-level policy competence on graph-planned pairs.

If no exact StateRef is available, the probe writes a failure record and does not run an approximate simulator rollout.
