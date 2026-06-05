# CAGE-CLP0 Reset Audit

Run:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python scripts/audit_env_reset_capability.py \
  --envs antmaze-giant-navigate-v0 antmaze-giant-stitch-v0 humanoidmaze-large-navigate-v0 \
  --out_json results/cage_clp0/reset_audit/reset_capability.json \
  --out_md results/cage_clp0/reset_audit/reset_capability.md
```

The generated markdown is the source of truth for the current environment.

Expected interpretation:

- AntMaze can be probed from dataset observations because observation equals `qpos || qvel`.
- HumanoidMaze runtime can capture exact MuJoCo state, but existing dataset observations do not reconstruct `qpos/qvel`; existing q_G pairs are not exact-reset capable unless future instrumentation stores exact state refs during rollout.
