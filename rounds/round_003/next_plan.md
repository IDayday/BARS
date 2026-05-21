# Round 004 Next Plan

Run certified failure diagnostics and oracle-headroom scan only on Round003 certified envs.

- Certified envs: scene-play-v0
- Use fallback_mode=none for planner evidence.
- Do not train p_bridge until PASS_ORACLE_HEADROOM.
- Keep medium tasks on artifact acquisition or full-budget training until their own baseline and adapter certification pass.
