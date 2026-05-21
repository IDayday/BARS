# AGENTS.md — BARS baseline-first research instructions

Before doing any BARS autonomous research task, read:

- `BARS_AUTONOMOUS_RESEARCH_LOOP_BASELINE_FIRST.md`

Core rule:

> Baseline certification precedes scientific interpretation.

Non-negotiable:
- Reduced training is `E0_SMOKE_ONLY`.
- Do not interpret failure modes, oracle headroom, p_bridge, boundary, or integrated BARS results unless baseline and adapter gates pass.
- Planner evidence must use `fallback_mode=none`, unless the task is explicitly a fallback-protocol ablation.
- direct-goal/progress fallback is not planner evidence.
- Do not train p_bridge until `PASS_ORACLE_HEADROOM`.
- Do not run integrated BARS-v3 until `PASS_ORACLE_HEADROOM && PASS_P_BRIDGE`.
- Boundary is diagnostic-only until coverage >= 0.05, supported_gap >= 0.10, and psi_AUROC >= 0.65.
- D4RL is protocol/debug only until `PASS_D4RL_PROTOCOL_AUDIT`.
- Every claim must be tied to a gate, evidence class, and report file.

Default next task:
Run the active round prompt from `BARS_AUTONOMOUS_RESEARCH_LOOP_BASELINE_FIRST.md`.