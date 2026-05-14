# Stage19 Route-A Protocol Ablation Implementation

This patch implements the next Route-A experiment plan:

1. Log `eval.condition` in every eval row so mixed protocol-ablation sweeps are analyzable.
2. Group Stage3 eval summaries by `condition/env/variant` when condition exists.
3. Add generated medium protocol-ablation sweeps:
   - `configs/sweeps/d4rl_stage19_protocol_ablation_medium50.json`
   - `configs/sweeps/d4rl_stage19_protocol_ablation_medium100_core.json`
4. Add `scripts/compare_protocol_ablation.py` for causal analysis of Stage18/Stage19 online results.
5. Add `scripts/package_stage19_protocol_results.sh` for uploading results.

The current `bars/eval/rollout.py` already supports `variant=direct_goal` and `eval.fallback_mode in [none, planner_only, direct_goal, direct_goal_after_k]`; this patch makes those capabilities usable for protocol ablation and analysis.
