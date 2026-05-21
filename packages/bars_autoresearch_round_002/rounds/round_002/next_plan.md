# Round 002 Next Plan

## Decision
Round 003 artifact acquisition / full-budget training plan.

## Primary question for next round
Can we obtain certified public-quality GAS artifacts for the medium certification targets, or should the certification target move to an official-artifact environment?

## Gates that unlock this question
- PASS_BASELINE_REGISTRY
- Need PASS_BASELINE_CERTIFICATION before adapter certification or scientific diagnostics.

## Experiments to run
- Audit whether official medium GAS checkpoints exist outside the current Hugging Face listing.
- If unavailable, plan full-budget 1M-step GAS training for antmaze-medium-stitch-v0 and antmaze-medium-navigate-v0.
- As a fallback audit target, consider certification on antmaze-large-explore-v0, antmaze-giant-stitch-v0, or scene-play-v0 where official artifacts are listed.

## Commands
```bash
python scripts/build_baseline_registry.py --round 003
bash scripts/certify_gas_baseline.sh ENVS=antmaze-large-explore-v0,antmaze-giant-stitch-v0,scene-play-v0 SEEDS=0 ROUND=003 USE_OFFICIAL_ARTIFACTS=1 FULL_BUDGET_ONLY=1
```

## Expected outcomes
- Either PASS_BASELINE_CERTIFICATION on an official-artifact target, or a concrete full-budget training queue for medium targets.

## Stop conditions
- STOP_REPO_OR_ARTIFACT_MISSING if official artifacts cannot be located and full-budget training is not feasible.
- STOP_COMPUTE_BUDGET_EXHAUSTED if 1M-step certification training cannot be run.
