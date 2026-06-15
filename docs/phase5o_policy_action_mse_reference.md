# Phase 5O Policy Action-MSE Reference Plan

This note records how action MSE should be used as a policy-development
reference for BARS.

## Current Answer

We should not quote a GAS, HIQL, CRL, or TMD action-MSE number from papers,
because their public benchmark tables report online return/success, not
heldout one-step action MSE. The number must be measured under a matched local
protocol.

References checked:

- OGBench project page and code release: https://seohong.me/projects/ogbench/
  and https://github.com/seohongpark/ogbench
- HIQL project page: https://seohong.me/projects/hiql/
- GAS project/code references should be treated as coupled graph-policy systems,
  not as plug-in actor checkpoints.

What we can say from code:

- GAS computes actor `mse = mean((dist.mode() - batch["actions"]) ** 2)` in
  `external_src/GAS/M_utils/agents/gas.py`.
- OGBench reference agents GCBC, GCIVL, GCIQL, QRL, CRL, TMD, and HIQL also
  expose the same style of actor `mse` in `external_src/tmd-release/impls/agents/`.
- Therefore a unified local actor-MSE benchmark is feasible.

What we can say from BARS evidence:

- Phase 3A edge-GCBC 100000-step AntMaze checkpoint:
  `final val_action_mse = 0.0426389`.
- Phase 5N planner-aware GCBC full run is in progress. At step 40000,
  `val_action_mse = 0.0530972`, still descending.
- Phase 5N 200-step smoke was only a wiring check, not a performance target.

## Why This Matters

Action MSE is a useful low-level policy diagnostic, but it is not the final
success metric.

Interpretation:

- If BARS has worse action MSE than GAS/HIQL under matched target semantics,
  the low-level policy is likely too weak.
- If BARS has similar action MSE but much worse success, the bottleneck is
  likely goal/skill representation, planner target distribution, subgoal
  switching, or closed-loop compounding error.
- If BARS has lower action MSE but still zero success, one-step supervised
  fitting is not the right proxy and policy training needs rollout-aware or
  recovery-aware data.

## Required Matched Protocol

Every policy must be evaluated with explicit target semantics:

1. **Native policy target MSE**
   - GAS: action predicted from GAS TDR direction target.
   - HIQL: low actor action predicted from HIQL's low-level target.
   - BARS: action predicted from BARS raw subgoal/edge target.
   - This asks whether each policy fits the target distribution it was trained
     for.

2. **BARS-target compatibility MSE**
   - Feed BARS planner/edge targets to each policy only if the policy's target
     representation supports a valid conversion.
   - GAS requires `phi(obs)` and `phi(goal)` to construct the direction vector;
     this is a compatibility diagnostic, not a final algorithm.

3. **Online natural-start success**
   - Action MSE must always be paired with success rate, final goal distance,
     completed edges, replans, and failed edge attempts.

## Practical Target

For the current AntMaze H10 B120 setting, BARS-native low-level policy
development should use the Phase 3A 100000-step edge-GCBC result
`0.0426389` as the immediate supervised baseline. A planner-aware policy that
cannot approach this number is not yet a stronger low-level policy. A
planner-aware policy that matches or beats it must still prove natural-start
success.

## Next Implementation

Build a `phase5o` benchmark script that:

- reads BARS PyTorch checkpoints and reports action MSE by target family;
- inventories live GAS / OGBench reference checkpoints;
- when artifacts are live, computes native actor MSE for GAS, HIQL, CRL, and
  GCBC using the same heldout dataset sampling budget;
- writes a table with both action MSE and any available online success record.

This benchmark should be run before deciding whether BARS needs a stronger raw
GCBC, a GAS-like learned skill representation, or rollout/recovery-aware
training.
