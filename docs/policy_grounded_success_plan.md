# Policy-Grounded Success Plan

Current evidence shows that support-certified graphs improve diagnostics and
can reduce final-goal distance in small AntMaze natural-start smokes, but they
do not yet improve task success. The next work should prioritize changes that
alter the final trained policy or the closed-loop execution protocol.

## Current Evidence

AntMaze large stitch, task id 1, 3 natural-start episodes, 120-step cap:

| method | success | mean final L2 | interpretation |
| --- | ---: | ---: | --- |
| Direct GCBC to final goal | 0.0 | 43.4040 | The trained policy cannot solve the task by directly chasing the reset goal. |
| Phase 5I state-conditioned outcome planner | 0.0 | 39.7197 | The graph helps partial progress but not success. |
| Phase 5K preplan policy mismatch | 0.0 | 39.8942 | Policy-risk features reduce some failures but do not solve execution. |
| Phase 5L progress guard | 0.0 | 42.8189 | Faster abort/replan alone hurts progress. |

The local `gcrlo` OGBench adapter returns a real 29-dimensional reset goal and
`info["success"]` from `env.step`, so this is not merely a missing success-field
problem.

## Working Diagnosis

The main bottleneck has moved from graph construction to policy execution:

1. The support graph can propose plausible option paths.
2. The GCBC policy can fit offline one-step action labels.
3. At runtime, planner-selected subgoals are often outside the policy's robust
   closed-loop controllability region from the current state.
4. Replanning, edge risk penalties, and early aborts only reshuffle these weak
   executions; they do not teach the policy to execute the planner's subgoals.

## Required Next Direction

Future algorithm changes should be accepted only if they connect graph decisions
to final-policy training or closed-loop success. A useful next phase should have
all three parts:

1. **Success protocol lock**
   - Verify reset goal, task ids, horizon, action clipping, and `info["success"]`
     against OGBench/GAS evaluation assumptions.
   - Report direct GCBC, hierarchical support, and GAS-style references under
     the same episode/task protocol.

2. **Planner-subgoal policy training**
   - Build training examples from the subgoal distribution actually emitted by
     the planner/subgoal selector, not only from arbitrary offline edge
     terminations.
   - Mix final-goal GCBC, support-edge GCBC, and planner-issued subgoal GCBC.
   - Track validation MSE separately for final-goal, edge-local, and
     planner-subgoal samples.

3. **Closed-loop policy gate**
   - Evaluate natural-start success as the primary metric.
   - Treat final L2, completed edges, and subgoal reach rate as secondary
     diagnostics.
   - Reject graph-only changes that improve coverage/risk metrics but do not
     improve direct or hierarchical closed-loop behavior.

## Near-Term Candidate Algorithm

The next concrete algorithm should be a policy-grounded planner, not another
graph-only cost:

```text
support graph + planner-issued subgoal replay
    -> train GCBC on final-goal / edge-goal / planner-subgoal mixture
    -> at runtime choose only subgoals with policy-consistency evidence
    -> evaluate natural-start success under fixed OGBench protocol
```

This keeps the support-certified graph as the semantic backbone, but makes the
policy training distribution match the executor's actual subgoal commands.

Before implementing the full version, review relevant goal-conditioned offline
RL and hierarchical planning work, including GAS/TDR/TMD/MQE code paths where
available, so the training protocol follows mature practice rather than adding
another ad hoc planner penalty.

