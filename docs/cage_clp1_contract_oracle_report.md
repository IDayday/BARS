# CAGE-CLP1 Contract Oracle Report

Outputs:
- Main 2x2 branchable probes: `results/cage_clp1/probes/`
- Candidate-aware 1x1 probes: `results/cage_clp1/probes_candidate/`
- Main oracle: `results/cage_clp1/oracle/contract_oracle_summary.md`
- Candidate-aware oracle: `results/cage_clp1/oracle_candidate/contract_oracle_summary.md`

## Key Result

Candidate-aware oracle over exact branchable rollout states:
- `num_segments`: 768 segment/horizon groups
- original target hit rate: 0.3880
- oracle hit rate over available candidate modes: 0.3880
- original progress mean: 0.0100
- oracle progress mean: 0.0352

The available target candidates improve progress slightly, but do not improve hit rate in this small CLP1 sample.

## Target Contract Summary At H=64

| env | target | hit | normalized progress | negative progress |
| --- | --- | ---: | ---: | ---: |
| antmaze-giant-navigate-v0 | original_target | 0.6953 | 0.0497 | 0.2969 |
| antmaze-giant-navigate-v0 | nearest_path_target | 0.6094 | 0.0418 | 0.3203 |
| antmaze-giant-navigate-v0 | farther_path_target | 0.0312 | 0.0353 | 0.2656 |
| antmaze-giant-navigate-v0 | final_goal | 0.0000 | 0.0005 | 0.5078 |
| humanoidmaze-large-navigate-v0 | original_target | 0.0859 | -0.0295 | 0.3750 |
| humanoidmaze-large-navigate-v0 | nearest_path_target | 0.0859 | -0.0281 | 0.3750 |
| humanoidmaze-large-navigate-v0 | farther_path_target | 0.0000 | 0.0111 | 0.3672 |
| humanoidmaze-large-navigate-v0 | final_goal | 0.0000 | 0.0002 | 0.4688 |

## Interpretation

- Humanoid has much lower closed-loop contractibility than AntMaze for the same candidate extraction protocol.
- Original GAS target is already the best available hit-rate candidate in this small sample.
- Nearest path target does not rescue humanoid: hit matches original but progress remains negative.
- Farther path and final-goal targets are mostly non-contractible, especially in humanoid.
- q_train matched targets are still unavailable because branchable q_train target sampling has not been wired in.
- Recovery candidates are too sparse in this smoke to make a stable claim.

Decision: CLP1 supports the policy-conditioned contract bottleneck hypothesis more strongly than an inference-time target-switching-only explanation. The next useful step is not risk-aware graph search yet; it is richer candidate recording plus q_train matched branchable controls, then a held-out contract model evaluation.
