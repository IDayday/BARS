# CAGE-CLP1 Policy Alignment Plan

The policy-alignment audit builds graph-induced hard-goal examples from closed-loop contract labels.

Categories:
- `hard_positive`: a hard target with direct action supervision, such as q_train/hindsight-style controls.
- `hard_unlabeled`: graph planner wants the target, but no direct supervised action label is known.
- `hard_negative`: closed-loop probe shows a poor contract; use for contract/ranking or conservative objectives, not naive behavior cloning.

CLP1 does not finetune the low-level policy by default. Policy training is only feasible after the hard-goal summary shows enough `hard_positive` examples and the existing GAS `train_policy.py` can accept an additional goal sampler without changing TDR/keygraph semantics.
