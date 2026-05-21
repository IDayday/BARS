# BARS Baseline-First Autonomous Research Loop

**Version:** baseline-first v1
**Date:** 2026-05-20
**Intended user:** Codex / coding agent operating inside the BARS repository
**Primary objective:** run autonomous BARS research loops without confusing under-trained or misconfigured baselines with scientific evidence.

---

## 0. How to use this file

This file is a **repository research runbook and Codex prompt**. It is intentionally longer than a normal `AGENTS.md`.

Recommended layout:

```text
<repo-root>/
  AGENTS.md
  BARS_AUTONOMOUS_RESEARCH_LOOP_BASELINE_FIRST.md
  scripts/
  configs/
  reports/
  research_state/
  rounds/
  packages/
```

Use one of the following patterns.

### Recommended pattern

1. Copy this file to the project root as:

```text
BARS_AUTONOMOUS_RESEARCH_LOOP_BASELINE_FIRST.md
```

2. Create a short root-level `AGENTS.md` that tells Codex to read this file before any BARS research task.

Example root `AGENTS.md`:

```md
# AGENTS.md — BARS repository instructions

Before doing any BARS research, read:

- `BARS_AUTONOMOUS_RESEARCH_LOOP_BASELINE_FIRST.md`

Non-negotiable:
- Baseline certification precedes scientific interpretation.
- Reduced training is smoke only.
- Do not interpret failure modes, oracle headroom, p_bridge, boundary, or integrated BARS results unless the relevant baseline and adapter gates pass.
- Planner evidence must use `fallback_mode=none`, unless the task is explicitly a fallback-protocol ablation.
- Do not train p_bridge until `PASS_ORACLE_HEADROOM`.
- Do not run integrated BARS-v3 until `PASS_ORACLE_HEADROOM && PASS_P_BRIDGE`.
```

3. Start Codex from the project root.

```bash
cd <repo-root>
codex
```

4. Paste the active round prompt, e.g. the Round 002 prompt in Section 28 of this file.

### Alternative pattern

Paste this entire file into Codex as the first prompt. This is acceptable for a one-off session, but it is less durable than keeping the file in the repository and using `AGENTS.md`.

### Important note

Do **not** assume that an arbitrary Markdown file is automatically loaded just because it is in the root directory. The root `AGENTS.md` shim is the safer way to ensure Codex receives the durable instructions.

---

## 1. Research discipline: baseline-first or no scientific claim

The BARS project must not mistake weak engineering runs for scientific evidence.

Before discussing whether a scientific problem exists in an environment, Codex must establish:

1. Which algorithm was run.
2. Whether the algorithm is a strong public baseline for the environment.
3. What public performance was reported.
4. Whether the exact environment has a public number.
5. If not, which related environments have public numbers.
6. What training budget and hyperparameters are required for the public-quality run.
7. Whether the current run used official artifacts, official evaluation, full-budget training, or a reduced smoke configuration.
8. Whether the BARS adapter matches official evaluation.

If a method is supposed to train for 1M gradient steps and the current experiment trained for 10k, the result is only smoke evidence. It cannot be used to conclude that the algorithm, the task, or the environment has a scientific failure mode.

This rule is the highest-priority rule in the research loop.

---

## 2. BARS research positioning

BARS is not simply "GAS + a reachability penalty." BARS is an execution-aware offline stitching framework whose intended components are:

```text
Component A: Bottleneck-temporal subgoal graph
Component B: Bottom-policy reachability edges
Component C: Boundary-consistent path search
Component D: Edge/path-level false-positive diagnostics
```

BARS should only make claims after strong baseline and adapter certification.

The intended scientific questions are:

```text
Q1. Does the baseline graph planner have path-level false-positive stitching under a certified strong backbone?
Q2. Are there executable and path-useful bridges that the baseline fails to exploit?
Q3. Can pπ / p_bridge predict edge execution success?
Q4. Can boundary consistency reduce multi-edge path failures without collapsing coverage?
Q5. Can BARS improve same-backbone planning without relying on fallback recovery?
```

---

## 3. Non-negotiable constraints

These constraints apply to every autonomous round.

```text
1. Baseline certification must precede failure interpretation.
2. Adapter certification must precede same-backbone method comparison.
3. Reduced training, reduced episodes, and truncated hyperparameter runs are E0_SMOKE_ONLY.
4. If exact public target is unavailable, label the cell HOLD_NO_EXACT_PUBLIC_TARGET.
5. If official artifact is unavailable and full-budget training is not run, label the cell SKIP_ARTIFACT_UNAVAILABLE.
6. Do not infer scientific failure modes from uncertified baselines.
7. Planner evidence must use fallback_mode=none.
8. direct-goal / progress fallback can be studied only as a fallback protocol, not as planner evidence.
9. Do not train p_bridge until PASS_ORACLE_HEADROOM.
10. Do not run integrated BARS-v3 until PASS_ORACLE_HEADROOM && PASS_P_BRIDGE.
11. Boundary is diagnostic-only until coverage >= 0.05, supported_gap >= 0.10, and psi_AUROC >= 0.65.
12. D4RL is protocol/debug only until PASS_D4RL_PROTOCOL_AUDIT.
13. WANDB must be disabled by default.
14. Every result package must include configs, commands, reports, failed jobs, gate status, and a reflection.
15. Every claim must be tied to a gate, evidence class, and report file.
```

---

## 4. Evidence classes

Every experiment must be classified.

| Class | Name | Definition | Allowed use |
|---|---|---|---|
| E0 | `E0_SMOKE_ONLY` | compile test, tiny episode count, reduced training, temporary hyperparameters, partial artifacts, 10k/100k when public method needs full budget | Code sanity only |
| E1 | `E1_BASELINE_REGISTRY` | public-target lookup, hyperparameter lookup, official artifact availability check | Planning and certification setup |
| E2 | `E2_CERTIFIED_BASELINE_DIAGNOSTIC` | official artifact or full-budget public-quality baseline passes certification | Failure diagnostics allowed |
| E3 | `E3_SAME_BACKBONE_METHOD_COMPARISON` | BARS variant and certified baseline share low-level policy, TDR, graph, env, goals, max steps, and eval loop | Mechanism claim allowed |
| E4 | `E4_FULL_BUDGET_TRAINED_METHOD` | full training budget and public-quality hyperparameters for BARS or baseline variants | Final performance claim allowed |

Default rule:

```text
If evidence class is missing, treat it as E0_SMOKE_ONLY.
```

---

## 5. Baseline certification gate

### 5.1 Baseline card schema

Codex must create one baseline card per environment and algorithm.

```json
{
  "env": "antmaze-medium-stitch-v0",
  "suite": "ogbench",
  "algorithm": "GAS",
  "baseline_role": "primary_strong_backbone",
  "exact_public_target_available": true,
  "public_source": "paper/project/github/benchmark",
  "public_metric": "success_rate_or_normalized_return",
  "public_mean": 0.981,
  "public_std": 0.012,
  "public_eval_protocol": {
    "num_task_goals": null,
    "rollouts_per_goal": null,
    "num_seeds": null,
    "max_episode_steps": null,
    "success_threshold": null,
    "goal_sampling": null
  },
  "required_train_steps": null,
  "required_batch_size": null,
  "required_hyperparameters": {},
  "official_checkpoint_available": null,
  "official_tdr_available": null,
  "official_graph_available": null,
  "we_used": {
    "source": "official_checkpoint | full_budget_train | reduced_train | unknown",
    "train_steps": null,
    "checkpoint_path": null,
    "tdr_path": null,
    "graph_path": null,
    "policy_path": null
  },
  "official_eval_score": null,
  "bars_adapter_score": null,
  "adapter_gap_pp": null,
  "lower_bound": null,
  "certification_status": "UNASSESSED"
}
```

### 5.2 Public reproduction lower bound

For a public target with mean and standard deviation:

```text
lower_bound = public_mean - max(2 * public_std, 0.05)
```

For percentage reports, use percentage points:

```text
lower_bound_pp = public_mean_pp - max(2 * public_std_pp, 5pp)
```

Example:

```text
public_mean = 98.1%
public_std = 1.2%
lower_bound = 98.1 - max(2.4, 5.0) = 93.1%
```

### 5.3 `PASS_BASELINE_CERTIFICATION`

A cell passes only if:

```text
exact_public_target_available == true
and evidence_class in {E2_CERTIFIED_BASELINE_DIAGNOSTIC, E4_FULL_BUDGET_TRAINED_METHOD}
and official_eval_score >= lower_bound
and no protocol mismatch is detected
```

If no exact public target exists:

```text
HOLD_NO_EXACT_PUBLIC_TARGET
```

If artifact is missing and full-budget training was not run:

```text
SKIP_ARTIFACT_UNAVAILABLE
```

If full-budget training was required but not completed:

```text
FAIL_UNDERTRAINED_BASELINE
```

If score is below lower bound under official evaluation:

```text
FAIL_PUBLIC_REPRODUCTION
```

---

## 6. Adapter certification gate

If BARS uses any adapter, wrapper, or custom evaluation loop, it must be compared against official evaluation.

### 6.1 Required comparison

For the same environment, task IDs, goals, seed, policy, TDR, and graph:

```text
official_eval_score = official script score
bars_adapter_score = BARS adapter score
adapter_gap_pp = bars_adapter_score - official_eval_score
```

### 6.2 `PASS_ADAPTER_CERTIFICATION`

The adapter passes only if:

```text
abs(adapter_gap_pp) <= 2pp
same_task_ids == true
same_goal_sampling == true
same_success_threshold == true
same_max_episode_steps == true
same_policy_checkpoint == true
same_tdr_checkpoint == true
same_graph_checkpoint == true
```

If official eval is strong but adapter is weak:

```text
FAIL_ADAPTER_CERTIFICATION
```

This outcome means the current problem is protocol/adapter mismatch, not a scientific failure of the algorithm.

---

## 7. Prior evidence reclassification

All previous BARS results must be reclassified under the evidence class system.

Required output:

```text
reports/stageXX_prior_evidence_reclassification.csv
reports/stageXX_prior_evidence_reclassification.md
```

Required columns:

```text
stage
run_root
env
algorithm_or_variant
condition
fallback_mode
train_steps
official_artifact_used
public_target_available
baseline_certification_status
adapter_certification_status
evidence_class
allowed_claim_level
downgrade_reason
```

Allowed claim levels:

```text
SMOKE_ONLY
PROTOCOL_DEBUG_ONLY
CERTIFIED_DIAGNOSTIC
SAME_BACKBONE_MECHANISM
FULL_PERFORMANCE
```

Rules:

```text
1. Any result with reduced training is SMOKE_ONLY.
2. Any result with uncertified baseline is PROTOCOL_DEBUG_ONLY.
3. Any result using direct-goal/progress fallback is not planner evidence.
4. Any result with adapter gap > 2pp is protocol evidence only.
5. Any method comparison without same backbone is not a mechanism claim.
```

---

## 8. Failure taxonomy after baseline certification

Failure taxonomy must not be treated as causal before baseline and adapter certification.

Once certified, failure analysis must split:

```text
causal_primary_failure_type
symptom_labels
chronological_first_failure_event
```

### 8.1 Label specificity gate

`PASS_LABEL_SPECIFICITY` requires:

```text
mean_labels_per_failed_episode <= 2.5
dominant_primary_label_share <= 0.85 unless justified by chronology
F4_LOCAL_EXECUTION_DRIFT can be primary only if local_drift_score_AUC_higher_failure >= 0.65
primary_failure_type_nan_rate == 0
unclassified_failure_rate <= 0.20
```

If label coverage is complete but labels are saturated:

```text
PASS_LABEL_INTEGRITY
FAIL_LABEL_SPECIFICITY
```

---

## 9. Oracle-headroom gate

Do not train p_bridge or integrated BARS until oracle headroom is established on a certified baseline.

Required metrics:

```text
artifact_available
set_state_rate
reset_label_source
safe_local_success_rate
risky_bridge_success_rate
oracle_bridge_count
oracle_bridge_usage_rate
oracle_shorter_path_rate
oracle_mean_path_cost_reduction
useful_bridge_score = oracle_bridge_usage_rate * oracle_mean_path_cost_reduction
```

`PASS_ORACLE_HEADROOM` requires:

```text
PASS_BASELINE_CERTIFICATION
PASS_ADAPTER_CERTIFICATION
artifact_available == true
(set_state_rate >= 0.95 or reset_label_source == RELIABLE_ALTERNATIVE)
safe_local_success_rate >= 0.85
oracle_bridge_count >= 50
oracle_bridge_usage_rate >= 0.20
(oracle_shorter_path_rate >= 0.20 or oracle_mean_path_cost_reduction >= 1.0)
useful_bridge_score >= 0.20
```

If executable bridges exist but do not affect path search:

```text
NO_PATH_USEFUL_ORACLE_BRIDGES
```

If reset/set_state labels are unreliable:

```text
NO_RELIABLE_ORACLE_LABELS
```

---

## 10. p_bridge gate

Only after `PASS_ORACLE_HEADROOM`.

`PASS_P_BRIDGE` requires:

```text
selected_bridge_AUROC >= 0.70
accepted_coverage >= 0.30
FP_relative_reduction >= 0.20
accepted_success_uplift >= 0.10
calibration_ECE <= 0.10 if calibration is reported
```

If p_bridge has AUROC but collapses coverage:

```text
HOLD_P_BRIDGE_COVERAGE_COLLAPSE
```

If p_bridge cannot distinguish executable vs false bridges:

```text
FAIL_P_BRIDGE_DISCRIMINATION
```

---

## 11. Boundary gate

Boundary remains diagnostic-only until:

```text
coverage >= 0.05
supported_success_rate - unsupported_success_rate >= 0.10
psi_AUROC >= 0.65
```

Before this gate, do not integrate boundary into the main planner.

---

## 12. Integrated BARS gate

Integrated BARS-v3 may run only if:

```text
PASS_BASELINE_CERTIFICATION
PASS_ADAPTER_CERTIFICATION
PASS_ORACLE_HEADROOM
PASS_P_BRIDGE
fallback_mode=none
same backbone
paired comparison
```

Boundary may be optional for early integration, but if boundary is used, it must pass the boundary gate.

---

## 13. Persistent state

Create and maintain:

```text
research_state/
  bars_research_state.json
  hypothesis_ledger.jsonl
  experiment_ledger.jsonl
  decision_ledger.jsonl
  baseline_registry.jsonl
  baseline_cards/
  current_best_method.json
  open_blockers.json
```

### 13.1 `bars_research_state.json`

```json
{
  "current_round": 0,
  "global_status": "INITIALIZED",
  "baseline_first_mode": true,
  "certified_envs": [],
  "blocked_envs": [],
  "current_best_method": "official_control_GAS_baseline",
  "active_primary_question": null,
  "active_secondary_questions": [],
  "last_package": null
}
```

### 13.2 `hypothesis_ledger.jsonl`

Each line:

```json
{
  "round": 2,
  "hypothesis_id": "H_BASELINE_MEDIUM_STITCH",
  "hypothesis": "BARS adapter can reproduce official GAS on antmaze-medium-stitch-v0 within 2pp.",
  "status": "OPEN",
  "evidence_required": ["PASS_BASELINE_CERTIFICATION", "PASS_ADAPTER_CERTIFICATION"],
  "result": null
}
```

### 13.3 `experiment_ledger.jsonl`

Each line:

```json
{
  "round": 2,
  "experiment_id": "E_CERT_GAS_MEDIUM_STITCH",
  "env": "antmaze-medium-stitch-v0",
  "algorithm": "GAS",
  "evidence_class": "E2_CERTIFIED_BASELINE_DIAGNOSTIC",
  "commands": [],
  "outputs": [],
  "status": "PLANNED"
}
```

### 13.4 `decision_ledger.jsonl`

Each line:

```json
{
  "round": 2,
  "decision": "HOLD_SCIENTIFIC_INTERPRETATION_BASELINE_UNCERTIFIED",
  "reason": "Official target exists but adapter gap exceeded 2pp.",
  "next_action": "Adapter protocol audit"
}
```

---

## 14. Per-round lifecycle

Each autonomous round must follow this order.

```text
Phase 0: Preflight
Phase 1: Read previous state
Phase 2: Identify primary question
Phase 3: Baseline registry / public target lookup
Phase 4: Baseline certification or artifact/full-budget decision
Phase 5: Adapter certification
Phase 6: Evidence reclassification
Phase 7: Only if allowed: failure diagnostics / oracle scan / method experiment
Phase 8: Gate analysis
Phase 9: Reflection
Phase 10: Next-round plan
Phase 11: Package
```

If Phase 3-6 fail, do not proceed to Phase 7 except for protocol/debug work.

---

## 15. Preflight

Run:

```bash
export PYTHONPATH=$PWD
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export D4RL_SUPPRESS_IMPORT_ERROR=1

python --version
git status --short || true
git rev-parse HEAD || true
python -m compileall bars scripts
```

Record:

```text
reports/round_XXX_preflight.md
```

Include:

```text
pwd
git commit
git status
python version
CUDA/JAX/PyTorch versions if relevant
available GPUs
directory summaries
existing Stage reports
existing artifacts/checkpoints
```

---

## 16. Baseline registry implementation

Create or update:

```text
scripts/build_baseline_registry.py
scripts/fetch_public_baseline_targets.py
scripts/verify_baseline_card.py
```

Required output:

```text
reports/round_XXX_baseline_registry.csv
reports/round_XXX_baseline_cards.jsonl
reports/round_XXX_public_target_lookup.md
research_state/baseline_registry.jsonl
research_state/baseline_cards/<env>__<algorithm>.json
```

Minimum target algorithms:

```text
GAS
HIQL
GCIQL
QRL
GCBC
OGBench reference methods where available
```

Priority environments:

```text
antmaze-medium-stitch-v0
antmaze-medium-navigate-v0
antmaze-large-stitch-v0
antmaze-large-navigate-v0
antmaze-giant-stitch-v0
antmaze-giant-navigate-v0
antmaze-large-explore-v0
scene-play-v0
visual-scene-play-v0
visual-antmaze-large-explore-v0
```

If exact public targets are unavailable, use nearest related environments but label clearly:

```text
HOLD_NO_EXACT_PUBLIC_TARGET
```

Do not substitute a related environment as if it were exact.

---

## 17. GAS baseline certification

Create or update:

```text
scripts/certify_gas_baseline.sh
scripts/run_official_gas_eval.py
scripts/compare_official_vs_bars_adapter.py
scripts/analyze_baseline_certification.py
```

Required outputs:

```text
reports/round_XXX_gas_official_eval.csv
reports/round_XXX_bars_adapter_eval.csv
reports/round_XXX_official_vs_adapter.csv
reports/round_XXX_baseline_certification.json
reports/round_XXX_baseline_certification.md
```

### 17.1 Official artifact path

Prefer official artifacts/checkpoints. If available:

```text
evidence_class = E2_CERTIFIED_BASELINE_DIAGNOSTIC
```

If official artifacts are missing:

```text
SKIP_ARTIFACT_UNAVAILABLE
```

Do not replace with 10k/100k training and claim certification.

### 17.2 Full-budget training path

If full-budget training is required:

```text
evidence_class = E4_FULL_BUDGET_TRAINED_METHOD
```

Training must use public-quality hyperparameters. If the public method trains for 1M gradient steps, run 1M. If compute budget is insufficient, mark skipped/blocked.

Reduced run:

```text
E0_SMOKE_ONLY
```

### 17.3 Medium-stitch sanity target

For `antmaze-medium-stitch-v0`, public tables report very high GAS-level performance. Treat this environment primarily as:

```text
adapter certification
baseline reproduction sanity check
same-backbone no-regression check
```

Do not use low internal scores on this environment as evidence that the environment is unsolved unless the baseline and adapter are certified.

---

## 18. Adapter audit checklist

For each environment:

```text
same_env_id
same_task_id_list
same_goal_sampling
same_start_sampling
same_observation_format
same_goal_format
same_success_source
same_success_threshold
same_max_episode_steps
same_policy_checkpoint
same_tdr_checkpoint
same_graph_checkpoint
same_graph_node_count
same_graph_edge_count
same_nearest_backend
same_action_scaling
same_eval_seed
same_episode_count
```

Any mismatch must be written to:

```text
reports/round_XXX_adapter_mismatch_report.md
```

---

## 19. Failure analysis is allowed only after certification

Allowed only if:

```text
PASS_BASELINE_CERTIFICATION
PASS_ADAPTER_CERTIFICATION
```

Then create:

```text
scripts/enrich_failure_atlas_certified.py
scripts/audit_failure_label_specificity.py
```

Output:

```text
reports/round_XXX_failure_atlas_certified.csv
reports/round_XXX_failure_label_specificity.csv
reports/round_XXX_failure_label_specificity.json
reports/round_XXX_failure_taxonomy_audit.md
```

If certification fails, failure analysis may run but must be marked:

```text
PROTOCOL_DEBUG_ONLY
```

---

## 20. Oracle-headroom scan is allowed only after certification

Allowed only if:

```text
PASS_BASELINE_CERTIFICATION
PASS_ADAPTER_CERTIFICATION
```

Then create or run:

```text
scripts/oracle_headroom_scan_v2.sh
scripts/audit_set_state.py
scripts/useful_bridge_analysis.py
```

Output:

```text
reports/round_XXX_set_state_audit.csv
reports/round_XXX_oracle_headroom_v2.csv
reports/round_XXX_useful_bridge_summary.csv
```

Do not run p_bridge in the same round unless explicitly permitted by the round plan and `PASS_ORACLE_HEADROOM` has already been achieved from a completed analysis.

---

## 21. Method experiments are gated

### 21.1 Reachability-supported GAS

Allowed only after:

```text
PASS_BASELINE_CERTIFICATION
PASS_ADAPTER_CERTIFICATION
```

Claim allowed only if:

```text
same_backbone == true
fallback_mode == none
mean_delta >= +2pp
paired_cell_wins >= 4/6
steps_inflation <= 10%
```

### 21.2 p_bridge

Allowed only after:

```text
PASS_ORACLE_HEADROOM
```

### 21.3 Boundary

Allowed only as diagnostic until `PASS_BOUNDARY_REENTRY`.

### 21.4 Integrated BARS-v3

Allowed only after:

```text
PASS_ORACLE_HEADROOM
PASS_P_BRIDGE
```

---

## 22. Central gate schema

Every round must produce:

```text
reports/round_XXX_gate_status.json
```

Schema:

```json
{
  "round": 2,
  "primary_question": "...",
  "baseline_registry": "PASS_BASELINE_REGISTRY | HOLD_NO_EXACT_PUBLIC_TARGET | FAIL_BASELINE_REGISTRY",
  "baseline_certification": "PASS_BASELINE_CERTIFICATION | FAIL_PUBLIC_REPRODUCTION | FAIL_UNDERTRAINED_BASELINE | SKIP_ARTIFACT_UNAVAILABLE | HOLD_NO_EXACT_PUBLIC_TARGET",
  "adapter_certification": "PASS_ADAPTER_CERTIFICATION | FAIL_ADAPTER_CERTIFICATION | SKIP_NO_OFFICIAL_EVAL",
  "evidence_reclassification": "PASS_EVIDENCE_RECLASSIFICATION | FAIL_EVIDENCE_RECLASSIFICATION",
  "label_specificity": "PASS_LABEL_SPECIFICITY | FAIL_LABEL_SPECIFICITY | SKIP_BASELINE_UNCERTIFIED",
  "oracle_headroom": "PASS_ORACLE_HEADROOM | NO_PATH_USEFUL_ORACLE_BRIDGES | NO_RELIABLE_ORACLE_LABELS | SKIP_BASELINE_UNCERTIFIED | SKIP_ARTIFACT_UNAVAILABLE",
  "p_bridge": "READY_FOR_P_BRIDGE | SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM | PASS_P_BRIDGE | FAIL_P_BRIDGE",
  "boundary": "HOLD_BOUNDARY_DIAGNOSTIC_ONLY | PASS_BOUNDARY_REENTRY",
  "integrated": "READY_FOR_INTEGRATED_NO_FALLBACK | SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE",
  "d4rl_protocol": "PASS_D4RL_PROTOCOL_AUDIT | HOLD_D4RL_PROTOCOL_REPAIR | FAIL_D4RL_ADAPTER_MISMATCH",
  "scientific_interpretation": "ALLOW_CERTIFIED_DIAGNOSTIC | HOLD_SCIENTIFIC_INTERPRETATION_BASELINE_UNCERTIFIED",
  "details": {}
}
```

---

## 23. Reflection format

Each round must write:

```text
rounds/round_XXX/reflection.md
```

Required sections:

```md
# Round XXX Reflection

## Primary question

## What was executed

## Completed jobs

## Failed jobs

## Baseline certification status

## Adapter certification status

## Evidence class summary

## Results

## What can be concluded

## What cannot be concluded

## New blockers

## Updated hypotheses

## Next round decision
```

The "What cannot be concluded" section is mandatory.

---

## 24. Next-plan format

Each round must write:

```text
rounds/round_XXX/next_plan.md
```

Required sections:

```md
# Round XXX Next Plan

## Decision

## Primary question for next round

## Gates that unlock this question

## Experiments to run

## Commands

## Expected outcomes

## Stop conditions
```

---

## 25. Packaging

Create or update:

```text
scripts/package_autoresearch_round.sh
```

Package:

```text
packages/bars_autoresearch_round_XXX.tar.gz
```

Include:

```text
configs/
scripts/
reports/round_XXX_*
rounds/round_XXX/
research_state/
commands/round_XXX_commands.sh
logs/round_XXX_failed_jobs.csv
logs/round_XXX_jobs.tsv
code/round_XXX_code_diff.patch
MANIFEST.txt
CHECKSUMS.sha256
```

Exclude:

```text
large checkpoints
raw datasets
wandb
cache
huge debug traces
```

---

## 26. Stop conditions

The loop may stop only with one of:

```text
STOP_SUCCESS_METHOD_FOUND
STOP_KEY_SCIENTIFIC_OBSTRUCTION_IDENTIFIED
STOP_COMPUTE_BUDGET_EXHAUSTED
STOP_PROTOCOL_BLOCKER_UNRESOLVED
STOP_REPO_OR_ARTIFACT_MISSING
STOP_MAX_ROUNDS_REACHED
```

A stop condition must include evidence and a written conclusion.

---

## 27. Round selection policy

After each round:

### If baseline certification fails

Next round:

```text
baseline/protocol repair
```

Do not run method experiments.

### If baseline passes but adapter fails

Next round:

```text
adapter audit and repair
```

Do not run method experiments.

### If baseline and adapter pass

Next round:

```text
certified failure diagnostics + oracle-headroom scan
```

### If oracle headroom passes

Next round:

```text
p_bridge training/evaluation on the passing env only
```

### If p_bridge passes

Next round:

```text
same-backbone no-fallback integrated BARS-v3
```

### If no natural environment has oracle headroom

Next round:

```text
controlled stitching stress-test benchmark design
```

### If label specificity fails

Next round:

```text
failure taxonomy repair
```

---

## 28. Active Round 002 prompt: baseline certification first

Paste this prompt into Codex after placing this file in the repository and creating the root `AGENTS.md` shim.

```text
Run BARS autonomous research Round 002 in baseline-first mode.

Primary question:
Can we certify a public-quality strong baseline and BARS adapter before interpreting any BARS failure modes?

Secondary question:
Which prior Stage19–Round001 results must be downgraded to smoke/protocol evidence because baseline certification, adapter certification, or full-budget training was missing?

Non-negotiable:
- Do not run p_bridge.
- Do not run integrated BARS-v3.
- Do not run oracle-headroom as scientific evidence until baseline and adapter certification pass.
- Do not interpret failure taxonomy as causal unless baseline and adapter are certified.
- Reduced training is E0_SMOKE_ONLY.
- Planner evidence must use fallback_mode=none.
- Any direct-goal/progress fallback result is fallback-protocol evidence, not planner evidence.

Tasks:

1. Implement or update:
   - scripts/autoresearch_init.py
   - scripts/autoresearch_run_round.py
   - scripts/autoresearch_analyze_round.py
   - scripts/autoresearch_plan_next.py
   - scripts/build_baseline_registry.py
   - scripts/fetch_public_baseline_targets.py
   - scripts/verify_baseline_card.py
   - scripts/certify_gas_baseline.sh
   - scripts/run_official_gas_eval.py
   - scripts/compare_official_vs_bars_adapter.py
   - scripts/reclassify_prior_evidence.py
   - scripts/analyze_baseline_certification.py
   - scripts/package_autoresearch_round.sh

2. Create persistent state if missing:
   - research_state/bars_research_state.json
   - research_state/hypothesis_ledger.jsonl
   - research_state/experiment_ledger.jsonl
   - research_state/decision_ledger.jsonl
   - research_state/baseline_registry.jsonl
   - research_state/current_best_method.json
   - research_state/open_blockers.json

3. Build baseline registry for:
   - antmaze-medium-stitch-v0
   - antmaze-medium-navigate-v0
   - antmaze-large-stitch-v0
   - antmaze-large-navigate-v0
   - antmaze-giant-stitch-v0
   - antmaze-giant-navigate-v0
   - antmaze-large-explore-v0
   - scene-play-v0

   Algorithms:
   - GAS
   - HIQL
   - GCIQL
   - QRL
   - GCBC
   - OGBench reference methods where available

4. For each env/algorithm card, record:
   - exact public target available?
   - public mean/std
   - public metric
   - public eval protocol
   - required train steps
   - required hyperparameters
   - official checkpoint/artifact availability
   - lower bound
   - certification status

5. Certify GAS first on:
   - antmaze-medium-stitch-v0
   - antmaze-medium-navigate-v0

   Prefer official checkpoints/artifacts if available.
   If official artifacts are unavailable, do not replace with reduced training.
   If full-budget training is required and compute is not available, mark SKIP_ARTIFACT_UNAVAILABLE or FAIL_UNDERTRAINED_BASELINE.

6. If official GAS evaluation can be run, compare:
   - official evaluate_gas.py score
   - BARS adapter score

   Requirements:
   - same env ID
   - same task IDs
   - same goal sampling
   - same success threshold
   - same max episode steps
   - same policy checkpoint
   - same TDR checkpoint
   - same graph checkpoint

   PASS_ADAPTER_CERTIFICATION only if abs(adapter_gap_pp) <= 2pp.

7. Reclassify prior evidence from Stage19–Round001:
   - mark reduced training as E0_SMOKE_ONLY
   - mark uncertified baseline results as PROTOCOL_DEBUG_ONLY
   - mark fallback runs as fallback-protocol evidence, not planner evidence
   - mark same-backbone comparisons as valid only if backbone and adapter certification are proven

8. Produce reports:
   - reports/round_002_preflight.md
   - reports/round_002_baseline_registry.csv
   - reports/round_002_baseline_cards.jsonl
   - reports/round_002_public_target_lookup.md
   - reports/round_002_gas_official_eval.csv
   - reports/round_002_bars_adapter_eval.csv
   - reports/round_002_official_vs_adapter.csv
   - reports/round_002_baseline_certification.json
   - reports/round_002_baseline_certification.md
   - reports/round_002_prior_evidence_reclassification.csv
   - reports/round_002_prior_evidence_reclassification.md
   - reports/round_002_gate_status.json
   - rounds/round_002/reflection.md
   - rounds/round_002/next_plan.md
   - packages/bars_autoresearch_round_002.tar.gz

9. Gate decisions:
   - If PASS_BASELINE_CERTIFICATION and PASS_ADAPTER_CERTIFICATION:
       Round 003 = certified failure diagnostics + oracle-headroom scan.
   - If baseline passes but adapter fails:
       Round 003 = adapter repair.
   - If official artifacts unavailable and full-budget training not completed:
       Round 003 = artifact acquisition or full-budget training plan.
   - If public target lookup fails:
       Round 003 = public target / benchmark protocol audit.
   - If all prior results are downgraded:
       say so explicitly and explain which claims remain valid.

10. Final answer from Codex must print:
   ROUND
   PRIMARY_QUESTION
   BASELINE_REGISTRY_STATUS
   BASELINE_CERTIFICATION_STATUS
   ADAPTER_CERTIFICATION_STATUS
   PRIOR_EVIDENCE_RECLASSIFICATION
   SCIENTIFIC_INTERPRETATION_STATUS
   NEXT_ROUND
   PACKAGE
```

---

## 29. Round 002 expected conclusions

Round 002 is not designed to discover a new BARS method. It is designed to answer whether previous and future experiments are scientifically interpretable.

### Conclusion A: baseline and adapter pass

If:

```text
PASS_BASELINE_CERTIFICATION
PASS_ADAPTER_CERTIFICATION
```

Then conclude:

```text
We now have a certified strong GAS backbone on the selected environments.
Future failure analysis and same-backbone BARS ablations can be interpreted as scientific diagnostic evidence.
Round 003 should run certified failure taxonomy + oracle-headroom scan.
```

### Conclusion B: baseline passes but adapter fails

If:

```text
PASS_BASELINE_CERTIFICATION
FAIL_ADAPTER_CERTIFICATION
```

Then conclude:

```text
The official algorithm is strong, but the BARS adapter/evaluation loop is not aligned.
Prior BARS adapter results are protocol/debug evidence only.
Round 003 should repair adapter/eval protocol.
```

### Conclusion C: artifact unavailable / no full-budget training

If:

```text
SKIP_ARTIFACT_UNAVAILABLE
or FAIL_UNDERTRAINED_BASELINE
```

Then conclude:

```text
We cannot currently make scientific claims on this environment.
Low scores from reduced training or incomplete artifacts do not imply the environment is unsolved or that BARS has found a real failure mode.
Round 003 should acquire official artifacts or run full-budget training.
```

### Conclusion D: no exact public target

If:

```text
HOLD_NO_EXACT_PUBLIC_TARGET
```

Then conclude:

```text
The environment lacks an exact public reference target.
Use it for exploratory work only until a reliable target is established or a nearest-neighbor target is explicitly justified.
```

---

## 30. Minimal command skeleton

Codex should adapt these commands to the actual repository.

```bash
export PYTHONPATH=$PWD
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export D4RL_SUPPRESS_IMPORT_ERROR=1

python -m compileall bars scripts

python scripts/build_baseline_registry.py \
  --envs antmaze-medium-stitch-v0,antmaze-medium-navigate-v0,antmaze-large-stitch-v0,antmaze-large-navigate-v0,antmaze-giant-stitch-v0,antmaze-giant-navigate-v0,antmaze-large-explore-v0,scene-play-v0 \
  --algorithms GAS,HIQL,GCIQL,QRL,GCBC \
  --out reports/round_002_baseline_registry.csv \
  --cards-out reports/round_002_baseline_cards.jsonl \
  --lookup-report reports/round_002_public_target_lookup.md

bash scripts/certify_gas_baseline.sh \
  ENVS=antmaze-medium-stitch-v0,antmaze-medium-navigate-v0 \
  SEEDS=0,1,2,3 \
  EPISODES_PER_GOAL=50 \
  USE_OFFICIAL_ARTIFACTS=1 \
  FULL_BUDGET_ONLY=1 \
  ROUND=002 \
  REPORTS_ROOT=reports

python scripts/compare_official_vs_bars_adapter.py \
  --official reports/round_002_gas_official_eval.csv \
  --adapter reports/round_002_bars_adapter_eval.csv \
  --out reports/round_002_official_vs_adapter.csv \
  --max-gap-pp 2.0

python scripts/reclassify_prior_evidence.py \
  --baseline-cards reports/round_002_baseline_cards.jsonl \
  --stage-reports reports \
  --out reports/round_002_prior_evidence_reclassification.csv \
  --md-out reports/round_002_prior_evidence_reclassification.md

python scripts/analyze_baseline_certification.py \
  --baseline-registry reports/round_002_baseline_registry.csv \
  --baseline-cards reports/round_002_baseline_cards.jsonl \
  --official-vs-adapter reports/round_002_official_vs_adapter.csv \
  --prior-reclass reports/round_002_prior_evidence_reclassification.csv \
  --gate-out reports/round_002_gate_status.json \
  --md-out reports/round_002_baseline_certification.md

bash scripts/package_autoresearch_round.sh \
  ROUND=002 \
  OUT=packages/bars_autoresearch_round_002.tar.gz
```

---

## 31. Final reminder to Codex

Do not optimize for speed at the cost of scientific validity.

The correct result of a round may be:

```text
We cannot interpret prior experiments because the baseline is uncertified.
```

That is a valid research outcome.

The wrong result is:

```text
We ran a reduced training baseline, got low success, and concluded the environment has a failure mode.
```

Every future BARS method claim must stand on a certified baseline, certified adapter, full-budget or official artifacts, and same-backbone comparison.
