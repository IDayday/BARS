# Stage30 Full Official GAS Diagnosis Summary

Status: FULL_OFFICIAL_GAS_DIAGNOSIS_COMPLETE_NO_ALGORITHM_MODIFICATION.

Run:
- Directory: `runs_stage30_official_gas/layered_full_official_gas_diag_active`
- Scope: 4 envs x seeds 44-46 x official auto task IDs x 100 episodes.
- Episode traces: 6000 total, 500 per env/seed.
- Phases completed: instrumentation 12/12, keygraph audit 12/12, exact semantic probe 12/12, nearest execution probe 12/12, taxonomy 12/12, global collector complete.
- Execution policy: official GAS graph/planner/policy/action outputs unchanged, `fallback_mode=none`, `EVAL_ON_CPU=0`.
- Scheduling: all unfinished env/seed jobs launched concurrently with `MAX_JOBS=12`, `GPUS=0,1`, round-robin assignment, `SKIP_EXISTING=1`.

Episode outcomes:

| env_name | episodes | success | no_path | timeout | stuck | divergence | reliable_failed_edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| antmaze-medium-navigate-v0 | 1500 | 0.9787 | 0.0000 | 0.0213 | 0.0107 | 0.0100 | 0.0167 |
| antmaze-medium-stitch-v0 | 1500 | 0.9627 | 0.0000 | 0.0373 | 0.0187 | 0.0160 | 0.0333 |
| antmaze-giant-navigate-v0 | 1500 | 0.8500 | 0.0000 | 0.1507 | 0.0360 | 0.0293 | 0.1013 |
| antmaze-giant-stitch-v0 | 1500 | 0.8980 | 0.0000 | 0.1020 | 0.0273 | 0.0200 | 0.0880 |

Probe evidence:
- exact semantic probe rows: 11520; valid executable samples: 0; `trajectory_semantics_valid`: 0.
- nearest execution probe rows: 11520; valid executable samples: 10759; `trajectory_semantics_valid`: 0 by design.
- nearest execution overall reach: 0.0767.
- nearest first-failed edge reach: 0.1145 over 297 valid samples.
- nearest long-hop reach: 0.0518 over 1391 valid samples.

Taxonomy:

| label | count | rate |
| --- | ---: | ---: |
| UNRESOLVED | 5534 | 0.9223 |
| POLICY_LOCAL_FAILURE | 247 | 0.0412 |
| SUBGOAL_SEQUENCE_DRIFT | 182 | 0.0303 |
| LONG_HOP_FAILURE | 37 | 0.0062 |

Conclusion:
- The full official GAS diagnosis does not identify a stable dominant evidence-backed failure mode.
- `UNRESOLVED=92.23%`, far above the <50% gate.
- The main blocker is diagnostic evidence coverage: official keygraph nodes did not exact-map to dataset states, so exact semantic edge categories cannot validate same/cross/dt labels.
- Nearest execution probes provide execution reach/progress evidence only; they cannot support cross-trajectory or temporal semantic conclusions.
- No algorithm modification is justified under the Stage30 rule.

Next diagnosis-only work:
- Recover official keygraph node provenance from official construction artifacts if available.
- Add official-path subgoal-level trace logging that records final-goal phase failures separately from graph-edge failures.
- Improve first-failed-edge reliability before interpreting edge-type failure modes.
