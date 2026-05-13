# Stage16 graph timing4 decision

Decision: GO_FULL12

Summary:
- Initial warm-start attempt with `ann.use_gpu=true` failed 4/4 at `graph_build/select_nodes_start` due to FAISS GPU `cublas failed (13)`.
- After switching to `ann.backend=auto` with `ann.use_gpu=false` and keeping all graph/boundary budgets unchanged, timing4 completed successfully.
- Completed status: 4/4 `completed_graph_timing`
- Dataset truncation: none observed
- profile_all.csv: present
- graph-related phases: clearly logged in `profile_all.csv`

Graph-related timing (successful CPU-ANN rerun):
- antmaze-medium-play-v2: 5.573 sec
- antmaze-medium-diverse-v2: 5.742 sec
- antmaze-large-play-v2: 5.504 sec
- antmaze-large-diverse-v2: 5.526 sec

Phase-level highlights from profile/graph CSV:
- select_nodes: about 1.53 to 1.57 sec
- build_edges: about 0.76 to 0.83 sec
- build_boundary: about 3.16 to 3.32 sec
- No graph-related phase approached the Stage 1 v2 2-3 hour black-box behavior.
- No single graph phase exceeded the 1.5 hour guardrail.

Rationale:
- This is far below the continue thresholds of 3600 sec for medium and 5400 sec for large.
- CPU ANN/FAISS CPU fallback is stable under concurrent runs while preserving node/support budgets.
- Proceed to full12 with warm-started verified checkpoints/embeddings and CPU ANN.
