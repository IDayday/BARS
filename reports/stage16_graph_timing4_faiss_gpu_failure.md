# Stage16 graph timing4 FAISS GPU failure

Observed on 2026-05-13 during warm-start timing4.

Symptom:
- 4/4 runs failed at `graph_build/select_nodes_start`
- return code `-6`
- stderr shows `Faiss assertion 'err == CUBLAS_STATUS_SUCCESS' failed`
- failing op shape: `(1024, 16) x (2048, 16)` during FAISS GPU GEMM

Action taken:
- Keep node/edge/support budgets unchanged
- Switch Stage 1.6 config from `ann.use_gpu=true` to `ann.use_gpu=false`
- Keep `ann.backend=auto` so FAISS CPU remains available with sklearn fallback
- Rerun timing4 from warm-started verified checkpoints/embeddings
