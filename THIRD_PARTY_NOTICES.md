# Third-Party Notices

## Graph-Assisted Stitching (GAS)

- Official repository: https://github.com/qortmdgh4141/GAS
- Local source path: `external_src/GAS`
- License: MIT License
- Copyright: retained by the GAS authors as specified in `external_src/GAS/LICENSE`.

Stage22 uses the official GAS implementation as the backbone for Temporal Distance Representation (TDR), keygraph construction, low-level actor policy, and environment/evaluation protocol. BARS Stage22 code calls GAS through adapter modules under `bars/external/`; the pruned GAS source tree is kept under `external_src/GAS`.

On a new machine, run:

```bash
bash scripts/setup_gas_repo.sh
```

This verifies the vendored GAS source and applies the compatibility patch at
`third_party/gas_stage22.patch` when needed. If `external_src/GAS` is absent, the
script clones the official repository first.

Local modifications:

- `external_src/GAS/K_utils/graph_builder.py`: fixed an adapter-level call-site mismatch where `setup_task_env` was called with an extra `dataset` argument. This preserves the official behavior and allows `construct_graph.py` to run with the current GAS `O_utils.evaluation.setup_task_env` signature.
- `external_src/GAS/O_utils/log_utils.py`: added a WandB-compatible TensorBoard shim so Stage22/Stage23 can run with `WANDB_DISABLED=true`.
- `external_src/GAS/{pretrain_tdr.py,train_policy.py,construct_graph.py,evaluate_gas.py}`: changed hard-coded `MUJOCO_GL='egl'` assignments to `setdefault` so server-specific EGL/OSMesa choices are respected.

Files copied or vendored into BARS:

- Pruned GAS source under `external_src/GAS`. BARS adapter code imports and invokes it.
