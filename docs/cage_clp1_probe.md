# CAGE-CLP1 Branchable Probe

The branchable probe estimates raw frozen-policy execution contracts from exact rollout segment starts.

Input:
- segment contract JSONL produced by `evaluate_gas.py --contract_trace_path ... --store_contract_state_refs --contract_state_ref_mode exact_only`
- official GAS checkpoint root
- env/seed

Probe logic:
1. Restore the environment exactly to `start_state_ref`.
2. Select a target mode, such as `original_target`, `recovery_candidate`, or `final_goal`.
3. Run the frozen low-level policy for horizon `H`.
4. Use the same skill computation as GAS evaluation:
   `skill = (target_phi - phi_obs) / ||target_phi - phi_obs||`.
5. Record `R_pi` proxy fields: `hit`, `time_to_hit`, `delta_phi`, `normalized_progress`, `negative_progress`, action norms, and termination flags.

Unsupported target modes are recorded with a failure reason rather than approximated. CLP1 does not fake rollouts from phi-only pairs.

Example:

```bash
python scripts/probe_branchable_contracts.py \
  --segments_path results/cage_clp1/segment_capture/antmaze-giant-navigate-v0_gas_seed42_segments.jsonl \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --env_name antmaze-giant-navigate-v0 \
  --seed 42 \
  --num_segments 128 \
  --horizons 16 32 64 \
  --target_modes original_target final_goal recovery_candidate \
  --out_jsonl results/cage_clp1/probes/antmaze_nav_seed42_branchable_probe.jsonl \
  --out_summary results/cage_clp1/probes/antmaze_nav_seed42_branchable_probe_summary.md
```
