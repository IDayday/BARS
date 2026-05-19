# Stage23 Adapter Protocol Repair

```csv
env,seed,official_B_success,adapter_original_success,adapter_official_control_success,original_minus_official_pp,official_control_minus_official_pp,episodes,mean_steps,eval_csv
antmaze-medium-navigate-v0,0,0.9399999999999998,0.86,0.93,-7.999999999999985,-0.9999999999999787,100,295.6,runs_stage23_repro_repair/antmaze-medium-navigate-v0/seed0/gas_shortest_official_control/budget999/fallback_none/eval.csv
antmaze-medium-stitch-v0,0,0.966,0.87,0.98,-9.599999999999998,1.4000000000000012,100,252.14,runs_stage23_repro_repair/antmaze-medium-stitch-v0/seed0/gas_shortest_official_control/budget999/fallback_none/eval.csv
```

- Finding: the original BARS adapter chunked fixed subgoals and lagged official GAS by 8.0-9.6pp on medium tasks.
- Repair: an adapter loop using official GAS per-step shortest-path control is within 1.4pp of official B on both medium envs.
- Decision: treat original adapter loop as a non-official experimental variant; use official-control adapter for reproduction-aligned GAS comparisons.
