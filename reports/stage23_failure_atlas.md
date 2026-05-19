# Stage23 Failure Atlas

- Episodes: 200
- Success rate: 0.865
- Failures classified: 27

```csv
env,seed,variant,primary_failure_type,episodes,success,steps,subgoal_reach_rate,no_path_rate,risky_bridge_count
antmaze-medium-navigate-v0,0,gas_shortest,F4_LOCAL_EXECUTION_DRIFT,13,0.0,1000.0,0.1494797649383435,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,F6_LONG_PATH_ACCUMULATION,1,0.0,1000.0,0.4455958549222797,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,SUCCESS,86,1.0,284.4651162790698,0.68507181867679,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,F4_LOCAL_EXECUTION_DRIFT,13,0.0,1000.0,0.23708414613598186,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,SUCCESS,87,1.0,271.6896551724138,0.6489287954561059,0.0,0.0
```

## Readout
- Dominant failure type: F4_LOCAL_EXECUTION_DRIFT (96.3% of failures).
- Failures with risky bridge/path evidence: 0.0%.
- Failures looking like local low-level execution: 96.3%.
- Protocol/goal mismatch share: 0.0%.
