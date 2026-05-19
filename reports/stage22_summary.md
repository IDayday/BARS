# Stage22 Summary

Updated: 2026-05-19 10:50:49

## Completion
```csv
env,seed,variant,budget,fallback_mode,episodes,success,steps,fallback_used,no_path_rate,budget_reject_rate
antmaze-medium-navigate-v0,0,gas_reachability_budget_calibrated,4.277040201695186,none,100,0.92,341.49,0.0,0.0,0.0
antmaze-medium-navigate-v0,0,gas_reachability_budget_calibrated,4.277040201695186,progress_stall_v3,100,0.69,535.03,1.0,0.0,0.0
antmaze-medium-navigate-v0,0,gas_reachability_soft_calibrated,4.277040201695186,none,100,0.9,337.86,0.0,0.0,0.0
antmaze-medium-navigate-v0,0,gas_reachability_soft_calibrated,4.277040201695186,progress_stall_v3,100,0.63,575.21,1.0,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,4.277040201695186,none,100,0.89,353.13,0.0,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,4.277040201695186,progress_stall_v3,100,0.65,544.04,0.99,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget_calibrated,3.0509568029941887,none,100,0.87,349.21,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget_calibrated,3.0509568029941887,progress_stall_v3,100,0.77,475.32,1.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_soft_calibrated,3.0509568029941887,none,100,0.91,327.88,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_soft_calibrated,3.0509568029941887,progress_stall_v3,100,0.73,499.75,1.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,3.0509568029941887,none,100,0.86,368.82,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,3.0509568029941887,progress_stall_v3,100,0.69,512.01,1.0,0.0,0.0
```

## Same-Backbone Comparison
```csv
variant,baseline,paired_n,success_delta,baseline_success,variant_success
gas_reachability_budget_calibrated,gas_shortest,400,0.04,0.7725,0.8125
gas_reachability_soft_calibrated,gas_shortest,400,0.02,0.7725,0.7925
```

## Fallback Attribution
```csv
fallback_mode,fallback_used,episodes,success
none,0,600,0.8916666666666667
progress_stall_v3,0,1,1.0
progress_stall_v3,1,599,0.6928213689482471
```

## Path Diagnostics
```csv
env,variant,budget,fallback_mode,pred_bucket,exec_risk_bucket,first_plan_edges_bucket,episodes,success,no_path_rate
antmaze-medium-navigate-v0,gas_reachability_budget_calibrated,4.277040201695186,none,0-0.1,2-3,6-10,5,0.8,0.0
antmaze-medium-navigate-v0,gas_reachability_budget_calibrated,4.277040201695186,none,0-0.1,3-5,6-10,15,0.8,0.0
antmaze-medium-navigate-v0,gas_reachability_budget_calibrated,4.277040201695186,none,0-0.1,3-5,11-20,80,0.95,0.0
antmaze-medium-navigate-v0,gas_reachability_budget_calibrated,4.277040201695186,progress_stall_v3,0-0.1,2-3,6-10,3,0.6666666666666666,0.0
antmaze-medium-navigate-v0,gas_reachability_budget_calibrated,4.277040201695186,progress_stall_v3,0-0.1,3-5,6-10,17,0.17647058823529413,0.0
antmaze-medium-navigate-v0,gas_reachability_budget_calibrated,4.277040201695186,progress_stall_v3,0-0.1,3-5,11-20,80,0.8,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,none,0-0.1,2-3,6-10,8,0.75,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,none,0-0.1,2-3,11-20,12,0.9166666666666666,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,none,0-0.1,3-5,6-10,10,0.8,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,none,0-0.1,3-5,11-20,61,0.9180327868852459,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,none,0.1-0.25,2-3,11-20,9,1.0,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,progress_stall_v3,0-0.1,2-3,6-10,2,0.5,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,progress_stall_v3,0-0.1,2-3,11-20,23,0.6521739130434783,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,progress_stall_v3,0-0.1,3-5,6-10,13,0.23076923076923078,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,progress_stall_v3,0-0.1,3-5,11-20,57,0.7368421052631579,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,progress_stall_v3,0.1-0.25,2-3,11-20,4,0.5,0.0
antmaze-medium-navigate-v0,gas_reachability_soft_calibrated,4.277040201695186,progress_stall_v3,0.25-0.5,1-2,6-10,1,0.0,0.0
antmaze-medium-navigate-v0,gas_shortest,4.277040201695186,none,0-0.1,2-3,6-10,5,0.8,0.0
antmaze-medium-navigate-v0,gas_shortest,4.277040201695186,none,0-0.1,3-5,6-10,15,0.7333333333333333,0.0
antmaze-medium-navigate-v0,gas_shortest,4.277040201695186,none,0-0.1,3-5,11-20,72,0.9305555555555556,0.0
antmaze-medium-navigate-v0,gas_shortest,4.277040201695186,none,0-0.1,>5,11-20,8,0.875,0.0
antmaze-medium-navigate-v0,gas_shortest,4.277040201695186,progress_stall_v3,0-0.1,2-3,6-10,5,0.2,0.0
antmaze-medium-navigate-v0,gas_shortest,4.277040201695186,progress_stall_v3,0-0.1,2-3,11-20,1,1.0,0.0
antmaze-medium-navigate-v0,gas_shortest,4.277040201695186,progress_stall_v3,0-0.1,3-5,6-10,14,0.21428571428571427,0.0
antmaze-medium-navigate-v0,gas_shortest,4.277040201695186,progress_stall_v3,0-0.1,3-5,11-20,71,0.8028169014084507,0.0
antmaze-medium-navigate-v0,gas_shortest,4.277040201695186,progress_stall_v3,0-0.1,>5,11-20,9,0.3333333333333333,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,none,0-0.1,2-3,6-10,11,0.8181818181818182,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,none,0-0.1,2-3,11-20,55,0.8545454545454545,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,none,0-0.1,3-5,11-20,23,0.8695652173913043,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,none,0.1-0.25,1-2,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,none,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,none,0.1-0.25,2-3,11-20,9,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,progress_stall_v3,0-0.1,2-3,6-10,12,0.8333333333333334,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,progress_stall_v3,0-0.1,2-3,11-20,56,0.8035714285714286,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,progress_stall_v3,0-0.1,3-5,6-10,1,0.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,progress_stall_v3,0-0.1,3-5,11-20,21,0.7142857142857143,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,progress_stall_v3,0.1-0.25,1-2,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,progress_stall_v3,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget_calibrated,3.0509568029941887,progress_stall_v3,0.1-0.25,2-3,11-20,8,0.625,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,none,0-0.1,2-3,6-10,6,0.8333333333333334,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,none,0-0.1,2-3,11-20,27,0.8518518518518519,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,none,0-0.1,3-5,11-20,52,0.9423076923076923,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,none,0-0.1,>5,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,none,0.1-0.25,1-2,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,none,0.1-0.25,1-2,11-20,3,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,none,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,none,0.1-0.25,2-3,11-20,9,0.8888888888888888,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,progress_stall_v3,0-0.1,2-3,6-10,6,0.6666666666666666,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,progress_stall_v3,0-0.1,2-3,11-20,18,0.6666666666666666,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,progress_stall_v3,0-0.1,3-5,6-10,1,0.0,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,progress_stall_v3,0-0.1,3-5,11-20,55,0.7272727272727273,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,progress_stall_v3,0.1-0.25,1-2,11-20,6,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,progress_stall_v3,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_soft_calibrated,3.0509568029941887,progress_stall_v3,0.1-0.25,2-3,11-20,13,0.7692307692307693,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,none,0-0.1,2-3,6-10,13,0.8461538461538461,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,none,0-0.1,2-3,11-20,2,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,none,0-0.1,3-5,6-10,2,0.5,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,none,0-0.1,3-5,11-20,69,0.9130434782608695,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,none,0-0.1,>5,11-20,11,0.5454545454545454,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,none,0.1-0.25,1-2,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,none,0.1-0.25,2-3,11-20,2,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,progress_stall_v3,0-0.1,2-3,6-10,12,0.5833333333333334,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,progress_stall_v3,0-0.1,2-3,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,progress_stall_v3,0-0.1,3-5,6-10,2,0.5,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,progress_stall_v3,0-0.1,3-5,11-20,67,0.7910447761194029,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,progress_stall_v3,0-0.1,>5,11-20,13,0.15384615384615385,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,progress_stall_v3,0.1-0.25,1-2,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,progress_stall_v3,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0509568029941887,progress_stall_v3,0.1-0.25,2-3,11-20,3,1.0,0.0
```

## p_exec Diagnostics
```csv
path,val_auroc,val_auprc,p_exec_mean,p_exec_q10,p_exec_q50,p_exec_q90
artifacts/stage22/antmaze-medium-navigate-v0/seed0/reachability_metrics.json,0.941183730900299,0.9541894003456234,0.9975948333740234,0.9986407697200775,0.9999974966049194,1.0
artifacts/stage22/antmaze-medium-stitch-v0/seed0/reachability_metrics.json,0.915530149188428,0.9413985224959389,0.9961159229278564,0.9978068768978119,0.999995231628418,1.0
```

## Boundary Diagnostics
```csv
path,coverage,supported_pair_rate,psi_q10,psi_q50,psi_q90
artifacts/stage22/antmaze-medium-navigate-v0/seed0/boundary_summary.json,15.775757575757575,0.9610065309258548,0.5769317448139191,0.8434385359287262,0.9749203622341156
artifacts/stage22/antmaze-medium-stitch-v0/seed0/boundary_summary.json,23.242966751918157,0.9628885089382448,0.5509623169898987,0.8364202082157135,0.970407122373581
```

## Decision Summary
- INSUFFICIENT_EVIDENCE
