# Stage22 Summary

Updated: 2026-05-19 09:44:52

## Completion
```csv
env,seed,variant,budget,fallback_mode,episodes,success,steps,fallback_used,no_path_rate,budget_reject_rate
antmaze-medium-navigate-v0,0,gas_boundary_budget,2.0,none,50,0.0,0.0,0.0,1.0,1.0
antmaze-medium-navigate-v0,0,gas_boundary_budget,2.0,progress_stall_v2,50,0.0,0.0,0.0,1.0,1.0
antmaze-medium-navigate-v0,0,gas_boundary_budget,3.0,none,50,0.0,0.0,0.0,1.0,1.0
antmaze-medium-navigate-v0,0,gas_boundary_budget,3.0,progress_stall_v2,50,0.0,0.0,0.0,1.0,1.0
antmaze-medium-navigate-v0,0,gas_reachability_budget,2.0,none,50,0.88,357.98,0.0,0.0,0.0
antmaze-medium-navigate-v0,0,gas_reachability_budget,2.0,progress_stall_v2,50,0.88,362.4,0.5,0.0,0.0
antmaze-medium-navigate-v0,0,gas_reachability_budget,3.0,none,50,0.8,424.9,0.0,0.0,0.0
antmaze-medium-navigate-v0,0,gas_reachability_budget,3.0,progress_stall_v2,50,0.9,369.98,0.62,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,2.0,none,50,0.86,381.1,0.0,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,2.0,progress_stall_v2,50,0.86,372.32,0.54,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,3.0,none,50,0.84,385.72,0.0,0.0,0.0
antmaze-medium-navigate-v0,0,gas_shortest,3.0,progress_stall_v2,50,0.88,354.68,0.62,0.0,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,2.0,none,50,0.0,0.0,0.0,1.0,1.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,2.0,progress_stall_v2,50,0.0,0.0,0.0,1.0,1.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,3.0,none,50,0.0,0.0,0.0,1.0,1.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,3.0,progress_stall_v2,50,0.0,0.0,0.0,1.0,1.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,5.0,none,3,0.0,0.0,0.0,1.0,1.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,5.0,progress_stall_v2,3,0.0,0.0,0.0,1.0,1.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,8.0,none,3,1.0,252.0,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,8.0,progress_stall_v2,3,1.0,211.66666666666666,0.3333333333333333,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget,2.0,none,50,0.82,375.4,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget,2.0,progress_stall_v2,50,0.9,321.26,0.26,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget,3.0,none,50,0.92,321.3,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget,3.0,progress_stall_v2,50,0.94,300.62,0.52,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget,5.0,none,50,0.88,333.48,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget,5.0,progress_stall_v2,50,0.9,318.86,0.4,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget,8.0,none,50,0.9,329.74,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_reachability_budget,8.0,progress_stall_v2,50,0.88,333.12,0.36,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,2.0,none,50,0.9,322.52,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,2.0,progress_stall_v2,50,0.88,339.32,0.46,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,3.0,none,50,0.92,321.36,0.0,0.0,0.0
antmaze-medium-stitch-v0,0,gas_shortest,3.0,progress_stall_v2,50,0.88,357.34,0.52,0.0,0.0
```

## Same-Backbone Comparison
```csv
variant,baseline,paired_n,success_delta,baseline_success,variant_success
gas_boundary_budget,gas_shortest,400,-0.8775,0.8775,0.0
gas_reachability_budget,gas_shortest,400,0.0025,0.8775,0.88
```

## Fallback Attribution
```csv
fallback_mode,fallback_used,episodes,success
none,0,706,0.6218130311614731
progress_stall_v2,0,465,0.4666666666666667
progress_stall_v2,1,241,0.9585062240663901
```

## Path Diagnostics
```csv
env,variant,budget,fallback_mode,pred_bucket,exec_risk_bucket,first_plan_edges_bucket,episodes,success,no_path_rate
antmaze-medium-navigate-v0,gas_boundary_budget,2.0,none,0-0.1,0-0.5,0,50,0.0,1.0
antmaze-medium-navigate-v0,gas_boundary_budget,2.0,progress_stall_v2,0-0.1,0-0.5,0,50,0.0,1.0
antmaze-medium-navigate-v0,gas_boundary_budget,3.0,none,0-0.1,0-0.5,0,50,0.0,1.0
antmaze-medium-navigate-v0,gas_boundary_budget,3.0,progress_stall_v2,0-0.1,0-0.5,0,50,0.0,1.0
antmaze-medium-navigate-v0,gas_reachability_budget,2.0,none,0.1-0.25,1-2,6-10,6,0.6666666666666666,0.0
antmaze-medium-navigate-v0,gas_reachability_budget,2.0,none,0.1-0.25,1-2,11-20,44,0.9090909090909091,0.0
antmaze-medium-navigate-v0,gas_reachability_budget,2.0,progress_stall_v2,0.1-0.25,1-2,6-10,4,0.75,0.0
antmaze-medium-navigate-v0,gas_reachability_budget,2.0,progress_stall_v2,0.1-0.25,1-2,11-20,46,0.8913043478260869,0.0
antmaze-medium-navigate-v0,gas_reachability_budget,3.0,none,0-0.1,2-3,6-10,9,0.5555555555555556,0.0
antmaze-medium-navigate-v0,gas_reachability_budget,3.0,none,0-0.1,2-3,11-20,41,0.8536585365853658,0.0
antmaze-medium-navigate-v0,gas_reachability_budget,3.0,progress_stall_v2,0-0.1,2-3,6-10,8,0.875,0.0
antmaze-medium-navigate-v0,gas_reachability_budget,3.0,progress_stall_v2,0-0.1,2-3,11-20,42,0.9047619047619048,0.0
antmaze-medium-navigate-v0,gas_shortest,2.0,none,0-0.1,2-3,6-10,1,1.0,0.0
antmaze-medium-navigate-v0,gas_shortest,2.0,none,0-0.1,3-5,6-10,9,0.7777777777777778,0.0
antmaze-medium-navigate-v0,gas_shortest,2.0,none,0-0.1,3-5,11-20,36,0.8888888888888888,0.0
antmaze-medium-navigate-v0,gas_shortest,2.0,none,0-0.1,>5,11-20,4,0.75,0.0
antmaze-medium-navigate-v0,gas_shortest,2.0,progress_stall_v2,0-0.1,2-3,6-10,1,1.0,0.0
antmaze-medium-navigate-v0,gas_shortest,2.0,progress_stall_v2,0-0.1,3-5,6-10,9,0.7777777777777778,0.0
antmaze-medium-navigate-v0,gas_shortest,2.0,progress_stall_v2,0-0.1,3-5,11-20,37,0.8648648648648649,0.0
antmaze-medium-navigate-v0,gas_shortest,2.0,progress_stall_v2,0-0.1,>5,11-20,3,1.0,0.0
antmaze-medium-navigate-v0,gas_shortest,3.0,none,0-0.1,2-3,6-10,1,1.0,0.0
antmaze-medium-navigate-v0,gas_shortest,3.0,none,0-0.1,3-5,6-10,9,0.6666666666666666,0.0
antmaze-medium-navigate-v0,gas_shortest,3.0,none,0-0.1,3-5,11-20,37,0.8648648648648649,0.0
antmaze-medium-navigate-v0,gas_shortest,3.0,none,0-0.1,>5,11-20,3,1.0,0.0
antmaze-medium-navigate-v0,gas_shortest,3.0,progress_stall_v2,0-0.1,3-5,6-10,10,0.8,0.0
antmaze-medium-navigate-v0,gas_shortest,3.0,progress_stall_v2,0-0.1,3-5,11-20,37,0.918918918918919,0.0
antmaze-medium-navigate-v0,gas_shortest,3.0,progress_stall_v2,0-0.1,>5,11-20,3,0.6666666666666666,0.0
antmaze-medium-stitch-v0,gas_boundary_budget,2.0,none,0-0.1,0-0.5,0,50,0.0,1.0
antmaze-medium-stitch-v0,gas_boundary_budget,2.0,progress_stall_v2,0-0.1,0-0.5,0,50,0.0,1.0
antmaze-medium-stitch-v0,gas_boundary_budget,3.0,none,0-0.1,0-0.5,0,50,0.0,1.0
antmaze-medium-stitch-v0,gas_boundary_budget,3.0,progress_stall_v2,0-0.1,0-0.5,0,50,0.0,1.0
antmaze-medium-stitch-v0,gas_boundary_budget,5.0,none,0-0.1,0-0.5,0,3,0.0,1.0
antmaze-medium-stitch-v0,gas_boundary_budget,5.0,progress_stall_v2,0-0.1,0-0.5,0,3,0.0,1.0
antmaze-medium-stitch-v0,gas_boundary_budget,8.0,none,0-0.1,2-3,11-20,3,1.0,0.0
antmaze-medium-stitch-v0,gas_boundary_budget,8.0,progress_stall_v2,0-0.1,2-3,11-20,3,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,2.0,none,0.1-0.25,1-2,11-20,50,0.82,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,2.0,progress_stall_v2,0.1-0.25,1-2,11-20,50,0.9,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,3.0,none,0-0.1,2-3,6-10,7,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,3.0,none,0-0.1,2-3,11-20,39,0.8974358974358975,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,3.0,none,0.1-0.25,1-2,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,3.0,none,0.1-0.25,2-3,11-20,3,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,3.0,progress_stall_v2,0-0.1,2-3,6-10,6,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,3.0,progress_stall_v2,0-0.1,2-3,11-20,40,0.925,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,3.0,progress_stall_v2,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,3.0,progress_stall_v2,0.1-0.25,2-3,11-20,3,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,none,0-0.1,2-3,6-10,6,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,none,0-0.1,3-5,11-20,40,0.85,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,none,0.1-0.25,1-2,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,none,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,none,0.1-0.25,2-3,11-20,2,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,progress_stall_v2,0-0.1,2-3,6-10,6,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,progress_stall_v2,0-0.1,3-5,11-20,40,0.875,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,progress_stall_v2,0.1-0.25,1-2,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,progress_stall_v2,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,5.0,progress_stall_v2,0.1-0.25,2-3,11-20,2,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,none,0-0.1,2-3,6-10,5,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,none,0-0.1,3-5,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,none,0-0.1,3-5,11-20,34,0.9411764705882353,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,none,0-0.1,>5,11-20,6,0.5,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,none,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,none,0.1-0.25,2-3,11-20,3,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,progress_stall_v2,0-0.1,2-3,6-10,7,1.0,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,progress_stall_v2,0-0.1,3-5,11-20,34,0.9411764705882353,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,progress_stall_v2,0-0.1,>5,11-20,6,0.3333333333333333,0.0
antmaze-medium-stitch-v0,gas_reachability_budget,8.0,progress_stall_v2,0.1-0.25,2-3,11-20,3,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,none,0-0.1,2-3,6-10,3,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,none,0-0.1,2-3,11-20,18,0.9444444444444444,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,none,0-0.1,3-5,11-20,20,0.8,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,none,0.1-0.25,1-2,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,none,0.1-0.25,1-2,11-20,4,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,none,0.1-0.25,2-3,6-10,3,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,none,0.1-0.25,2-3,11-20,1,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,progress_stall_v2,0-0.1,2-3,6-10,6,0.8333333333333334,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,progress_stall_v2,0-0.1,3-5,11-20,34,0.9411764705882353,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,progress_stall_v2,0-0.1,>5,11-20,6,0.5,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,progress_stall_v2,0.1-0.25,2-3,6-10,1,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,2.0,progress_stall_v2,0.1-0.25,2-3,11-20,3,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0,none,0-0.1,2-3,6-10,3,1.0,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0,none,0-0.1,2-3,11-20,12,0.9166666666666666,0.0
antmaze-medium-stitch-v0,gas_shortest,3.0,none,0-0.1,3-5,11-20,24,0.875,0.0
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
- HOLD_BOUNDARY
