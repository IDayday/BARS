# Stage22R Boundary Feasibility

## Summary
```csv
env,seed,section,episodes,exec_risk_q50,exec_risk_q60,exec_risk_q70,exec_risk_q80,boundary_risk_q50,boundary_risk_q60,boundary_risk_q70,boundary_risk_q80,total_risk_q50,total_risk_q60,total_risk_q70,total_risk_q80,missing_boundary_pair_rate,virtual_boundary_pair_count,unsupported_pair_rate,budget_reject_rate,no_path_rate,B_q50,B_q60,B_q70,B_q80
antmaze-medium-navigate-v0,0,shortest,100,4.053035076731126,4.277040201695186,4.450541132461325,4.598044733611958,5.410929666733429,7.588755562670816,7.721526605273478,8.652819412554846,9.76593372124477,11.935600818204565,12.50820803860719,13.422441130933098,0.17480386280386276,200.0,0.07896869796869796,0.0,0.0,,,,
antmaze-medium-navigate-v0,0,boundary_eval,200,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,,,,
antmaze-medium-navigate-v0,0,recommended_budget,100,,,,,,,,,,,,,,,,,,9.76593372124477,11.935600818204565,12.50820803860719,13.422441130933098
antmaze-medium-stitch-v0,0,shortest,100,2.9281707459035102,3.0509568029941887,3.101527554013284,3.182596181065707,6.736804156778545,8.368907380127942,9.258972971063574,9.513465589323891,10.284622343888078,11.601137245305774,11.741843927194726,12.611872504355048,0.1656140526140526,200.0,0.06297868797868798,0.0,0.0,,,,
antmaze-medium-stitch-v0,0,boundary_eval,212,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.004323899371069182,12.0,0.0,0.9716981132075472,0.9716981132075472,,,,
antmaze-medium-stitch-v0,0,recommended_budget,100,,,,,,,,,,,,,,,,,,10.284622343888078,11.601137245305774,11.741843927194726,12.611872504355048
```

## Reject Reasons
```csv
env,seed,variant,budget,fallback_mode,reject_class,episodes,success
antmaze-medium-navigate-v0,0,gas_boundary_budget,2.0,none,line-graph pruning,50,0.0
antmaze-medium-navigate-v0,0,gas_boundary_budget,2.0,progress_stall_v2,line-graph pruning,50,0.0
antmaze-medium-navigate-v0,0,gas_boundary_budget,3.0,none,line-graph pruning,50,0.0
antmaze-medium-navigate-v0,0,gas_boundary_budget,3.0,progress_stall_v2,line-graph pruning,50,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,2.0,none,line-graph pruning,50,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,2.0,progress_stall_v2,line-graph pruning,50,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,3.0,none,line-graph pruning,50,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,3.0,progress_stall_v2,line-graph pruning,50,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,5.0,none,line-graph pruning,3,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,5.0,progress_stall_v2,line-graph pruning,3,0.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,8.0,none,missing virtual edge-pair,3,1.0
antmaze-medium-stitch-v0,0,gas_boundary_budget,8.0,progress_stall_v2,missing virtual edge-pair,3,1.0
```

## Interpretation
- HOLD_BOUNDARY: at least one boundary setting rejects >=50% of paths.