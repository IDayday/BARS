# Stage22R Risk Calibration

## Recommended Budgets
```csv
env,seed,shortest_episodes,reachability_episodes,exec_median,boundary_median,alpha_boundary,boundary_status,exec_q50,exec_q60,exec_q70,exec_q80,reach_exec_q50,reach_exec_q60,reach_exec_q70,total_q50,total_q60,total_q70,total_q80,reach_exec_q80
antmaze-medium-navigate-v0,0,100,200,4.053035076731126,5.410929666733429,0.7490459729405313,CALIBRATED,4.053035076731126,4.277040201695186,4.450541132461325,4.598044733611958,2.176670096486279,2.81638062579149,2.8893971614298044,9.76593372124477,11.935600818204565,12.50820803860719,13.422441130933098,
antmaze-medium-stitch-v0,0,100,400,2.9281707459035102,6.736804156778545,0.43465279348475594,CALIBRATED,2.9281707459035102,3.0509568029941887,3.101527554013284,3.182596181065707,2.878800011114404,3.1633406682684555,4.056115893180538,10.284622343888078,11.601137245305774,11.741843927194726,12.611872504355048,4.353366562973274
```

## JSON
```json
{
  "envs": {
    "antmaze-medium-navigate-v0/seed0": {
      "alpha_boundary": 0.7490459729405313,
      "boundary_status": "CALIBRATED",
      "env": "antmaze-medium-navigate-v0",
      "exec_budgets_from_shortest": {
        "exec_q50": 4.053035076731126,
        "exec_q60": 4.277040201695186,
        "exec_q70": 4.450541132461325,
        "exec_q80": 4.598044733611958
      },
      "reachability_budgets_from_selected": {
        "reach_exec_q50": 2.176670096486279,
        "reach_exec_q60": 2.81638062579149,
        "reach_exec_q70": 2.8893971614298044
      },
      "recommended_boundary_budget": 11.935600818204565,
      "recommended_reachability_budget": 4.277040201695186,
      "seed": 0,
      "total_budgets_for_boundary": {
        "total_q50": 9.76593372124477,
        "total_q60": 11.935600818204565,
        "total_q70": 12.50820803860719,
        "total_q80": 13.422441130933098
      }
    },
    "antmaze-medium-stitch-v0/seed0": {
      "alpha_boundary": 0.43465279348475594,
      "boundary_status": "CALIBRATED",
      "env": "antmaze-medium-stitch-v0",
      "exec_budgets_from_shortest": {
        "exec_q50": 2.9281707459035102,
        "exec_q60": 3.0509568029941887,
        "exec_q70": 3.101527554013284,
        "exec_q80": 3.182596181065707
      },
      "reachability_budgets_from_selected": {
        "reach_exec_q50": 2.878800011114404,
        "reach_exec_q60": 3.1633406682684555,
        "reach_exec_q70": 4.056115893180538,
        "reach_exec_q80": 4.353366562973274
      },
      "recommended_boundary_budget": 11.601137245305774,
      "recommended_reachability_budget": 3.0509568029941887,
      "seed": 0,
      "total_budgets_for_boundary": {
        "total_q50": 10.284622343888078,
        "total_q60": 11.601137245305774,
        "total_q70": 11.741843927194726,
        "total_q80": 12.611872504355048
      }
    }
  }
}
```