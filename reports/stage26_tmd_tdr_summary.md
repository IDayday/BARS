# Stage 26 TMD/TDR Summary

Evaluation rows: 14450
Graph rows: 26

## Phase B Aggregate Comparisons

| env | run_episodes | weight | gas_n | gas_success_rate | variant_n | variant_success_rate | delta_success_rate | normal95_low | normal95_high | bootstrap95_low | bootstrap95_high | delta_mean_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| antmaze-giant-navigate-v0 | 20 | 0.1 | 600 | 0.658 | 600 | 0.660 | 0.002 | -0.052 | 0.055 | -0.050 | 0.055 | 0.680 |
| antmaze-giant-navigate-v0 | 20 | 0.2 | 600 | 0.658 | 600 | 0.700 | 0.042 | -0.011 | 0.094 | 0.000 | 0.082 | -12.618 |
| antmaze-giant-navigate-v0 | 20 | 0.25 | 600 | 0.658 | 600 | 0.733 | 0.075 | 0.023 | 0.127 | 0.023 | 0.130 | -18.992 |
| antmaze-giant-navigate-v0 | 20 | 0.3 | 600 | 0.658 | 600 | 0.690 | 0.032 | -0.021 | 0.085 | -0.010 | 0.073 | -6.382 |
| antmaze-giant-navigate-v0 | 20 | 0.4 | 600 | 0.658 | 600 | 0.690 | 0.032 | -0.021 | 0.085 | -0.015 | 0.080 | -4.790 |
| antmaze-giant-navigate-v0 | 20 | 0.5 | 600 | 0.658 | 600 | 0.653 | -0.005 | -0.059 | 0.049 | -0.060 | 0.047 | 10.162 |
| antmaze-giant-navigate-v0 | 50 | 0.25 | 1500 | 0.677 | 1500 | 0.713 | 0.037 | 0.004 | 0.070 | 0.016 | 0.059 | -10.622 |
| antmaze-giant-stitch-v0 | 20 | 0.1 | 600 | 0.865 | 600 | 0.867 | 0.002 | -0.037 | 0.040 | -0.033 | 0.037 | -4.047 |
| antmaze-giant-stitch-v0 | 20 | 0.2 | 600 | 0.865 | 600 | 0.862 | -0.003 | -0.042 | 0.036 | -0.033 | 0.032 | 6.557 |
| antmaze-giant-stitch-v0 | 20 | 0.25 | 600 | 0.865 | 600 | 0.877 | 0.012 | -0.026 | 0.050 | -0.020 | 0.042 | -4.968 |
| antmaze-giant-stitch-v0 | 20 | 0.3 | 600 | 0.865 | 600 | 0.848 | -0.017 | -0.056 | 0.023 | -0.060 | 0.022 | 4.868 |
| antmaze-giant-stitch-v0 | 20 | 0.4 | 600 | 0.865 | 600 | 0.852 | -0.013 | -0.053 | 0.026 | -0.045 | 0.020 | 1.680 |
| antmaze-giant-stitch-v0 | 20 | 0.5 | 600 | 0.865 | 600 | 0.873 | 0.008 | -0.030 | 0.046 | -0.028 | 0.045 | -2.795 |
| antmaze-medium-navigate-v0 | 20 | 0.1 | 400 | 0.958 | 400 | 0.965 | 0.007 | -0.019 | 0.034 | -0.018 | 0.035 | -8.803 |
| antmaze-medium-navigate-v0 | 20 | 0.25 | 400 | 0.958 | 400 | 0.975 | 0.017 | -0.007 | 0.042 | -0.010 | 0.047 | -13.848 |
| antmaze-medium-stitch-v0 | 20 | 0.1 | 400 | 0.978 | 400 | 0.985 | 0.007 | -0.011 | 0.026 | -0.013 | 0.030 | -1.925 |
| antmaze-medium-stitch-v0 | 20 | 0.25 | 400 | 0.978 | 400 | 0.983 | 0.005 | -0.014 | 0.024 | -0.010 | 0.020 | -3.888 |


## Phase B Task-Wise Deltas

| env | run_episodes | task_id | weight | gas_success_rate | variant_success_rate | delta_success_rate | delta_mean_steps |
|---|---|---|---|---|---|---|---|
| antmaze-giant-navigate-v0 | 20 | 1 | 0.1 | 0.183 | 0.217 | 0.033 | -0.667 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.2 | 0.183 | 0.175 | -0.008 | 0.175 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.25 | 0.183 | 0.233 | 0.050 | -1.325 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.3 | 0.183 | 0.142 | -0.042 | -2.008 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.4 | 0.183 | 0.225 | 0.042 | -4.417 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.5 | 0.183 | 0.175 | -0.008 | 2.817 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.1 | 0.833 | 0.875 | 0.042 | -6.333 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.2 | 0.833 | 0.883 | 0.050 | -12.150 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.25 | 0.833 | 0.908 | 0.075 | -15.358 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.3 | 0.833 | 0.933 | 0.100 | -30.608 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.4 | 0.833 | 0.892 | 0.058 | -11.683 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.5 | 0.833 | 0.850 | 0.017 | 7.450 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.1 | 0.758 | 0.708 | -0.050 | 8.358 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.2 | 0.758 | 0.833 | 0.075 | -14.742 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.25 | 0.758 | 0.867 | 0.108 | -17.842 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.3 | 0.758 | 0.750 | -0.008 | 9.700 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.4 | 0.758 | 0.708 | -0.050 | 43.517 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.5 | 0.758 | 0.617 | -0.142 | 73.800 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.1 | 0.800 | 0.742 | -0.058 | 13.100 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.2 | 0.800 | 0.825 | 0.025 | 0.933 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.25 | 0.800 | 0.883 | 0.083 | -31.283 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.3 | 0.800 | 0.833 | 0.033 | 2.858 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.4 | 0.800 | 0.842 | 0.042 | -24.333 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.5 | 0.800 | 0.850 | 0.050 | -19.967 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.1 | 0.717 | 0.758 | 0.042 | -11.058 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.2 | 0.717 | 0.783 | 0.067 | -37.308 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.25 | 0.717 | 0.775 | 0.058 | -29.150 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.3 | 0.717 | 0.792 | 0.075 | -11.850 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.4 | 0.717 | 0.783 | 0.067 | -27.033 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.5 | 0.717 | 0.775 | 0.058 | -13.292 |
| antmaze-giant-navigate-v0 | 50 | 1 | 0.25 | 0.180 | 0.233 | 0.053 | -7.873 |
| antmaze-giant-navigate-v0 | 50 | 2 | 0.25 | 0.877 | 0.903 | 0.027 | 3.663 |
| antmaze-giant-navigate-v0 | 50 | 3 | 0.25 | 0.780 | 0.833 | 0.053 | -17.813 |
| antmaze-giant-navigate-v0 | 50 | 4 | 0.25 | 0.787 | 0.820 | 0.033 | -21.807 |
| antmaze-giant-navigate-v0 | 50 | 5 | 0.25 | 0.760 | 0.777 | 0.017 | -9.280 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.1 | 0.750 | 0.708 | -0.042 | 7.283 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.2 | 0.750 | 0.808 | 0.058 | -0.850 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.25 | 0.750 | 0.708 | -0.042 | 23.650 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.3 | 0.750 | 0.692 | -0.058 | 19.933 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.4 | 0.750 | 0.742 | -0.008 | 2.600 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.5 | 0.750 | 0.758 | 0.008 | 5.625 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.1 | 0.942 | 0.908 | -0.033 | 9.217 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.2 | 0.942 | 0.917 | -0.025 | 14.792 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.25 | 0.942 | 0.942 | 0.000 | 4.933 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.3 | 0.942 | 0.908 | -0.033 | 12.442 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.4 | 0.942 | 0.892 | -0.050 | 24.025 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.5 | 0.942 | 0.917 | -0.025 | 10.533 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.1 | 0.775 | 0.842 | 0.067 | -23.200 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.2 | 0.775 | 0.817 | 0.042 | -25.483 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.25 | 0.775 | 0.858 | 0.083 | -38.492 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.3 | 0.775 | 0.825 | 0.050 | -26.700 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.4 | 0.775 | 0.808 | 0.033 | -23.567 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.5 | 0.775 | 0.833 | 0.058 | -30.733 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.1 | 0.917 | 0.925 | 0.008 | -0.167 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.2 | 0.917 | 0.867 | -0.050 | 28.517 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.25 | 0.917 | 0.942 | 0.025 | -14.267 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.3 | 0.917 | 0.908 | -0.008 | 10.325 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.4 | 0.917 | 0.883 | -0.033 | 5.958 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.5 | 0.917 | 0.942 | 0.025 | -3.983 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.1 | 0.942 | 0.950 | 0.008 | -13.367 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.2 | 0.942 | 0.900 | -0.042 | 15.808 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.25 | 0.942 | 0.933 | -0.008 | -0.667 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.3 | 0.942 | 0.908 | -0.033 | 8.342 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.4 | 0.942 | 0.933 | -0.008 | -0.617 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.5 | 0.942 | 0.917 | -0.025 | 4.583 |
| antmaze-medium-navigate-v0 | 20 | 1 | 0.1 | 0.963 | 0.925 | -0.037 | 20.550 |
| antmaze-medium-navigate-v0 | 20 | 1 | 0.25 | 0.963 | 0.912 | -0.050 | 32.375 |
| antmaze-medium-navigate-v0 | 20 | 2 | 0.1 | 0.975 | 0.963 | -0.012 | 10.312 |
| antmaze-medium-navigate-v0 | 20 | 2 | 0.25 | 0.975 | 0.988 | 0.013 | -3.000 |
| antmaze-medium-navigate-v0 | 20 | 3 | 0.1 | 0.912 | 0.950 | 0.037 | -34.037 |
| antmaze-medium-navigate-v0 | 20 | 3 | 0.25 | 0.912 | 0.988 | 0.075 | -61.388 |
| antmaze-medium-navigate-v0 | 20 | 4 | 0.1 | 0.963 | 1.000 | 0.037 | -28.213 |
| antmaze-medium-navigate-v0 | 20 | 4 | 0.25 | 0.963 | 1.000 | 0.037 | -32.063 |
| antmaze-medium-navigate-v0 | 20 | 5 | 0.1 | 0.975 | 0.988 | 0.013 | -12.625 |
| antmaze-medium-navigate-v0 | 20 | 5 | 0.25 | 0.975 | 0.988 | 0.013 | -5.162 |
| antmaze-medium-stitch-v0 | 20 | 1 | 0.1 | 0.938 | 0.975 | 0.037 | -29.875 |
| antmaze-medium-stitch-v0 | 20 | 1 | 0.25 | 0.938 | 0.975 | 0.037 | -29.712 |
| antmaze-medium-stitch-v0 | 20 | 2 | 0.1 | 1.000 | 1.000 | 0.000 | -10.412 |
| antmaze-medium-stitch-v0 | 20 | 2 | 0.25 | 1.000 | 0.975 | -0.025 | 10.137 |
| antmaze-medium-stitch-v0 | 20 | 3 | 0.1 | 1.000 | 1.000 | 0.000 | -0.075 |
| antmaze-medium-stitch-v0 | 20 | 3 | 0.25 | 1.000 | 0.988 | -0.012 | 8.838 |
| antmaze-medium-stitch-v0 | 20 | 4 | 0.1 | 0.975 | 0.988 | 0.013 | 9.250 |
| antmaze-medium-stitch-v0 | 20 | 4 | 0.25 | 0.975 | 0.988 | 0.013 | 1.450 |
| antmaze-medium-stitch-v0 | 20 | 5 | 0.1 | 0.975 | 0.963 | -0.012 | 21.488 |
| antmaze-medium-stitch-v0 | 20 | 5 | 0.25 | 0.975 | 0.988 | 0.013 | -10.150 |


## Phase D Low-Level Condition Comparisons

| env | run_episodes | variant | gas_n | gas_success_rate | variant_n | variant_success_rate | delta_success_rate | normal95_low | normal95_high | bootstrap95_low | bootstrap95_high | delta_mean_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| antmaze-giant-navigate-v0 | 5 | lowcond_factor_only_nearestgoal | 25 | 0.520 | 25 | 0.000 | -0.520 | -0.716 | -0.324 | -0.800 | -0.240 | 138.680 |
| antmaze-giant-navigate-v0 | 5 | lowcond_full_localres | 25 | 0.520 | 25 | 0.040 | -0.480 | -0.690 | -0.270 | -0.800 | -0.200 | 134.000 |
| antmaze-giant-navigate-v0 | 5 | lowcond_full_nearestgoal | 25 | 0.520 | 25 | 0.080 | -0.440 | -0.663 | -0.217 | -0.760 | -0.120 | 116.240 |
| antmaze-giant-navigate-v0 | 5 | lowcond_full_nomask_nearestgoal | 25 | 0.520 | 25 | 0.040 | -0.480 | -0.690 | -0.270 | -0.800 | -0.200 | 130.480 |
| antmaze-giant-navigate-v0 | 5 | lowcond_full_rawdist_nearestgoal | 25 | 0.520 | 25 | 0.000 | -0.520 | -0.716 | -0.324 | -0.800 | -0.240 | 138.680 |
| antmaze-giant-navigate-v0 | 5 | lowcond_full_trajend | 25 | 0.520 | 25 | 0.080 | -0.440 | -0.663 | -0.217 | -0.760 | -0.120 | 115.640 |
| antmaze-giant-navigate-v0 | 5 | lowcond_tdr_only_local | 25 | 0.520 | 25 | 0.200 | -0.320 | -0.571 | -0.069 | -0.760 | 0.160 | 98.120 |
| antmaze-medium-navigate-v0 | 5 | lowcond_factor_only_nearestgoal | 25 | 1.000 | 25 | 0.000 | -1.000 | -1.000 | -1.000 | -1.000 | -1.000 | 771.320 |
| antmaze-medium-navigate-v0 | 5 | lowcond_full_localres | 25 | 1.000 | 25 | 0.880 | -0.120 | -0.247 | 0.007 | -0.360 | 0.000 | 250.480 |
| antmaze-medium-navigate-v0 | 5 | lowcond_full_nearestgoal | 25 | 1.000 | 25 | 0.720 | -0.280 | -0.456 | -0.104 | -0.560 | -0.080 | 392.720 |
| antmaze-medium-navigate-v0 | 5 | lowcond_full_nomask_nearestgoal | 25 | 1.000 | 25 | 0.760 | -0.240 | -0.407 | -0.073 | -0.360 | -0.120 | 355.200 |
| antmaze-medium-navigate-v0 | 5 | lowcond_full_rawdist_nearestgoal | 25 | 1.000 | 25 | 0.520 | -0.480 | -0.676 | -0.284 | -0.800 | -0.160 | 551.840 |
| antmaze-medium-navigate-v0 | 5 | lowcond_full_trajend | 25 | 1.000 | 25 | 0.600 | -0.400 | -0.592 | -0.208 | -0.560 | -0.240 | 474.040 |
| antmaze-medium-navigate-v0 | 5 | lowcond_tdr_only_local | 25 | 1.000 | 25 | 0.760 | -0.240 | -0.407 | -0.073 | -0.360 | -0.120 | 282.760 |
| antmaze-medium-stitch-v0 | 5 | lowcond_factor_only_nearestgoal | 25 | 1.000 | 25 | 0.080 | -0.920 | -1.026 | -0.814 | -1.000 | -0.760 | 741.000 |
| antmaze-medium-stitch-v0 | 5 | lowcond_full | 25 | 1.000 | 25 | 0.720 | -0.280 | -0.456 | -0.104 | -0.520 | -0.040 | 364.800 |
| antmaze-medium-stitch-v0 | 5 | lowcond_full_localres | 25 | 1.000 | 25 | 0.800 | -0.200 | -0.357 | -0.043 | -0.320 | -0.080 | 285.280 |
| antmaze-medium-stitch-v0 | 5 | lowcond_full_localres_from_full | 25 | 1.000 | 25 | 0.920 | -0.080 | -0.186 | 0.026 | -0.240 | 0.000 | 157.080 |
| antmaze-medium-stitch-v0 | 5 | lowcond_full_nearestgoal | 25 | 1.000 | 25 | 0.880 | -0.120 | -0.247 | 0.007 | -0.200 | -0.040 | 189.280 |
| antmaze-medium-stitch-v0 | 5 | lowcond_full_nomask_nearestgoal | 25 | 1.000 | 25 | 0.760 | -0.240 | -0.407 | -0.073 | -0.360 | -0.120 | 266.400 |
| antmaze-medium-stitch-v0 | 5 | lowcond_full_rawdist_nearestgoal | 25 | 1.000 | 25 | 0.840 | -0.160 | -0.304 | -0.016 | -0.280 | -0.040 | 233.240 |
| antmaze-medium-stitch-v0 | 5 | lowcond_full_trajend | 25 | 1.000 | 25 | 0.720 | -0.280 | -0.456 | -0.104 | -0.640 | -0.040 | 301.040 |
| antmaze-medium-stitch-v0 | 5 | lowcond_tdr_only_local | 25 | 1.000 | 25 | 0.960 | -0.040 | -0.117 | 0.037 | -0.120 | 0.000 | 158.200 |


## Phase D Low-Level Condition Task-Wise Deltas

| env | run_episodes | task_id | variant | gas_success_rate | variant_success_rate | delta_success_rate | delta_mean_steps |
|---|---|---|---|---|---|---|---|
| antmaze-giant-navigate-v0 | 5 | 1 | lowcond_factor_only_nearestgoal | 0.200 | 0.000 | -0.200 | 1.000 |
| antmaze-giant-navigate-v0 | 5 | 1 | lowcond_full_localres | 0.200 | 0.000 | -0.200 | 1.000 |
| antmaze-giant-navigate-v0 | 5 | 1 | lowcond_full_nearestgoal | 0.200 | 0.000 | -0.200 | 1.000 |
| antmaze-giant-navigate-v0 | 5 | 1 | lowcond_full_nomask_nearestgoal | 0.200 | 0.000 | -0.200 | 1.000 |
| antmaze-giant-navigate-v0 | 5 | 1 | lowcond_full_rawdist_nearestgoal | 0.200 | 0.000 | -0.200 | 1.000 |
| antmaze-giant-navigate-v0 | 5 | 1 | lowcond_full_trajend | 0.200 | 0.000 | -0.200 | 1.000 |
| antmaze-giant-navigate-v0 | 5 | 1 | lowcond_tdr_only_local | 0.200 | 0.000 | -0.200 | 1.000 |
| antmaze-giant-navigate-v0 | 5 | 2 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 262.400 |
| antmaze-giant-navigate-v0 | 5 | 2 | lowcond_full_localres | 1.000 | 0.000 | -1.000 | 262.400 |
| antmaze-giant-navigate-v0 | 5 | 2 | lowcond_full_nearestgoal | 1.000 | 0.000 | -1.000 | 262.400 |
| antmaze-giant-navigate-v0 | 5 | 2 | lowcond_full_nomask_nearestgoal | 1.000 | 0.000 | -1.000 | 262.400 |
| antmaze-giant-navigate-v0 | 5 | 2 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.000 | -1.000 | 262.400 |
| antmaze-giant-navigate-v0 | 5 | 2 | lowcond_full_trajend | 1.000 | 0.000 | -1.000 | 262.400 |
| antmaze-giant-navigate-v0 | 5 | 2 | lowcond_tdr_only_local | 1.000 | 0.000 | -1.000 | 262.400 |
| antmaze-giant-navigate-v0 | 5 | 3 | lowcond_factor_only_nearestgoal | 0.200 | 0.000 | -0.200 | 54.600 |
| antmaze-giant-navigate-v0 | 5 | 3 | lowcond_full_localres | 0.200 | 0.000 | -0.200 | 54.600 |
| antmaze-giant-navigate-v0 | 5 | 3 | lowcond_full_nearestgoal | 0.200 | 0.000 | -0.200 | 54.600 |
| antmaze-giant-navigate-v0 | 5 | 3 | lowcond_full_nomask_nearestgoal | 0.200 | 0.000 | -0.200 | 54.600 |
| antmaze-giant-navigate-v0 | 5 | 3 | lowcond_full_rawdist_nearestgoal | 0.200 | 0.000 | -0.200 | 54.600 |
| antmaze-giant-navigate-v0 | 5 | 3 | lowcond_full_trajend | 0.200 | 0.000 | -0.200 | 54.600 |
| antmaze-giant-navigate-v0 | 5 | 3 | lowcond_tdr_only_local | 0.200 | 0.000 | -0.200 | 54.600 |
| antmaze-giant-navigate-v0 | 5 | 4 | lowcond_factor_only_nearestgoal | 0.800 | 0.000 | -0.800 | 175.000 |
| antmaze-giant-navigate-v0 | 5 | 4 | lowcond_full_localres | 0.800 | 0.000 | -0.800 | 175.000 |
| antmaze-giant-navigate-v0 | 5 | 4 | lowcond_full_nearestgoal | 0.800 | 0.000 | -0.800 | 175.000 |
| antmaze-giant-navigate-v0 | 5 | 4 | lowcond_full_nomask_nearestgoal | 0.800 | 0.000 | -0.800 | 175.000 |
| antmaze-giant-navigate-v0 | 5 | 4 | lowcond_full_rawdist_nearestgoal | 0.800 | 0.000 | -0.800 | 175.000 |
| antmaze-giant-navigate-v0 | 5 | 4 | lowcond_full_trajend | 0.800 | 0.000 | -0.800 | 175.000 |
| antmaze-giant-navigate-v0 | 5 | 4 | lowcond_tdr_only_local | 0.800 | 0.000 | -0.800 | 175.000 |
| antmaze-giant-navigate-v0 | 5 | 5 | lowcond_factor_only_nearestgoal | 0.400 | 0.000 | -0.400 | 200.400 |
| antmaze-giant-navigate-v0 | 5 | 5 | lowcond_full_localres | 0.400 | 0.200 | -0.200 | 177.000 |
| antmaze-giant-navigate-v0 | 5 | 5 | lowcond_full_nearestgoal | 0.400 | 0.400 | 0.000 | 88.200 |
| antmaze-giant-navigate-v0 | 5 | 5 | lowcond_full_nomask_nearestgoal | 0.400 | 0.200 | -0.200 | 159.400 |
| antmaze-giant-navigate-v0 | 5 | 5 | lowcond_full_rawdist_nearestgoal | 0.400 | 0.000 | -0.400 | 200.400 |
| antmaze-giant-navigate-v0 | 5 | 5 | lowcond_full_trajend | 0.400 | 0.400 | 0.000 | 85.200 |
| antmaze-giant-navigate-v0 | 5 | 5 | lowcond_tdr_only_local | 0.400 | 1.000 | 0.600 | -2.400 |
| antmaze-medium-navigate-v0 | 5 | 1 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 729.400 |
| antmaze-medium-navigate-v0 | 5 | 1 | lowcond_full_localres | 1.000 | 1.000 | 0.000 | 177.200 |
| antmaze-medium-navigate-v0 | 5 | 1 | lowcond_full_nearestgoal | 1.000 | 0.200 | -0.800 | 628.400 |
| antmaze-medium-navigate-v0 | 5 | 1 | lowcond_full_nomask_nearestgoal | 1.000 | 0.600 | -0.400 | 393.800 |
| antmaze-medium-navigate-v0 | 5 | 1 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.200 | -0.800 | 662.200 |
| antmaze-medium-navigate-v0 | 5 | 1 | lowcond_full_trajend | 1.000 | 0.600 | -0.400 | 454.800 |
| antmaze-medium-navigate-v0 | 5 | 1 | lowcond_tdr_only_local | 1.000 | 0.600 | -0.400 | 370.400 |
| antmaze-medium-navigate-v0 | 5 | 2 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 789.200 |
| antmaze-medium-navigate-v0 | 5 | 2 | lowcond_full_localres | 1.000 | 1.000 | 0.000 | 168.200 |
| antmaze-medium-navigate-v0 | 5 | 2 | lowcond_full_nearestgoal | 1.000 | 0.800 | -0.200 | 315.000 |
| antmaze-medium-navigate-v0 | 5 | 2 | lowcond_full_nomask_nearestgoal | 1.000 | 1.000 | 0.000 | 269.400 |
| antmaze-medium-navigate-v0 | 5 | 2 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.000 | -1.000 | 789.200 |
| antmaze-medium-navigate-v0 | 5 | 2 | lowcond_full_trajend | 1.000 | 0.400 | -0.600 | 497.200 |
| antmaze-medium-navigate-v0 | 5 | 2 | lowcond_tdr_only_local | 1.000 | 0.600 | -0.400 | 389.600 |
| antmaze-medium-navigate-v0 | 5 | 3 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 846.800 |
| antmaze-medium-navigate-v0 | 5 | 3 | lowcond_full_localres | 1.000 | 1.000 | 0.000 | 253.600 |
| antmaze-medium-navigate-v0 | 5 | 3 | lowcond_full_nearestgoal | 1.000 | 0.800 | -0.200 | 293.600 |
| antmaze-medium-navigate-v0 | 5 | 3 | lowcond_full_nomask_nearestgoal | 1.000 | 0.800 | -0.200 | 383.800 |
| antmaze-medium-navigate-v0 | 5 | 3 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.600 | -0.400 | 619.600 |
| antmaze-medium-navigate-v0 | 5 | 3 | lowcond_full_trajend | 1.000 | 0.800 | -0.200 | 308.800 |
| antmaze-medium-navigate-v0 | 5 | 3 | lowcond_tdr_only_local | 1.000 | 0.800 | -0.200 | 258.200 |
| antmaze-medium-navigate-v0 | 5 | 4 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 705.200 |
| antmaze-medium-navigate-v0 | 5 | 4 | lowcond_full_localres | 1.000 | 0.400 | -0.600 | 460.600 |
| antmaze-medium-navigate-v0 | 5 | 4 | lowcond_full_nearestgoal | 1.000 | 1.000 | 0.000 | 284.200 |
| antmaze-medium-navigate-v0 | 5 | 4 | lowcond_full_nomask_nearestgoal | 1.000 | 0.600 | -0.400 | 410.200 |
| antmaze-medium-navigate-v0 | 5 | 4 | lowcond_full_rawdist_nearestgoal | 1.000 | 1.000 | 0.000 | 234.800 |
| antmaze-medium-navigate-v0 | 5 | 4 | lowcond_full_trajend | 1.000 | 0.400 | -0.600 | 587.600 |
| antmaze-medium-navigate-v0 | 5 | 4 | lowcond_tdr_only_local | 1.000 | 0.800 | -0.200 | 273.000 |
| antmaze-medium-navigate-v0 | 5 | 5 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 786.000 |
| antmaze-medium-navigate-v0 | 5 | 5 | lowcond_full_localres | 1.000 | 1.000 | 0.000 | 192.800 |
| antmaze-medium-navigate-v0 | 5 | 5 | lowcond_full_nearestgoal | 1.000 | 0.800 | -0.200 | 442.400 |
| antmaze-medium-navigate-v0 | 5 | 5 | lowcond_full_nomask_nearestgoal | 1.000 | 0.800 | -0.200 | 318.800 |
| antmaze-medium-navigate-v0 | 5 | 5 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.800 | -0.200 | 453.400 |
| antmaze-medium-navigate-v0 | 5 | 5 | lowcond_full_trajend | 1.000 | 0.800 | -0.200 | 521.800 |
| antmaze-medium-navigate-v0 | 5 | 5 | lowcond_tdr_only_local | 1.000 | 1.000 | 0.000 | 122.600 |
| antmaze-medium-stitch-v0 | 5 | 1 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 708.000 |
| antmaze-medium-stitch-v0 | 5 | 1 | lowcond_full | 1.000 | 0.400 | -0.600 | 588.000 |
| antmaze-medium-stitch-v0 | 5 | 1 | lowcond_full_localres | 1.000 | 0.800 | -0.200 | 292.000 |
| antmaze-medium-stitch-v0 | 5 | 1 | lowcond_full_localres_from_full | 1.000 | 0.600 | -0.400 | 371.400 |
| antmaze-medium-stitch-v0 | 5 | 1 | lowcond_full_nearestgoal | 1.000 | 1.000 | 0.000 | 140.000 |
| antmaze-medium-stitch-v0 | 5 | 1 | lowcond_full_nomask_nearestgoal | 1.000 | 0.600 | -0.400 | 346.400 |
| antmaze-medium-stitch-v0 | 5 | 1 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.600 | -0.400 | 435.600 |
| antmaze-medium-stitch-v0 | 5 | 1 | lowcond_full_trajend | 1.000 | 1.000 | 0.000 | 181.200 |
| antmaze-medium-stitch-v0 | 5 | 1 | lowcond_tdr_only_local | 1.000 | 1.000 | 0.000 | 200.000 |
| antmaze-medium-stitch-v0 | 5 | 2 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 795.800 |
| antmaze-medium-stitch-v0 | 5 | 2 | lowcond_full | 1.000 | 0.400 | -0.600 | 622.400 |
| antmaze-medium-stitch-v0 | 5 | 2 | lowcond_full_localres | 1.000 | 1.000 | 0.000 | 164.000 |
| antmaze-medium-stitch-v0 | 5 | 2 | lowcond_full_localres_from_full | 1.000 | 1.000 | 0.000 | 114.600 |
| antmaze-medium-stitch-v0 | 5 | 2 | lowcond_full_nearestgoal | 1.000 | 0.800 | -0.200 | 267.200 |
| antmaze-medium-stitch-v0 | 5 | 2 | lowcond_full_nomask_nearestgoal | 1.000 | 1.000 | 0.000 | 109.400 |
| antmaze-medium-stitch-v0 | 5 | 2 | lowcond_full_rawdist_nearestgoal | 1.000 | 1.000 | 0.000 | 64.200 |
| antmaze-medium-stitch-v0 | 5 | 2 | lowcond_full_trajend | 1.000 | 1.000 | 0.000 | 138.400 |
| antmaze-medium-stitch-v0 | 5 | 2 | lowcond_tdr_only_local | 1.000 | 1.000 | 0.000 | 84.400 |
| antmaze-medium-stitch-v0 | 5 | 3 | lowcond_factor_only_nearestgoal | 1.000 | 0.400 | -0.600 | 738.400 |
| antmaze-medium-stitch-v0 | 5 | 3 | lowcond_full | 1.000 | 1.000 | 0.000 | 124.800 |
| antmaze-medium-stitch-v0 | 5 | 3 | lowcond_full_localres | 1.000 | 0.600 | -0.400 | 459.200 |
| antmaze-medium-stitch-v0 | 5 | 3 | lowcond_full_localres_from_full | 1.000 | 1.000 | 0.000 | 136.400 |
| antmaze-medium-stitch-v0 | 5 | 3 | lowcond_full_nearestgoal | 1.000 | 0.800 | -0.200 | 351.400 |
| antmaze-medium-stitch-v0 | 5 | 3 | lowcond_full_nomask_nearestgoal | 1.000 | 0.800 | -0.200 | 283.800 |
| antmaze-medium-stitch-v0 | 5 | 3 | lowcond_full_rawdist_nearestgoal | 1.000 | 1.000 | 0.000 | 94.800 |
| antmaze-medium-stitch-v0 | 5 | 3 | lowcond_full_trajend | 1.000 | 0.800 | -0.200 | 316.200 |
| antmaze-medium-stitch-v0 | 5 | 3 | lowcond_tdr_only_local | 1.000 | 1.000 | 0.000 | 134.800 |
| antmaze-medium-stitch-v0 | 5 | 4 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 726.400 |
| antmaze-medium-stitch-v0 | 5 | 4 | lowcond_full | 1.000 | 1.000 | 0.000 | 200.800 |
| antmaze-medium-stitch-v0 | 5 | 4 | lowcond_full_localres | 1.000 | 0.800 | -0.200 | 188.400 |
| antmaze-medium-stitch-v0 | 5 | 4 | lowcond_full_localres_from_full | 1.000 | 1.000 | 0.000 | -3.000 |
| antmaze-medium-stitch-v0 | 5 | 4 | lowcond_full_nearestgoal | 1.000 | 1.000 | 0.000 | 9.600 |
| antmaze-medium-stitch-v0 | 5 | 4 | lowcond_full_nomask_nearestgoal | 1.000 | 0.600 | -0.400 | 322.200 |
| antmaze-medium-stitch-v0 | 5 | 4 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.800 | -0.200 | 271.800 |
| antmaze-medium-stitch-v0 | 5 | 4 | lowcond_full_trajend | 1.000 | 0.000 | -1.000 | 726.400 |
| antmaze-medium-stitch-v0 | 5 | 4 | lowcond_tdr_only_local | 1.000 | 0.800 | -0.200 | 194.400 |
| antmaze-medium-stitch-v0 | 5 | 5 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 736.400 |
| antmaze-medium-stitch-v0 | 5 | 5 | lowcond_full | 1.000 | 0.800 | -0.200 | 288.000 |
| antmaze-medium-stitch-v0 | 5 | 5 | lowcond_full_localres | 1.000 | 0.800 | -0.200 | 322.800 |
| antmaze-medium-stitch-v0 | 5 | 5 | lowcond_full_localres_from_full | 1.000 | 1.000 | 0.000 | 166.000 |
| antmaze-medium-stitch-v0 | 5 | 5 | lowcond_full_nearestgoal | 1.000 | 0.800 | -0.200 | 178.200 |
| antmaze-medium-stitch-v0 | 5 | 5 | lowcond_full_nomask_nearestgoal | 1.000 | 0.800 | -0.200 | 270.200 |
| antmaze-medium-stitch-v0 | 5 | 5 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.800 | -0.200 | 299.800 |
| antmaze-medium-stitch-v0 | 5 | 5 | lowcond_full_trajend | 1.000 | 0.800 | -0.200 | 143.000 |
| antmaze-medium-stitch-v0 | 5 | 5 | lowcond_tdr_only_local | 1.000 | 1.000 | 0.000 | 177.400 |
