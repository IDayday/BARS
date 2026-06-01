# Stage 26 TMD/TDR Summary

Evaluation rows: 22450
Graph rows: 28

## Phase B Aggregate Comparisons

| env | run_episodes | weight | gas_n | gas_success_rate | variant_n | variant_success_rate | delta_success_rate | normal95_low | normal95_high | bootstrap95_low | bootstrap95_high | delta_mean_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| antmaze-giant-navigate-v0 | 20 | 0.1 | 1500 | 0.768 | 600 | 0.660 | -0.108 | -0.152 | -0.064 | -0.050 | 0.055 | 48.174 |
| antmaze-giant-navigate-v0 | 20 | 0.2 | 1500 | 0.768 | 600 | 0.700 | -0.068 | -0.110 | -0.026 | 0.000 | 0.082 | 34.876 |
| antmaze-giant-navigate-v0 | 20 | 0.25 | 1500 | 0.768 | 1500 | 0.813 | 0.045 | 0.016 | 0.074 | 0.018 | 0.075 | -10.331 |
| antmaze-giant-navigate-v0 | 20 | 0.3 | 1500 | 0.768 | 600 | 0.690 | -0.078 | -0.121 | -0.035 | -0.010 | 0.073 | 41.113 |
| antmaze-giant-navigate-v0 | 20 | 0.4 | 1500 | 0.768 | 600 | 0.690 | -0.078 | -0.121 | -0.035 | -0.015 | 0.080 | 42.704 |
| antmaze-giant-navigate-v0 | 20 | 0.5 | 1500 | 0.768 | 600 | 0.653 | -0.115 | -0.158 | -0.071 | -0.060 | 0.047 | 57.656 |
| antmaze-giant-navigate-v0 | 50 | 0.25 | 1500 | 0.677 | 1500 | 0.713 | 0.037 | 0.004 | 0.070 | 0.016 | 0.059 | -10.622 |
| antmaze-giant-stitch-v0 | 20 | 0.1 | 1500 | 0.878 | 600 | 0.867 | -0.011 | -0.043 | 0.021 | -0.033 | 0.037 | 2.019 |
| antmaze-giant-stitch-v0 | 20 | 0.2 | 1500 | 0.878 | 600 | 0.862 | -0.016 | -0.049 | 0.016 | -0.033 | 0.032 | 12.622 |
| antmaze-giant-stitch-v0 | 20 | 0.25 | 1500 | 0.878 | 1500 | 0.864 | -0.014 | -0.038 | 0.010 | -0.035 | 0.007 | 2.369 |
| antmaze-giant-stitch-v0 | 20 | 0.3 | 1500 | 0.878 | 600 | 0.848 | -0.030 | -0.063 | 0.003 | -0.060 | 0.022 | 10.934 |
| antmaze-giant-stitch-v0 | 20 | 0.4 | 1500 | 0.878 | 600 | 0.852 | -0.026 | -0.059 | 0.007 | -0.045 | 0.020 | 7.745 |
| antmaze-giant-stitch-v0 | 20 | 0.5 | 1500 | 0.878 | 600 | 0.873 | -0.005 | -0.036 | 0.027 | -0.028 | 0.045 | 3.270 |
| antmaze-large-navigate-v0 | 10 | 0.25 | 150 | 0.947 | 150 | 0.953 | 0.007 | -0.043 | 0.056 | -0.040 | 0.053 | -2.507 |
| antmaze-large-stitch-v0 | 10 | 0.25 | 150 | 0.960 | 150 | 0.960 | 0.000 | -0.044 | 0.044 | -0.033 | 0.040 | 2.967 |
| antmaze-medium-navigate-v0 | 20 | 0.1 | 1300 | 0.971 | 400 | 0.965 | -0.006 | -0.026 | 0.014 | -0.018 | 0.035 | 0.324 |
| antmaze-medium-navigate-v0 | 20 | 0.25 | 1300 | 0.971 | 1300 | 0.967 | -0.004 | -0.017 | 0.010 | -0.017 | 0.011 | 4.006 |
| antmaze-medium-stitch-v0 | 20 | 0.1 | 1300 | 0.968 | 400 | 0.985 | 0.017 | 0.002 | 0.033 | -0.013 | 0.030 | -16.836 |
| antmaze-medium-stitch-v0 | 20 | 0.25 | 1300 | 0.968 | 1300 | 0.962 | -0.006 | -0.020 | 0.008 | -0.018 | 0.007 | 5.080 |


## Phase B Task-Wise Deltas

| env | run_episodes | task_id | weight | gas_success_rate | variant_success_rate | delta_success_rate | delta_mean_steps |
|---|---|---|---|---|---|---|---|
| antmaze-giant-navigate-v0 | 20 | 1 | 0.1 | 0.437 | 0.217 | -0.220 | 36.748 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.2 | 0.437 | 0.175 | -0.262 | 37.590 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.25 | 0.437 | 0.490 | 0.053 | -5.527 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.3 | 0.437 | 0.142 | -0.295 | 35.407 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.4 | 0.437 | 0.225 | -0.212 | 32.998 |
| antmaze-giant-navigate-v0 | 20 | 1 | 0.5 | 0.437 | 0.175 | -0.262 | 40.232 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.1 | 0.887 | 0.875 | -0.012 | 25.497 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.2 | 0.887 | 0.883 | -0.003 | 19.680 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.25 | 0.887 | 0.937 | 0.050 | -11.103 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.3 | 0.887 | 0.933 | 0.047 | 1.222 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.4 | 0.887 | 0.892 | 0.005 | 20.147 |
| antmaze-giant-navigate-v0 | 20 | 2 | 0.5 | 0.887 | 0.850 | -0.037 | 39.280 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.1 | 0.823 | 0.708 | -0.115 | 35.185 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.2 | 0.823 | 0.833 | 0.010 | 12.085 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.25 | 0.823 | 0.880 | 0.057 | -9.630 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.3 | 0.823 | 0.750 | -0.073 | 36.527 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.4 | 0.823 | 0.708 | -0.115 | 70.343 |
| antmaze-giant-navigate-v0 | 20 | 3 | 0.5 | 0.823 | 0.617 | -0.207 | 100.627 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.1 | 0.870 | 0.742 | -0.128 | 58.370 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.2 | 0.870 | 0.825 | -0.045 | 46.203 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.25 | 0.870 | 0.897 | 0.027 | -8.023 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.3 | 0.870 | 0.833 | -0.037 | 48.128 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.4 | 0.870 | 0.842 | -0.028 | 20.937 |
| antmaze-giant-navigate-v0 | 20 | 4 | 0.5 | 0.870 | 0.850 | -0.020 | 25.303 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.1 | 0.823 | 0.758 | -0.065 | 85.072 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.2 | 0.823 | 0.783 | -0.040 | 58.822 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.25 | 0.823 | 0.863 | 0.040 | -17.373 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.3 | 0.823 | 0.792 | -0.032 | 84.280 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.4 | 0.823 | 0.783 | -0.040 | 69.097 |
| antmaze-giant-navigate-v0 | 20 | 5 | 0.5 | 0.823 | 0.775 | -0.048 | 82.838 |
| antmaze-giant-navigate-v0 | 50 | 1 | 0.25 | 0.180 | 0.233 | 0.053 | -7.873 |
| antmaze-giant-navigate-v0 | 50 | 2 | 0.25 | 0.877 | 0.903 | 0.027 | 3.663 |
| antmaze-giant-navigate-v0 | 50 | 3 | 0.25 | 0.780 | 0.833 | 0.053 | -17.813 |
| antmaze-giant-navigate-v0 | 50 | 4 | 0.25 | 0.787 | 0.820 | 0.033 | -21.807 |
| antmaze-giant-navigate-v0 | 50 | 5 | 0.25 | 0.760 | 0.777 | 0.017 | -9.280 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.1 | 0.803 | 0.708 | -0.095 | 14.365 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.2 | 0.803 | 0.808 | 0.005 | 6.232 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.25 | 0.803 | 0.723 | -0.080 | 26.753 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.3 | 0.803 | 0.692 | -0.112 | 27.015 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.4 | 0.803 | 0.742 | -0.062 | 9.682 |
| antmaze-giant-stitch-v0 | 20 | 1 | 0.5 | 0.803 | 0.758 | -0.045 | 12.707 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.1 | 0.940 | 0.908 | -0.032 | 7.898 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.2 | 0.940 | 0.917 | -0.023 | 13.473 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.25 | 0.940 | 0.927 | -0.013 | 3.183 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.3 | 0.940 | 0.908 | -0.032 | 11.123 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.4 | 0.940 | 0.892 | -0.048 | 22.707 |
| antmaze-giant-stitch-v0 | 20 | 2 | 0.5 | 0.940 | 0.917 | -0.023 | 9.215 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.1 | 0.803 | 0.842 | 0.038 | 6.248 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.2 | 0.803 | 0.817 | 0.013 | 3.965 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.25 | 0.803 | 0.867 | 0.063 | -23.130 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.3 | 0.803 | 0.825 | 0.022 | 2.748 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.4 | 0.803 | 0.808 | 0.005 | 5.882 |
| antmaze-giant-stitch-v0 | 20 | 3 | 0.5 | 0.803 | 0.833 | 0.030 | -1.285 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.1 | 0.913 | 0.925 | 0.012 | -4.050 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.2 | 0.913 | 0.867 | -0.047 | 24.633 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.25 | 0.913 | 0.897 | -0.017 | -1.550 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.3 | 0.913 | 0.908 | -0.005 | 6.442 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.4 | 0.913 | 0.883 | -0.030 | 2.075 |
| antmaze-giant-stitch-v0 | 20 | 4 | 0.5 | 0.913 | 0.942 | 0.028 | -7.867 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.1 | 0.930 | 0.950 | 0.020 | -14.368 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.2 | 0.930 | 0.900 | -0.030 | 14.807 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.25 | 0.930 | 0.907 | -0.023 | 6.590 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.3 | 0.930 | 0.908 | -0.022 | 7.340 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.4 | 0.930 | 0.933 | 0.003 | -1.618 |
| antmaze-giant-stitch-v0 | 20 | 5 | 0.5 | 0.930 | 0.917 | -0.013 | 3.582 |
| antmaze-large-navigate-v0 | 10 | 1 | 0.25 | 0.967 | 0.967 | 0.000 | 1.067 |
| antmaze-large-navigate-v0 | 10 | 2 | 0.25 | 0.900 | 0.900 | 0.000 | -17.500 |
| antmaze-large-navigate-v0 | 10 | 3 | 0.25 | 1.000 | 1.000 | 0.000 | 9.567 |
| antmaze-large-navigate-v0 | 10 | 4 | 0.25 | 0.900 | 0.933 | 0.033 | -14.133 |
| antmaze-large-navigate-v0 | 10 | 5 | 0.25 | 0.967 | 0.967 | 0.000 | 8.467 |
| antmaze-large-stitch-v0 | 10 | 1 | 0.25 | 0.933 | 0.967 | 0.033 | -14.300 |
| antmaze-large-stitch-v0 | 10 | 2 | 0.25 | 0.900 | 0.900 | 0.000 | 7.500 |
| antmaze-large-stitch-v0 | 10 | 3 | 0.25 | 1.000 | 1.000 | 0.000 | 1.133 |
| antmaze-large-stitch-v0 | 10 | 4 | 0.25 | 0.967 | 0.933 | -0.033 | 19.300 |
| antmaze-large-stitch-v0 | 10 | 5 | 0.25 | 1.000 | 1.000 | 0.000 | 1.200 |
| antmaze-medium-navigate-v0 | 20 | 1 | 0.1 | 0.962 | 0.925 | -0.037 | 19.132 |
| antmaze-medium-navigate-v0 | 20 | 1 | 0.25 | 0.962 | 0.923 | -0.038 | 29.262 |
| antmaze-medium-navigate-v0 | 20 | 2 | 0.1 | 0.985 | 0.963 | -0.022 | 19.131 |
| antmaze-medium-navigate-v0 | 20 | 2 | 0.25 | 0.985 | 0.977 | -0.008 | 11.777 |
| antmaze-medium-navigate-v0 | 20 | 3 | 0.1 | 0.958 | 0.950 | -0.008 | 5.305 |
| antmaze-medium-navigate-v0 | 20 | 3 | 0.25 | 0.958 | 0.969 | 0.012 | -13.619 |
| antmaze-medium-navigate-v0 | 20 | 4 | 0.1 | 0.962 | 1.000 | 0.038 | -34.344 |
| antmaze-medium-navigate-v0 | 20 | 4 | 0.25 | 0.962 | 0.981 | 0.019 | -15.108 |
| antmaze-medium-navigate-v0 | 20 | 5 | 0.1 | 0.988 | 0.988 | -0.001 | -7.601 |
| antmaze-medium-navigate-v0 | 20 | 5 | 0.25 | 0.988 | 0.985 | -0.004 | 7.719 |
| antmaze-medium-stitch-v0 | 20 | 1 | 0.1 | 0.958 | 0.975 | 0.017 | -18.176 |
| antmaze-medium-stitch-v0 | 20 | 1 | 0.25 | 0.958 | 0.962 | 0.004 | -9.427 |
| antmaze-medium-stitch-v0 | 20 | 2 | 0.1 | 0.985 | 1.000 | 0.015 | -35.071 |
| antmaze-medium-stitch-v0 | 20 | 2 | 0.25 | 0.985 | 0.981 | -0.004 | 5.612 |
| antmaze-medium-stitch-v0 | 20 | 3 | 0.1 | 0.969 | 1.000 | 0.031 | -25.931 |
| antmaze-medium-stitch-v0 | 20 | 3 | 0.25 | 0.969 | 0.973 | 0.004 | -3.946 |
| antmaze-medium-stitch-v0 | 20 | 4 | 0.1 | 0.946 | 0.988 | 0.041 | -26.257 |
| antmaze-medium-stitch-v0 | 20 | 4 | 0.25 | 0.946 | 0.923 | -0.023 | 20.373 |
| antmaze-medium-stitch-v0 | 20 | 5 | 0.1 | 0.981 | 0.963 | -0.018 | 21.253 |
| antmaze-medium-stitch-v0 | 20 | 5 | 0.25 | 0.981 | 0.969 | -0.012 | 12.788 |


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
| antmaze-giant-stitch-v0 | 5 | lowcond_factor_only_nearestgoal | 25 | 0.920 | 25 | 0.000 | -0.920 | -1.026 | -0.814 | -1.000 | -0.840 | 293.400 |
| antmaze-giant-stitch-v0 | 5 | lowcond_full_localres | 25 | 0.920 | 25 | 0.120 | -0.800 | -0.966 | -0.634 | -0.960 | -0.600 | 252.400 |
| antmaze-giant-stitch-v0 | 5 | lowcond_full_nearestgoal | 25 | 0.920 | 25 | 0.120 | -0.800 | -0.966 | -0.634 | -0.960 | -0.600 | 248.480 |
| antmaze-giant-stitch-v0 | 5 | lowcond_full_nomask_nearestgoal | 25 | 0.920 | 25 | 0.200 | -0.720 | -0.909 | -0.531 | -0.880 | -0.560 | 242.600 |
| antmaze-giant-stitch-v0 | 5 | lowcond_full_rawdist_nearestgoal | 25 | 0.920 | 25 | 0.120 | -0.800 | -0.966 | -0.634 | -0.960 | -0.600 | 259.040 |
| antmaze-giant-stitch-v0 | 5 | lowcond_full_trajend | 25 | 0.920 | 25 | 0.000 | -0.920 | -1.026 | -0.814 | -1.000 | -0.840 | 293.400 |
| antmaze-giant-stitch-v0 | 5 | lowcond_tdr_only_local | 25 | 0.920 | 25 | 0.200 | -0.720 | -0.909 | -0.531 | -0.880 | -0.560 | 252.840 |
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
| antmaze-giant-stitch-v0 | 5 | 1 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 153.400 |
| antmaze-giant-stitch-v0 | 5 | 1 | lowcond_full_localres | 1.000 | 0.000 | -1.000 | 153.400 |
| antmaze-giant-stitch-v0 | 5 | 1 | lowcond_full_nearestgoal | 1.000 | 0.000 | -1.000 | 153.400 |
| antmaze-giant-stitch-v0 | 5 | 1 | lowcond_full_nomask_nearestgoal | 1.000 | 0.000 | -1.000 | 153.400 |
| antmaze-giant-stitch-v0 | 5 | 1 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.000 | -1.000 | 153.400 |
| antmaze-giant-stitch-v0 | 5 | 1 | lowcond_full_trajend | 1.000 | 0.000 | -1.000 | 153.400 |
| antmaze-giant-stitch-v0 | 5 | 1 | lowcond_tdr_only_local | 1.000 | 0.000 | -1.000 | 153.400 |
| antmaze-giant-stitch-v0 | 5 | 2 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 314.200 |
| antmaze-giant-stitch-v0 | 5 | 2 | lowcond_full_localres | 1.000 | 0.000 | -1.000 | 314.200 |
| antmaze-giant-stitch-v0 | 5 | 2 | lowcond_full_nearestgoal | 1.000 | 0.000 | -1.000 | 314.200 |
| antmaze-giant-stitch-v0 | 5 | 2 | lowcond_full_nomask_nearestgoal | 1.000 | 0.400 | -0.600 | 295.600 |
| antmaze-giant-stitch-v0 | 5 | 2 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.000 | -1.000 | 314.200 |
| antmaze-giant-stitch-v0 | 5 | 2 | lowcond_full_trajend | 1.000 | 0.000 | -1.000 | 314.200 |
| antmaze-giant-stitch-v0 | 5 | 2 | lowcond_tdr_only_local | 1.000 | 0.400 | -0.600 | 277.800 |
| antmaze-giant-stitch-v0 | 5 | 3 | lowcond_factor_only_nearestgoal | 0.800 | 0.000 | -0.800 | 193.200 |
| antmaze-giant-stitch-v0 | 5 | 3 | lowcond_full_localres | 0.800 | 0.000 | -0.800 | 193.200 |
| antmaze-giant-stitch-v0 | 5 | 3 | lowcond_full_nearestgoal | 0.800 | 0.000 | -0.800 | 193.200 |
| antmaze-giant-stitch-v0 | 5 | 3 | lowcond_full_nomask_nearestgoal | 0.800 | 0.000 | -0.800 | 193.200 |
| antmaze-giant-stitch-v0 | 5 | 3 | lowcond_full_rawdist_nearestgoal | 0.800 | 0.000 | -0.800 | 193.200 |
| antmaze-giant-stitch-v0 | 5 | 3 | lowcond_full_trajend | 0.800 | 0.000 | -0.800 | 193.200 |
| antmaze-giant-stitch-v0 | 5 | 3 | lowcond_tdr_only_local | 0.800 | 0.000 | -0.800 | 193.200 |
| antmaze-giant-stitch-v0 | 5 | 4 | lowcond_factor_only_nearestgoal | 0.800 | 0.000 | -0.800 | 254.800 |
| antmaze-giant-stitch-v0 | 5 | 4 | lowcond_full_localres | 0.800 | 0.000 | -0.800 | 254.800 |
| antmaze-giant-stitch-v0 | 5 | 4 | lowcond_full_nearestgoal | 0.800 | 0.000 | -0.800 | 254.800 |
| antmaze-giant-stitch-v0 | 5 | 4 | lowcond_full_nomask_nearestgoal | 0.800 | 0.000 | -0.800 | 254.800 |
| antmaze-giant-stitch-v0 | 5 | 4 | lowcond_full_rawdist_nearestgoal | 0.800 | 0.000 | -0.800 | 254.800 |
| antmaze-giant-stitch-v0 | 5 | 4 | lowcond_full_trajend | 0.800 | 0.000 | -0.800 | 254.800 |
| antmaze-giant-stitch-v0 | 5 | 4 | lowcond_tdr_only_local | 0.800 | 0.000 | -0.800 | 254.800 |
| antmaze-giant-stitch-v0 | 5 | 5 | lowcond_factor_only_nearestgoal | 1.000 | 0.000 | -1.000 | 551.400 |
| antmaze-giant-stitch-v0 | 5 | 5 | lowcond_full_localres | 1.000 | 0.600 | -0.400 | 346.400 |
| antmaze-giant-stitch-v0 | 5 | 5 | lowcond_full_nearestgoal | 1.000 | 0.600 | -0.400 | 326.800 |
| antmaze-giant-stitch-v0 | 5 | 5 | lowcond_full_nomask_nearestgoal | 1.000 | 0.600 | -0.400 | 316.000 |
| antmaze-giant-stitch-v0 | 5 | 5 | lowcond_full_rawdist_nearestgoal | 1.000 | 0.600 | -0.400 | 379.600 |
| antmaze-giant-stitch-v0 | 5 | 5 | lowcond_full_trajend | 1.000 | 0.000 | -1.000 | 551.400 |
| antmaze-giant-stitch-v0 | 5 | 5 | lowcond_tdr_only_local | 1.000 | 0.600 | -0.400 | 385.000 |
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
