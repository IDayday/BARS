# tmd-test Evaluation Summary

Rows: 9102

## Run aggregate

| run_name | env | seed | gas_seed | mode | rows | success | success_rate | mean_steps |
|---|---|---|---|---|---|---|---|---|
| gas_direct_goal | antmaze-medium-stitch-v0 | 0 | 42 | gas_direct_goal | 2 | 2.0 | 1.000 | 419.0 |
| gas_direct_goal_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_direct_goal | 10 | 5.0 | 0.500 | 643.2 |
| gas_graph_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 10 | 9.0 | 0.900 | 296.4 |
| gas_graph_policy_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 50 | 48.0 | 0.960 | 259.1 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 50 | 27.0 | 0.540 | 841.0 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_policy | 50 | 39.0 | 0.780 | 764.8 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 50 | 47.0 | 0.940 | 438.4 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-stitch-v0 | 0 | 42 | gas_graph_policy | 50 | 45.0 | 0.900 | 434.0 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 50 | 36.0 | 0.720 | 755.8 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_policy | 50 | 39.0 | 0.780 | 735.9 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-large-stitch-v0 | 0 | 43 | gas_graph_policy | 50 | 48.0 | 0.960 | 419.3 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 100 | 54.0 | 0.540 | 844.3 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_policy | 100 | 59.0 | 0.590 | 850.3 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 100 | 96.0 | 0.960 | 264.4 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 100 | 78.0 | 0.780 | 742.0 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_policy | 100 | 66.0 | 0.660 | 777.0 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 100 | 99.0 | 0.990 | 239.6 |
| gas_graph_policy_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 250 | 242.0 | 0.968 | 257.7 |
| gas_graph_policy_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_policy | 250 | 247.0 | 0.988 | 235.3 |
| gas_graph_policy_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 100 | 96.0 | 0.960 | 195.6 |
| gas_graph_policy_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 1 | 42 | gas_graph_policy | 100 | 95.0 | 0.950 | 199.3 |
| gas_graph_policy_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 100 | 98.0 | 0.980 | 172.2 |
| gas_graph_policy_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 1 | 43 | gas_graph_policy | 100 | 97.0 | 0.970 | 178.4 |
| gas_graph_policy_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 50 | 48.0 | 0.960 | 201.1 |
| gas_graph_policy_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 50 | 48.0 | 0.960 | 189.5 |
| gas_graph_policy_tasks3_5_ep50_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 100 | 75.0 | 0.750 | 788.3 |
| gas_graph_policy_tasks3_5_ep50_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 100 | 83.0 | 0.830 | 647.6 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 38.0 | 0.760 | 762.7 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 40.0 | 0.800 | 726.4 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_cc_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 48.0 | 0.960 | 196.7 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 48.0 | 0.960 | 217.5 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 47.0 | 0.940 | 218.2 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 28.0 | 0.560 | 576.0 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 35.0 | 0.700 | 414.4 |
| gas_graph_tmd_exec_rescue_s125_p7_scale15_cc_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 48.0 | 0.960 | 189.7 |
| gas_graph_tmd_exec_rescue_s150_p9_scale15_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 100 | 97.0 | 0.970 | 191.2 |
| gas_graph_tmd_exec_rescue_s150_p9_scale15_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 100 | 96.0 | 0.960 | 186.6 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 100 | 97.0 | 0.970 | 191.4 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 1 | 42 | gas_graph_tmd_exec_rescue_policy | 100 | 95.0 | 0.950 | 199.2 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 100 | 99.0 | 0.990 | 162.3 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 1 | 43 | gas_graph_tmd_exec_rescue_policy | 100 | 97.0 | 0.970 | 179.5 |
| gas_graph_tmd_exec_rescue_s200_p15_scale10_cc_sticky25_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 46.0 | 0.920 | 376.8 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 49.0 | 0.980 | 246.4 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 250 | 244.0 | 0.976 | 255.5 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 250 | 249.0 | 0.996 | 229.8 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 250 | 243.0 | 0.972 | 255.6 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_cc_n512_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 47.0 | 0.940 | 354.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_cc_nosticky_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 46.0 | 0.920 | 384.9 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_cc_tmdactor_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 46.0 | 0.920 | 386.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_nocc_sticky25_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 46.0 | 0.920 | 361.2 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 48.0 | 0.960 | 199.8 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 48.0 | 0.960 | 186.7 |
| gas_graph_tmd_exec_rescue_s200_p15_scale20_cc_sticky25_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 46.0 | 0.920 | 370.0 |
| gas_graph_tmd_exec_rescue_s200_p6_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 49.0 | 0.980 | 186.9 |
| gas_graph_tmd_exec_rescue_s200_p6_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 48.0 | 0.960 | 187.9 |
| gas_graph_tmd_exec_rescue_s200_p7_scale15_cc_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 49.0 | 0.980 | 177.1 |
| gas_graph_tmd_exec_rescue_s200_p8_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 49.0 | 0.980 | 188.5 |
| gas_graph_tmd_exec_rescue_s200_p8_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 48.0 | 0.960 | 187.8 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 100 | 98.0 | 0.980 | 183.4 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 100 | 97.0 | 0.970 | 182.4 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 49.0 | 0.980 | 188.4 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 49.0 | 0.980 | 169.5 |
| gas_graph_tmd_exec_rescue_s250_p6_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 47.0 | 0.940 | 214.8 |
| gas_graph_tmd_exec_rescue_s250_p6_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 49.0 | 0.980 | 172.1 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 7.0 | 0.140 | 921.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 11.0 | 0.220 | 886.4 |
| gas_graph_tmd_exec_rescue_s300_p5_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 48.0 | 0.960 | 203.8 |
| gas_graph_tmd_exec_rescue_s300_p5_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 49.0 | 0.980 | 170.0 |
| gas_graph_tmd_exec_rescue_s400_p5_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 48.0 | 0.960 | 199.0 |
| gas_graph_tmd_exec_rescue_s400_p5_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 49.0 | 0.980 | 172.0 |
| gas_graph_tmd_exec_rescue_s50_p7_scale15_cc_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 41.0 | 0.820 | 409.8 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 39.0 | 0.780 | 760.3 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 38.0 | 0.760 | 738.4 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 35.0 | 0.700 | 794.6 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 40.0 | 0.800 | 728.1 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 50 | 37.0 | 0.740 | 771.5 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 50 | 41.0 | 0.820 | 726.4 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 100 | 61.0 | 0.610 | 828.6 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 100 | 63.0 | 0.630 | 846.8 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 100 | 86.0 | 0.860 | 709.8 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 100 | 82.0 | 0.820 | 732.4 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 50 | 29.0 | 0.580 | 851.2 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 50 | 41.0 | 0.820 | 721.6 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 100 | 55.0 | 0.550 | 858.2 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 100 | 55.0 | 0.550 | 877.9 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 100 | 80.0 | 0.800 | 733.4 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 100 | 78.0 | 0.780 | 741.5 |
| tmd100keff_exec_gasphi_cc_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 47.0 | 0.940 | 388.5 |
| tmd100keff_exec_gasphi_cc_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 46.0 | 0.920 | 415.5 |
| tmd100keff_exec_gasphi_cc_scale15_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 10.0 | 0.200 | 920.4 |
| tmd100keff_exec_gasphi_cc_scale15_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 3.0 | 0.060 | 964.7 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 15.0 | 0.300 | 906.2 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 10.0 | 0.200 | 980.7 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 5.0 | 0.100 | 970.4 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 23.0 | 0.460 | 882.7 |
| tmd150keff_teall_exec_gasphi_cc_scale15_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 47.0 | 0.940 | 248.0 |
| tmd150keff_teall_exec_gasphi_cc_scale15_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 49.0 | 0.980 | 221.3 |
| tmd150keff_teall_exec_gasphi_cc_scale18_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 46.0 | 0.920 | 287.4 |
| tmd150keff_teall_exec_gasphi_cc_scale18_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 46.0 | 0.920 | 251.4 |
| tmd200keff_teall_exec_gasphi_cc_scale15_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 46.0 | 0.920 | 315.4 |
| tmd200keff_teall_exec_gasphi_cc_scale15_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 44.0 | 0.880 | 336.3 |
| tmd50k_exec_gasphi_cc_final4_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 2.0 | 1.000 | 670.5 |
| tmd50k_exec_gasphi_cc_nosticky_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 1.0 | 0.500 | 599.5 |
| tmd50k_exec_gasphi_cc_scale10_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 1.0 | 0.020 | 998.9 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 25.0 | 0.500 | 702.2 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 100 | 57.0 | 0.570 | 623.4 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 100 | 57.0 | 0.570 | 618.9 |
| tmd50k_exec_gasphi_cc_scale15_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 2.0 | 1.000 | 220.0 |
| tmd50k_exec_gasphi_cc_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 46.0 | 0.920 | 336.3 |
| tmd50k_exec_gasphi_cc_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 47.0 | 0.940 | 301.5 |
| tmd50k_exec_gasphi_cc_scale15_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 46.0 | 0.920 | 410.6 |
| tmd50k_exec_gasphi_cc_scale15_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 50.0 | 1.000 | 301.8 |
| tmd50k_exec_gasphi_cc_scale16_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 46.0 | 0.920 | 338.5 |
| tmd50k_exec_gasphi_cc_scale17_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 47.0 | 0.940 | 316.9 |
| tmd50k_exec_gasphi_cc_scale18_task3_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 20 | 18.0 | 0.900 | 288.1 |
| tmd50k_exec_gasphi_cc_scale18_task3_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 20 | 19.0 | 0.950 | 301.6 |
| tmd50k_exec_gasphi_cc_scale18_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 49.0 | 0.980 | 251.6 |
| tmd50k_exec_gasphi_cc_scale18_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 49.0 | 0.980 | 230.8 |
| tmd50k_exec_gasphi_cc_scale19_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 47.0 | 0.940 | 308.4 |
| tmd50k_exec_gasphi_cc_scale20_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 2.0 | 1.000 | 235.0 |
| tmd50k_exec_gasphi_cc_scale20_task3_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 20 | 12.0 | 0.600 | 610.0 |
| tmd50k_exec_gasphi_cc_scale20_task3_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 20 | 14.0 | 0.700 | 486.2 |
| tmd50k_exec_gasphi_cc_scale20_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 47.0 | 0.940 | 287.3 |
| tmd50k_exec_gasphi_cc_scale21_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 48.0 | 0.960 | 242.8 |
| tmd50k_exec_gasphi_cc_scale22_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 48.0 | 0.960 | 251.2 |
| tmd50k_exec_gasphi_cc_scale25_task3_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 20 | 5.0 | 0.250 | 894.5 |
| tmd50k_exec_gasphi_cc_scale25_task3_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 20 | 8.0 | 0.400 | 812.6 |
| tmd50k_exec_gasphi_cc_scale25_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 41.0 | 0.820 | 446.3 |
| tmd50k_exec_gasphi_cc_scale25_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 39.0 | 0.780 | 456.5 |
| tmd50k_exec_gasphi_cc_scale30_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 1.0 | 0.020 | 988.1 |
| tmd50k_exec_gasphi_cc_scale30_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 0.0 | 0.000 | 1000.0 |
| tmd50k_exec_gasphi_cc_sticky25_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 1.0 | 0.500 | 735.5 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 4.0 | 0.080 | 991.7 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 2.0 | 0.040 | 991.9 |
| tmd50k_q75_t90_sticky25_tmd_graph_gas_policy_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 2 | 0.0 | 0.000 | 1000.0 |
| tmd50k_q75_t90_sticky25_tmd_graph_gas_policy_task3_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 10 | 0.0 | 0.000 | 1000.0 |
| tmd50k_q75_t90_sticky25_tmd_graph_tmd_actor_task3 | antmaze-medium-stitch-v0 | 0 |  | tmd_graph_tmd_actor | 2 | 0.0 | 0.000 | 1000.0 |
| tmd50k_q75_t90_tmd_graph_gas_policy_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 2 | 0.0 | 0.000 | 1000.0 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 21.0 | 0.420 | 887.2 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 21.0 | 0.420 | 882.0 |
| tmd50k_q98_exec_gasphi_cc_scale15_tasks3_5_ep50_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 100 | 73.0 | 0.730 | 769.0 |
| tmd50k_q98_exec_gasphi_cc_scale15_tasks3_5_ep50_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 100 | 62.0 | 0.620 | 812.2 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 50 | 0.0 | 0.000 | 1000.0 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 50 | 0.0 | 0.000 | 1000.0 |
| tmd50k_q98_graph_gas_policy_alltasks_ep2_gasseed42_gpu | antmaze-giant-navigate-v0 | 0 | 42 | tmd_graph_gas_policy | 10 | 0.0 | 0.000 | 1000.0 |
| tmd50k_q98_graph_tmd_actor_alltasks_ep2_gpu | antmaze-giant-navigate-v0 | 0 |  | tmd_graph_tmd_actor | 10 | 0.0 | 0.000 | 1000.0 |
| tmd_graph_gas_policy | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 2 | 2.0 | 1.000 | 651.5 |
| tmd_graph_gas_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 10 | 6.0 | 0.600 | 635.3 |
| tmd_graph_tmd_actor | antmaze-medium-stitch-v0 | 0 |  | tmd_graph_tmd_actor | 2 | 0.0 | 0.000 | 1000.0 |

## First rows

| run_name | env | seed | gas_seed | mode | task_id | episodes | success | steps | no_path_rate | goal_distance_improvement | subgoal_switch_count | final_goal_mode_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 1.0 | 487 | 0.0 | 38.283289452825414 | 0 | 76 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 1.0 | 389 | 0.0 | 36.99746279878675 | 0 | 30 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 1.0 | 356 | 0.0 | 38.87799510961611 | 0 | 27 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 1.0 | 384 | 0.0 | 36.535789674864084 | 0 | 24 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 1.0 | 467 | 0.0 | 37.65497683716214 | 0 | 29 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 1.0 | 507 | 0.0 | 38.84981722306094 | 0 | 106 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 1.0 | 373 | 0.0 | 37.65965497779699 | 0 | 52 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 1.0 | 360 | 0.0 | 37.3864062687289 | 0 | 26 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 0.0 | 1000 | 0.0 | 24.796393193195854 | 0 | 0 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 1 | 0.0 | 1000 | 0.0 | 38.48691987827925 | 0 | 584 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 1.0 | 523 | 0.0 | 9.2059561401685 | 0 | 24 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 1.0 | 459 | 0.0 | 9.598910950712433 | 0 | 19 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 1.0 | 612 | 0.0 | 10.210248239922823 | 0 | 24 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 1.0 | 552 | 0.0 | 11.077442792038237 | 0 | 36 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 0.0 | 1000 | 0.0 | 10.985084326592103 | 0 | 483 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 1.0 | 526 | 0.0 | 10.708280494120068 | 0 | 49 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 1.0 | 712 | 0.0 | 11.846218561829676 | 0 | 56 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 1.0 | 562 | 0.0 | 8.9613441269571 | 0 | 40 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 1.0 | 447 | 0.0 | 9.261703914013115 | 0 | 18 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 1 | 1.0 | 480 | 0.0 | 8.55225940229378 | 0 | 38 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 272 | 0.0 | 27.740967373478398 | 0 | 23 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 264 | 0.0 | 27.869252230415082 | 0 | 30 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 273 | 0.0 | 28.703813158433647 | 0 | 21 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 272 | 0.0 | 27.79557419455946 | 0 | 23 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 276 | 0.0 | 29.81014399771788 | 0 | 40 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 265 | 0.0 | 29.016999197318817 | 0 | 21 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 349 | 0.0 | 28.41563681485836 | 0 | 68 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 318 | 0.0 | 29.616449727234155 | 0 | 26 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 275 | 0.0 | 28.11334910634976 | 0 | 41 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 1 | 1.0 | 323 | 0.0 | 30.590242953265857 | 0 | 35 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 361 | 0.0 | 12.407430744361307 | 0 | 16 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 470 | 0.0 | 15.01531770536113 | 0 | 161 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 404 | 0.0 | 12.459672320607144 | 0 | 20 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 361 | 0.0 | 13.69977749434854 | 0 | 31 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 327 | 0.0 | 13.461397729963895 | 0 | 17 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 361 | 0.0 | 13.262559898273713 | 0 | 67 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 411 | 0.0 | 11.993506144346398 | 0 | 27 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 395 | 0.0 | 12.47970973208605 | 0 | 17 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 369 | 0.0 | 13.607290363341896 | 0 | 18 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 1 | 1.0 | 464 | 0.0 | 13.637420365336084 | 0 | 51 |
