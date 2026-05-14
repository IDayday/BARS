# Stage17 PU Retrain vs Loaded Baseline Comparison

- loaded_root: `runs_stage16_full12`
- pu_root: `runs_stage17_pu_retrain4`


## Run Status

| condition       | env                       | status    |   count |
|:----------------|:--------------------------|:----------|--------:|
| loaded_baseline | antmaze-large-diverse-v2  | completed |      36 |
| loaded_baseline | antmaze-large-diverse-v2  | running   |      27 |
| loaded_baseline | antmaze-large-diverse-v2  | started   |       3 |
| loaded_baseline | antmaze-large-play-v2     | completed |      36 |
| loaded_baseline | antmaze-large-play-v2     | running   |      27 |
| loaded_baseline | antmaze-large-play-v2     | started   |       3 |
| loaded_baseline | antmaze-medium-diverse-v2 | completed |      36 |
| loaded_baseline | antmaze-medium-diverse-v2 | running   |      27 |
| loaded_baseline | antmaze-medium-diverse-v2 | started   |       3 |
| loaded_baseline | antmaze-medium-play-v2    | completed |      36 |
| loaded_baseline | antmaze-medium-play-v2    | running   |      27 |
| loaded_baseline | antmaze-medium-play-v2    | started   |       3 |
| pu_retrain      | antmaze-large-diverse-v2  | completed |      10 |
| pu_retrain      | antmaze-large-diverse-v2  | running   |       8 |
| pu_retrain      | antmaze-large-diverse-v2  | started   |       1 |
| pu_retrain      | antmaze-large-play-v2     | completed |      10 |
| pu_retrain      | antmaze-large-play-v2     | running   |       8 |
| pu_retrain      | antmaze-large-play-v2     | started   |       1 |
| pu_retrain      | antmaze-medium-diverse-v2 | completed |      10 |
| pu_retrain      | antmaze-medium-diverse-v2 | running   |       8 |
| pu_retrain      | antmaze-medium-diverse-v2 | started   |       1 |
| pu_retrain      | antmaze-medium-play-v2    | completed |      10 |
| pu_retrain      | antmaze-medium-play-v2    | running   |       8 |
| pu_retrain      | antmaze-medium-play-v2    | started   |       1 |


## Balanced Edge Diagnostics

| condition       | env                       |   edge_auc_balanced |   edge_auprc_balanced |   supported_edge_rate |   selected_supported_rate |   selected_hard_neg_proxy_rate |   selected_unlabeled_bridge_rate |   score_supported_mean |   score_hard_neg_proxy_mean |   score_unlabeled_bridge_mean |   supported_minus_hard_selected |   supported_minus_hard_score |
|:----------------|:--------------------------|--------------------:|----------------------:|----------------------:|--------------------------:|-------------------------------:|---------------------------------:|-----------------------:|----------------------------:|------------------------------:|--------------------------------:|-----------------------------:|
| loaded_baseline | antmaze-large-diverse-v2  |            0.783091 |              0.780163 |              0.465    |                  0.448235 |                      0.0948917 |                         0.443675 |               0.434966 |                    0.114483 |                      0.430651 |                        0.353343 |                     0.320483 |
| loaded_baseline | antmaze-large-play-v2     |            0.771748 |              0.756358 |              0.468148 |                  0.448167 |                      0.110333  |                         0.445351 |               0.435347 |                    0.12739  |                      0.433007 |                        0.337834 |                     0.307957 |
| loaded_baseline | antmaze-medium-diverse-v2 |            0.774849 |              0.75933  |              0.480741 |                  0.539677 |                      0.158973  |                         0.53823  |               0.52141  |                    0.174434 |                      0.520254 |                        0.380704 |                     0.346976 |
| loaded_baseline | antmaze-medium-play-v2    |            0.778097 |              0.770801 |              0.471296 |                  0.522956 |                      0.137513  |                         0.525842 |               0.501007 |                    0.150858 |                      0.502825 |                        0.385443 |                     0.350149 |
| pu_retrain      | antmaze-large-diverse-v2  |            0.684687 |              0.704871 |              0.439444 |                  0.455752 |                      0.191309  |                         0.453668 |               0.504222 |                    0.371985 |                      0.502946 |                        0.264444 |                     0.132237 |
| pu_retrain      | antmaze-large-play-v2     |            0.660042 |              0.666957 |              0.456667 |                  0.518248 |                      0.259432  |                         0.517028 |               0.526592 |                    0.412262 |                      0.526688 |                        0.258817 |                     0.11433  |
| pu_retrain      | antmaze-medium-diverse-v2 |            0.711465 |              0.72131  |              0.468611 |                  0.632484 |                      0.337546  |                         0.629921 |               0.582197 |                    0.403466 |                      0.581325 |                        0.294938 |                     0.17873  |
| pu_retrain      | antmaze-medium-play-v2    |            0.718011 |              0.72737  |              0.473333 |                  0.612676 |                      0.291045  |                         0.618302 |               0.584008 |                    0.384652 |                      0.586613 |                        0.321631 |                     0.199355 |


## Edge Rollout Diagnostics

| condition       | env                       |   edge_rollout_auc |   edge_rollout_auprc |   success_rate |   selected_edge_success_rate |   unselected_edge_success_rate |   success_rate_selected_supported |   success_rate_selected_hard_neg_proxy |   success_rate_unselected_supported |   success_rate_unselected_hard_neg_proxy |   p_exec_mean_selected_supported |   p_exec_mean_selected_hard_neg_proxy |   final_dist_mean_selected_supported |   final_dist_mean_selected_hard_neg_proxy |   reset_available |   reset_ok_count |   reset_unavailable_count |   num_edges_eval |   selected_minus_unselected_success |   selected_supported_minus_hard_success |
|:----------------|:--------------------------|-------------------:|---------------------:|---------------:|-----------------------------:|-------------------------------:|----------------------------------:|---------------------------------------:|------------------------------------:|-----------------------------------------:|---------------------------------:|--------------------------------------:|-------------------------------------:|------------------------------------------:|------------------:|-----------------:|--------------------------:|-----------------:|------------------------------------:|----------------------------------------:|
| loaded_baseline | antmaze-large-diverse-v2  |           0.806552 |             0.657628 |       0.354167 |                     0.537401 |                       0.106815 |                          0.58415  |                               0.441468 |                            0.151001 |                                0.0692993 |                         0.819038 |                              0.755601 |                              1.55409 |                                   1.85219 |                 1 |              256 |                         0 |              256 |                            0.430586 |                                0.142681 |
| loaded_baseline | antmaze-large-play-v2     |           0.767124 |             0.604368 |       0.325521 |                     0.458203 |                       0.138827 |                          0.554418 |                               0.262738 |                            0.186369 |                                0.0908405 |                         0.814448 |                              0.749773 |                              1.64087 |                                   2.33232 |                 1 |              256 |                         0 |              256 |                            0.319376 |                                0.291679 |
| loaded_baseline | antmaze-medium-diverse-v2 |           0.775726 |             0.707567 |       0.428385 |                     0.601876 |                       0.193042 |                          0.666667 |                               0.473611 |                            0.230134 |                                0.155578  |                         0.860552 |                              0.794378 |                              1.28512 |                                   1.74763 |                 1 |              256 |                         0 |              256 |                            0.408834 |                                0.193056 |
| loaded_baseline | antmaze-medium-play-v2    |           0.764858 |             0.651448 |       0.390625 |                     0.546238 |                       0.174162 |                          0.593706 |                               0.448554 |                            0.233919 |                                0.12039   |                         0.838871 |                              0.799825 |                              1.42476 |                                   1.64805 |                 1 |              256 |                         0 |              256 |                            0.372076 |                                0.145151 |


## Path Diagnostics

| condition       | env                       | variant      |   lambda_risk |   found |   total_risk |   total_boundary |   total_cost |   objective |   num_edges |   num_subgoals |
|:----------------|:--------------------------|:-------------|--------------:|--------:|-------------:|-----------------:|-------------:|------------:|------------:|---------------:|
| loaded_baseline | antmaze-large-diverse-v2  | full_bars    |           0   |       1 |     2.0993   |         0.399234 |      1.27366 |     1.39343 |     1.50781 |       0.507812 |
| loaded_baseline | antmaze-large-diverse-v2  | full_bars    |           0.1 |       1 |     1.85882  |         0.394536 |      1.2888  |     1.59304 |     1.53646 |       0.536458 |
| loaded_baseline | antmaze-large-diverse-v2  | full_bars    |           0.3 |       1 |     1.57163  |         0.431747 |      1.3325  |     1.93351 |     1.61979 |       0.619792 |
| loaded_baseline | antmaze-large-diverse-v2  | full_bars    |           1   |       1 |     1.24992  |         0.587889 |      1.47014 |     2.89643 |     1.83073 |       0.830729 |
| loaded_baseline | antmaze-large-diverse-v2  | full_bars    |           3   |       1 |     1.05757  |         0.93786  |      1.69419 |     5.14827 |     2.14323 |       1.14323  |
| loaded_baseline | antmaze-large-diverse-v2  | reachability |           0   |       1 |     2.19345  |         0        |      1.25633 |     1.25633 |     1.49219 |       0.492188 |
| loaded_baseline | antmaze-large-diverse-v2  | reachability |           0.1 |       1 |     1.82694  |         0        |      1.27387 |     1.45656 |     1.52865 |       0.528646 |
| loaded_baseline | antmaze-large-diverse-v2  | reachability |           0.3 |       1 |     1.51903  |         0        |      1.33011 |     1.78582 |     1.625   |       0.625    |
| loaded_baseline | antmaze-large-diverse-v2  | reachability |           1   |       1 |     1.16774  |         0        |      1.51922 |     2.68696 |     1.90365 |       0.903646 |
| loaded_baseline | antmaze-large-diverse-v2  | reachability |           3   |       1 |     1.03382  |         0        |      1.73499 |     4.83646 |     2.21094 |       1.21094  |
| loaded_baseline | antmaze-large-diverse-v2  | shortest     |           0   |       1 |     2.19345  |         0        |      1.25633 |     1.25633 |     1.49219 |       0.492188 |
| loaded_baseline | antmaze-large-diverse-v2  | shortest     |           0.1 |       1 |     2.19345  |         0        |      1.25633 |     1.25633 |     1.49219 |       0.492188 |
| loaded_baseline | antmaze-large-diverse-v2  | shortest     |           0.3 |       1 |     2.19345  |         0        |      1.25633 |     1.25633 |     1.49219 |       0.492188 |
| loaded_baseline | antmaze-large-diverse-v2  | shortest     |           1   |       1 |     2.19345  |         0        |      1.25633 |     1.25633 |     1.49219 |       0.492188 |
| loaded_baseline | antmaze-large-diverse-v2  | shortest     |           3   |       1 |     2.19345  |         0        |      1.25633 |     1.25633 |     1.49219 |       0.492188 |
| loaded_baseline | antmaze-large-play-v2     | full_bars    |           0   |       1 |     2.42509  |         0.418997 |      1.26139 |     1.38709 |     1.50781 |       0.507812 |
| loaded_baseline | antmaze-large-play-v2     | full_bars    |           0.1 |       1 |     2.2364   |         0.430887 |      1.26525 |     1.61815 |     1.52083 |       0.520833 |
| loaded_baseline | antmaze-large-play-v2     | full_bars    |           0.3 |       1 |     1.86902  |         0.509586 |      1.31456 |     2.02814 |     1.60677 |       0.606771 |
| loaded_baseline | antmaze-large-play-v2     | full_bars    |           1   |       1 |     1.50848  |         0.685821 |      1.46844 |     3.18266 |     1.84896 |       0.848958 |
| loaded_baseline | antmaze-large-play-v2     | full_bars    |           3   |       1 |     1.31458  |         1.04259  |      1.66937 |     5.92589 |     2.1276  |       1.1276   |
| loaded_baseline | antmaze-large-play-v2     | reachability |           0   |       1 |     2.49072  |         0        |      1.24783 |     1.24783 |     1.5026  |       0.502604 |
| loaded_baseline | antmaze-large-play-v2     | reachability |           0.1 |       1 |     2.12276  |         0        |      1.26507 |     1.47734 |     1.53906 |       0.539062 |
| loaded_baseline | antmaze-large-play-v2     | reachability |           0.3 |       1 |     1.74073  |         0        |      1.33684 |     1.85906 |     1.64323 |       0.643229 |
| loaded_baseline | antmaze-large-play-v2     | reachability |           1   |       1 |     1.39781  |         0        |      1.53314 |     2.93095 |     1.9375  |       0.9375   |
| loaded_baseline | antmaze-large-play-v2     | reachability |           3   |       1 |     1.28398  |         0        |      1.72236 |     5.57431 |     2.20573 |       1.20573  |
| loaded_baseline | antmaze-large-play-v2     | shortest     |           0   |       1 |     2.49072  |         0        |      1.24783 |     1.24783 |     1.5026  |       0.502604 |
| loaded_baseline | antmaze-large-play-v2     | shortest     |           0.1 |       1 |     2.49072  |         0        |      1.24783 |     1.24783 |     1.5026  |       0.502604 |
| loaded_baseline | antmaze-large-play-v2     | shortest     |           0.3 |       1 |     2.49072  |         0        |      1.24783 |     1.24783 |     1.5026  |       0.502604 |
| loaded_baseline | antmaze-large-play-v2     | shortest     |           1   |       1 |     2.49072  |         0        |      1.24783 |     1.24783 |     1.5026  |       0.502604 |
| loaded_baseline | antmaze-large-play-v2     | shortest     |           3   |       1 |     2.49072  |         0        |      1.24783 |     1.24783 |     1.5026  |       0.502604 |
| loaded_baseline | antmaze-medium-diverse-v2 | full_bars    |           0   |       1 |     1.46703  |         0.406012 |      1.33294 |     1.45475 |     1.60156 |       0.601562 |
| loaded_baseline | antmaze-medium-diverse-v2 | full_bars    |           0.1 |       1 |     1.32223  |         0.392093 |      1.34335 |     1.5932  |     1.6224  |       0.622396 |
| loaded_baseline | antmaze-medium-diverse-v2 | full_bars    |           0.3 |       1 |     1.20422  |         0.424611 |      1.35554 |     1.84419 |     1.64062 |       0.640625 |
| loaded_baseline | antmaze-medium-diverse-v2 | full_bars    |           1   |       1 |     0.951987 |         0.557157 |      1.46114 |     2.58027 |     1.79167 |       0.791667 |
| loaded_baseline | antmaze-medium-diverse-v2 | full_bars    |           3   |       1 |     0.811251 |         0.818724 |      1.63145 |     4.31082 |     2.04688 |       1.04688  |
| loaded_baseline | antmaze-medium-diverse-v2 | reachability |           0   |       1 |     1.46094  |         0        |      1.32168 |     1.32168 |     1.59375 |       0.59375  |
| loaded_baseline | antmaze-medium-diverse-v2 | reachability |           0.1 |       1 |     1.25064  |         0        |      1.33238 |     1.45744 |     1.60938 |       0.609375 |
| loaded_baseline | antmaze-medium-diverse-v2 | reachability |           0.3 |       1 |     1.14218  |         0        |      1.35748 |     1.70014 |     1.65365 |       0.653646 |
| loaded_baseline | antmaze-medium-diverse-v2 | reachability |           1   |       1 |     0.89986  |         0        |      1.48251 |     2.38237 |     1.82031 |       0.820312 |
| loaded_baseline | antmaze-medium-diverse-v2 | reachability |           3   |       1 |     0.792575 |         0        |      1.66141 |     4.03914 |     2.08854 |       1.08854  |
| loaded_baseline | antmaze-medium-diverse-v2 | shortest     |           0   |       1 |     1.46094  |         0        |      1.32168 |     1.32168 |     1.59375 |       0.59375  |
| loaded_baseline | antmaze-medium-diverse-v2 | shortest     |           0.1 |       1 |     1.46094  |         0        |      1.32168 |     1.32168 |     1.59375 |       0.59375  |
| loaded_baseline | antmaze-medium-diverse-v2 | shortest     |           0.3 |       1 |     1.46094  |         0        |      1.32168 |     1.32168 |     1.59375 |       0.59375  |
| loaded_baseline | antmaze-medium-diverse-v2 | shortest     |           1   |       1 |     1.46094  |         0        |      1.32168 |     1.32168 |     1.59375 |       0.59375  |
| loaded_baseline | antmaze-medium-diverse-v2 | shortest     |           3   |       1 |     1.46094  |         0        |      1.32168 |     1.32168 |     1.59375 |       0.59375  |
| loaded_baseline | antmaze-medium-play-v2    | full_bars    |           0   |       1 |     1.49489  |         0.468223 |      1.40226 |     1.54273 |     1.70312 |       0.703125 |
| loaded_baseline | antmaze-medium-play-v2    | full_bars    |           0.1 |       1 |     1.41715  |         0.4636   |      1.40624 |     1.68703 |     1.71354 |       0.713542 |
| loaded_baseline | antmaze-medium-play-v2    | full_bars    |           0.3 |       1 |     1.2286   |         0.474271 |      1.44086 |     1.95172 |     1.77865 |       0.778646 |
| loaded_baseline | antmaze-medium-play-v2    | full_bars    |           1   |       1 |     1.01005  |         0.630985 |      1.52472 |     2.72406 |     1.90625 |       0.90625  |
| loaded_baseline | antmaze-medium-play-v2    | full_bars    |           3   |       1 |     0.870076 |         0.817356 |      1.70237 |     4.5578  |     2.13802 |       1.13802  |
| loaded_baseline | antmaze-medium-play-v2    | reachability |           0   |       1 |     1.58702  |         0        |      1.38414 |     1.38414 |     1.67708 |       0.677083 |
| loaded_baseline | antmaze-medium-play-v2    | reachability |           0.1 |       1 |     1.37987  |         0        |      1.39148 |     1.52947 |     1.70312 |       0.703125 |
| loaded_baseline | antmaze-medium-play-v2    | reachability |           0.3 |       1 |     1.17067  |         0        |      1.43036 |     1.78156 |     1.77604 |       0.776042 |
| loaded_baseline | antmaze-medium-play-v2    | reachability |           1   |       1 |     0.965284 |         0        |      1.54744 |     2.51273 |     1.94792 |       0.947917 |
| loaded_baseline | antmaze-medium-play-v2    | reachability |           3   |       1 |     0.849248 |         0        |      1.73319 |     4.28093 |     2.17708 |       1.17708  |
| loaded_baseline | antmaze-medium-play-v2    | shortest     |           0   |       1 |     1.58702  |         0        |      1.38414 |     1.38414 |     1.67708 |       0.677083 |
| loaded_baseline | antmaze-medium-play-v2    | shortest     |           0.1 |       1 |     1.58702  |         0        |      1.38414 |     1.38414 |     1.67708 |       0.677083 |
| loaded_baseline | antmaze-medium-play-v2    | shortest     |           0.3 |       1 |     1.58702  |         0        |      1.38414 |     1.38414 |     1.67708 |       0.677083 |
| loaded_baseline | antmaze-medium-play-v2    | shortest     |           1   |       1 |     1.58702  |         0        |      1.38414 |     1.38414 |     1.67708 |       0.677083 |
| loaded_baseline | antmaze-medium-play-v2    | shortest     |           3   |       1 |     1.58702  |         0        |      1.38414 |     1.38414 |     1.67708 |       0.677083 |
| pu_retrain      | antmaze-large-diverse-v2  | full_bars    |           0   |       1 |     1.40702  |         0.637906 |      1.48908 |     1.68045 |     1.875   |       0.875    |
| pu_retrain      | antmaze-large-diverse-v2  | full_bars    |           0.1 |       1 |     1.37296  |         0.654945 |      1.48603 |     1.81981 |     1.85938 |       0.859375 |
| pu_retrain      | antmaze-large-diverse-v2  | full_bars    |           0.3 |       1 |     1.32486  |         0.656316 |      1.49418 |     2.08854 |     1.88281 |       0.882812 |
| pu_retrain      | antmaze-large-diverse-v2  | full_bars    |           1   |       1 |     1.21767  |         0.737797 |      1.52944 |     2.96845 |     1.96094 |       0.960938 |
| pu_retrain      | antmaze-large-diverse-v2  | full_bars    |           3   |       1 |     1.11943  |         0.924477 |      1.62736 |     5.26299 |     2.15625 |       1.15625  |
| pu_retrain      | antmaze-large-diverse-v2  | reachability |           0   |       1 |     1.47708  |         0        |      1.46325 |     1.46325 |     1.82812 |       0.828125 |
| pu_retrain      | antmaze-large-diverse-v2  | reachability |           0.1 |       1 |     1.36869  |         0        |      1.46806 |     1.60493 |     1.84375 |       0.84375  |
| pu_retrain      | antmaze-large-diverse-v2  | reachability |           0.3 |       1 |     1.28639  |         0        |      1.48483 |     1.87075 |     1.88281 |       0.882812 |
| pu_retrain      | antmaze-large-diverse-v2  | reachability |           1   |       1 |     1.16893  |         0        |      1.55405 |     2.72298 |     2       |       1        |
| pu_retrain      | antmaze-large-diverse-v2  | reachability |           3   |       1 |     1.10558  |         0        |      1.65183 |     4.96857 |     2.20312 |       1.20312  |
| pu_retrain      | antmaze-large-diverse-v2  | shortest     |           0   |       1 |     1.47708  |         0        |      1.46325 |     1.46325 |     1.82812 |       0.828125 |
| pu_retrain      | antmaze-large-diverse-v2  | shortest     |           0.1 |       1 |     1.47708  |         0        |      1.46325 |     1.46325 |     1.82812 |       0.828125 |
| pu_retrain      | antmaze-large-diverse-v2  | shortest     |           0.3 |       1 |     1.47708  |         0        |      1.46325 |     1.46325 |     1.82812 |       0.828125 |
| pu_retrain      | antmaze-large-diverse-v2  | shortest     |           1   |       1 |     1.47708  |         0        |      1.46325 |     1.46325 |     1.82812 |       0.828125 |
| pu_retrain      | antmaze-large-diverse-v2  | shortest     |           3   |       1 |     1.47708  |         0        |      1.46325 |     1.46325 |     1.82812 |       0.828125 |
| pu_retrain      | antmaze-large-play-v2     | full_bars    |           0   |       1 |     1.32755  |         0.652871 |      1.36314 |     1.559   |     1.75781 |       0.757812 |
| pu_retrain      | antmaze-large-play-v2     | full_bars    |           0.1 |       1 |     1.30666  |         0.65547  |      1.36375 |     1.69106 |     1.76562 |       0.765625 |
| pu_retrain      | antmaze-large-play-v2     | full_bars    |           0.3 |       1 |     1.27322  |         0.651946 |      1.37009 |     1.94764 |     1.78125 |       0.78125  |
| pu_retrain      | antmaze-large-play-v2     | full_bars    |           1   |       1 |     1.214    |         0.756937 |      1.38465 |     2.82573 |     1.8125  |       0.8125   |
| pu_retrain      | antmaze-large-play-v2     | full_bars    |           3   |       1 |     1.14544  |         0.871792 |      1.45804 |     5.1559  |     1.9375  |       0.9375   |
| pu_retrain      | antmaze-large-play-v2     | reachability |           0   |       1 |     1.30925  |         0        |      1.34024 |     1.34024 |     1.73438 |       0.734375 |
| pu_retrain      | antmaze-large-play-v2     | reachability |           0.1 |       1 |     1.28971  |         0        |      1.34077 |     1.46974 |     1.73438 |       0.734375 |
| pu_retrain      | antmaze-large-play-v2     | reachability |           0.3 |       1 |     1.24374  |         0        |      1.35058 |     1.72371 |     1.75781 |       0.757812 |
| pu_retrain      | antmaze-large-play-v2     | reachability |           1   |       1 |     1.16777  |         0        |      1.40315 |     2.57093 |     1.84375 |       0.84375  |
| pu_retrain      | antmaze-large-play-v2     | reachability |           3   |       1 |     1.12201  |         0        |      1.48957 |     4.8556  |     1.96875 |       0.96875  |
| pu_retrain      | antmaze-large-play-v2     | shortest     |           0   |       1 |     1.30925  |         0        |      1.34024 |     1.34024 |     1.73438 |       0.734375 |
| pu_retrain      | antmaze-large-play-v2     | shortest     |           0.1 |       1 |     1.30925  |         0        |      1.34024 |     1.34024 |     1.73438 |       0.734375 |
| pu_retrain      | antmaze-large-play-v2     | shortest     |           0.3 |       1 |     1.30925  |         0        |      1.34024 |     1.34024 |     1.73438 |       0.734375 |
| pu_retrain      | antmaze-large-play-v2     | shortest     |           1   |       1 |     1.30925  |         0        |      1.34024 |     1.34024 |     1.73438 |       0.734375 |
| pu_retrain      | antmaze-large-play-v2     | shortest     |           3   |       1 |     1.30925  |         0        |      1.34024 |     1.34024 |     1.73438 |       0.734375 |
| pu_retrain      | antmaze-medium-diverse-v2 | full_bars    |           0   |       1 |     1.0607   |         0.523259 |      1.39234 |     1.54931 |     1.6875  |       0.6875   |
| pu_retrain      | antmaze-medium-diverse-v2 | full_bars    |           0.1 |       1 |     0.964525 |         0.498798 |      1.40246 |     1.64855 |     1.70312 |       0.703125 |
| pu_retrain      | antmaze-medium-diverse-v2 | full_bars    |           0.3 |       1 |     0.893087 |         0.484324 |      1.41958 |     1.83281 |     1.74219 |       0.742188 |
| pu_retrain      | antmaze-medium-diverse-v2 | full_bars    |           1   |       1 |     0.824659 |         0.544098 |      1.45028 |     2.43816 |     1.78906 |       0.789062 |
| pu_retrain      | antmaze-medium-diverse-v2 | full_bars    |           3   |       1 |     0.76747  |         0.689327 |      1.5138  |     4.02301 |     1.86719 |       0.867188 |
| pu_retrain      | antmaze-medium-diverse-v2 | reachability |           0   |       1 |     1.0158   |         0        |      1.3831  |     1.3831  |     1.6875  |       0.6875   |
| pu_retrain      | antmaze-medium-diverse-v2 | reachability |           0.1 |       1 |     1.00072  |         0        |      1.38365 |     1.48372 |     1.69531 |       0.695312 |
| pu_retrain      | antmaze-medium-diverse-v2 | reachability |           0.3 |       1 |     0.896911 |         0        |      1.40174 |     1.67081 |     1.74219 |       0.742188 |
| pu_retrain      | antmaze-medium-diverse-v2 | reachability |           1   |       1 |     0.792091 |         0        |      1.4681  |     2.2602  |     1.8125  |       0.8125   |
| pu_retrain      | antmaze-medium-diverse-v2 | reachability |           3   |       1 |     0.736238 |         0        |      1.57897 |     3.78768 |     1.96875 |       0.96875  |
| pu_retrain      | antmaze-medium-diverse-v2 | shortest     |           0   |       1 |     1.0158   |         0        |      1.3831  |     1.3831  |     1.6875  |       0.6875   |
| pu_retrain      | antmaze-medium-diverse-v2 | shortest     |           0.1 |       1 |     1.0158   |         0        |      1.3831  |     1.3831  |     1.6875  |       0.6875   |
| pu_retrain      | antmaze-medium-diverse-v2 | shortest     |           0.3 |       1 |     1.0158   |         0        |      1.3831  |     1.3831  |     1.6875  |       0.6875   |
| pu_retrain      | antmaze-medium-diverse-v2 | shortest     |           1   |       1 |     1.0158   |         0        |      1.3831  |     1.3831  |     1.6875  |       0.6875   |
| pu_retrain      | antmaze-medium-diverse-v2 | shortest     |           3   |       1 |     1.0158   |         0        |      1.3831  |     1.3831  |     1.6875  |       0.6875   |
| pu_retrain      | antmaze-medium-play-v2    | full_bars    |           0   |       1 |     1.10902  |         0.597541 |      1.48343 |     1.66269 |     1.89062 |       0.890625 |
| pu_retrain      | antmaze-medium-play-v2    | full_bars    |           0.1 |       1 |     1.05211  |         0.600941 |      1.4856  |     1.7711  |     1.90625 |       0.90625  |
| pu_retrain      | antmaze-medium-play-v2    | full_bars    |           0.3 |       1 |     0.982981 |         0.594358 |      1.49978 |     1.97298 |     1.92188 |       0.921875 |
| pu_retrain      | antmaze-medium-play-v2    | full_bars    |           1   |       1 |     0.875552 |         0.654984 |      1.54989 |     2.62193 |     2       |       1        |
| pu_retrain      | antmaze-medium-play-v2    | full_bars    |           3   |       1 |     0.805682 |         0.731127 |      1.66184 |     4.29822 |     2.14844 |       1.14844  |
| pu_retrain      | antmaze-medium-play-v2    | reachability |           0   |       1 |     1.14153  |         0        |      1.45956 |     1.45956 |     1.84375 |       0.84375  |
| pu_retrain      | antmaze-medium-play-v2    | reachability |           0.1 |       1 |     1.10003  |         0        |      1.46217 |     1.57218 |     1.85156 |       0.851562 |
| pu_retrain      | antmaze-medium-play-v2    | reachability |           0.3 |       1 |     0.995241 |         0        |      1.48041 |     1.77898 |     1.88281 |       0.882812 |
| pu_retrain      | antmaze-medium-play-v2    | reachability |           1   |       1 |     0.854192 |         0        |      1.55801 |     2.4122  |     2       |       1        |
| pu_retrain      | antmaze-medium-play-v2    | reachability |           3   |       1 |     0.782108 |         0        |      1.68979 |     4.03611 |     2.14062 |       1.14062  |
| pu_retrain      | antmaze-medium-play-v2    | shortest     |           0   |       1 |     1.14153  |         0        |      1.45956 |     1.45956 |     1.84375 |       0.84375  |
| pu_retrain      | antmaze-medium-play-v2    | shortest     |           0.1 |       1 |     1.14153  |         0        |      1.45956 |     1.45956 |     1.84375 |       0.84375  |
| pu_retrain      | antmaze-medium-play-v2    | shortest     |           0.3 |       1 |     1.14153  |         0        |      1.45956 |     1.45956 |     1.84375 |       0.84375  |
| pu_retrain      | antmaze-medium-play-v2    | shortest     |           1   |       1 |     1.14153  |         0        |      1.45956 |     1.45956 |     1.84375 |       0.84375  |
| pu_retrain      | antmaze-medium-play-v2    | shortest     |           3   |       1 |     1.14153  |         0        |      1.45956 |     1.45956 |     1.84375 |       0.84375  |


## Edge Proxy Diagnostics

| condition       | env                       |   reach_auc_proxy |   reach_auprc_proxy |   cross_traj_selected_rate |   reachable_edge_coverage_proxy |   selected_edges |   num_edges |
|:----------------|:--------------------------|------------------:|--------------------:|---------------------------:|--------------------------------:|-----------------:|------------:|
| loaded_baseline | antmaze-large-diverse-v2  |          0.885504 |         0.0236295   |                   0.970822 |                        0.888889 |          936.667 |        3600 |
| loaded_baseline | antmaze-large-play-v2     |          0.741054 |         0.00586144  |                   0.975437 |                        0.5      |          971.667 |        3600 |
| loaded_baseline | antmaze-medium-diverse-v2 |          0.863305 |         0.00869441  |                   0.975139 |                        1        |         1236     |        3600 |
| loaded_baseline | antmaze-medium-play-v2    |          0.801617 |         0.00607699  |                   0.984243 |                        0.777778 |         1150     |        3600 |
| pu_retrain      | antmaze-large-diverse-v2  |          0.841676 |         0.0613423   |                   0.98018  |                        0.857143 |         1110     |        3600 |
| pu_retrain      | antmaze-large-play-v2     |          0.582465 |         0.0124857   |                   0.981645 |                        0.25     |         1362     |        3600 |
| pu_retrain      | antmaze-medium-diverse-v2 |          0.85662  |         0.0038737   |                   0.980175 |                        1        |         1715     |        3600 |
| pu_retrain      | antmaze-medium-play-v2    |          0.617755 |         0.000739098 |                   0.984972 |                        1        |         1597     |        3600 |


## Graph Summary

_No data._


## Profile Summary

| condition       | env                       | phase       | event                    |   duration_sec |
|:----------------|:--------------------------|:------------|:-------------------------|---------------:|
| loaded_baseline | antmaze-large-diverse-v2  | diagnostics | boundary_end             |     1.33275    |
| loaded_baseline | antmaze-large-diverse-v2  | diagnostics | boundary_start           |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | diagnostics | edge_end                 |     1.06083    |
| loaded_baseline | antmaze-large-diverse-v2  | diagnostics | edge_start               |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | diagnostics | path_end                 |    33.5509     |
| loaded_baseline | antmaze-large-diverse-v2  | diagnostics | path_start               |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | build_boundary_end       |     3.31435    |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | build_boundary_start     |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | build_edges_end          |     0.73345    |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | build_edges_start        |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | save_boundary_end        |     0.0164161  |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | save_boundary_start      |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | save_graph_end           |     0.00502539 |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | save_graph_start         |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | select_nodes_end         |     1.53678    |
| loaded_baseline | antmaze-large-diverse-v2  | graph_build | select_nodes_start       |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | embed_dataset_end        |     0.0233702  |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | embed_dataset_start      |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | load_dataset_end         |     3.27142    |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | load_dataset_start       |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | train_policy_end         |     0.00420252 |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | train_policy_start       |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | train_reachability_end   |     0.00340549 |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | train_reachability_start |   nan          |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | train_tdr_end            |     1.2585     |
| loaded_baseline | antmaze-large-diverse-v2  | pipeline    | train_tdr_start          |   nan          |
| loaded_baseline | antmaze-large-play-v2     | diagnostics | boundary_end             |     1.32775    |
| loaded_baseline | antmaze-large-play-v2     | diagnostics | boundary_start           |   nan          |
| loaded_baseline | antmaze-large-play-v2     | diagnostics | edge_end                 |     1.05097    |
| loaded_baseline | antmaze-large-play-v2     | diagnostics | edge_start               |   nan          |
| loaded_baseline | antmaze-large-play-v2     | diagnostics | path_end                 |    37.4729     |
| loaded_baseline | antmaze-large-play-v2     | diagnostics | path_start               |   nan          |
| loaded_baseline | antmaze-large-play-v2     | graph_build | build_boundary_end       |     3.27746    |
| loaded_baseline | antmaze-large-play-v2     | graph_build | build_boundary_start     |   nan          |
| loaded_baseline | antmaze-large-play-v2     | graph_build | build_edges_end          |     0.744453   |
| loaded_baseline | antmaze-large-play-v2     | graph_build | build_edges_start        |   nan          |
| loaded_baseline | antmaze-large-play-v2     | graph_build | save_boundary_end        |     0.0163249  |
| loaded_baseline | antmaze-large-play-v2     | graph_build | save_boundary_start      |   nan          |
| loaded_baseline | antmaze-large-play-v2     | graph_build | save_graph_end           |     0.00503643 |
| loaded_baseline | antmaze-large-play-v2     | graph_build | save_graph_start         |   nan          |
| loaded_baseline | antmaze-large-play-v2     | graph_build | select_nodes_end         |     1.52396    |
| loaded_baseline | antmaze-large-play-v2     | graph_build | select_nodes_start       |   nan          |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | embed_dataset_end        |     0.0210145  |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | embed_dataset_start      |   nan          |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | load_dataset_end         |     3.17604    |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | load_dataset_start       |   nan          |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | train_policy_end         |     0.00422462 |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | train_policy_start       |   nan          |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | train_reachability_end   |     0.00342194 |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | train_reachability_start |   nan          |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | train_tdr_end            |     1.27503    |
| loaded_baseline | antmaze-large-play-v2     | pipeline    | train_tdr_start          |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | diagnostics | boundary_end             |     1.35424    |
| loaded_baseline | antmaze-medium-diverse-v2 | diagnostics | boundary_start           |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | diagnostics | edge_end                 |     1.06144    |
| loaded_baseline | antmaze-medium-diverse-v2 | diagnostics | edge_start               |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | diagnostics | path_end                 |    38.9274     |
| loaded_baseline | antmaze-medium-diverse-v2 | diagnostics | path_start               |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | build_boundary_end       |     3.32975    |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | build_boundary_start     |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | build_edges_end          |     0.747691   |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | build_edges_start        |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | save_boundary_end        |     0.0164775  |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | save_boundary_start      |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | save_graph_end           |     0.00487622 |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | save_graph_start         |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | select_nodes_end         |     1.53024    |
| loaded_baseline | antmaze-medium-diverse-v2 | graph_build | select_nodes_start       |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | embed_dataset_end        |     0.0189268  |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | embed_dataset_start      |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | load_dataset_end         |     3.08886    |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | load_dataset_start       |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | train_policy_end         |     0.00423924 |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | train_policy_start       |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | train_reachability_end   |     0.0046134  |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | train_reachability_start |   nan          |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | train_tdr_end            |     1.28727    |
| loaded_baseline | antmaze-medium-diverse-v2 | pipeline    | train_tdr_start          |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | diagnostics | boundary_end             |     1.33348    |
| loaded_baseline | antmaze-medium-play-v2    | diagnostics | boundary_start           |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | diagnostics | edge_end                 |     1.04555    |
| loaded_baseline | antmaze-medium-play-v2    | diagnostics | edge_start               |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | diagnostics | path_end                 |    49.2696     |
| loaded_baseline | antmaze-medium-play-v2    | diagnostics | path_start               |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | build_boundary_end       |     3.24808    |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | build_boundary_start     |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | build_edges_end          |     0.78848    |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | build_edges_start        |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | save_boundary_end        |     0.016247   |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | save_boundary_start      |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | save_graph_end           |     0.00486374 |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | save_graph_start         |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | select_nodes_end         |     1.52845    |
| loaded_baseline | antmaze-medium-play-v2    | graph_build | select_nodes_start       |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | embed_dataset_end        |     0.0186463  |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | embed_dataset_start      |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | load_dataset_end         |     3.04564    |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | load_dataset_start       |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | train_policy_end         |     0.00415309 |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | train_policy_start       |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | train_reachability_end   |     0.00333285 |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | train_reachability_start |   nan          |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | train_tdr_end            |     1.36717    |
| loaded_baseline | antmaze-medium-play-v2    | pipeline    | train_tdr_start          |   nan          |
| pu_retrain      | antmaze-large-diverse-v2  | diagnostics | boundary_end             |     1.27087    |
| pu_retrain      | antmaze-large-diverse-v2  | diagnostics | boundary_start           |   nan          |
| pu_retrain      | antmaze-large-diverse-v2  | diagnostics | edge_end                 |     1.02442    |
| pu_retrain      | antmaze-large-diverse-v2  | diagnostics | edge_start               |   nan          |
| pu_retrain      | antmaze-large-diverse-v2  | diagnostics | path_end                 |    59.0296     |
| pu_retrain      | antmaze-large-diverse-v2  | diagnostics | path_start               |   nan          |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | build_boundary_end       |     3.16374    |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | build_boundary_start     |   nan          |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | build_edges_end          |     0.0224986  |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | build_edges_start        |   nan          |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | save_boundary_end        |     0.0162535  |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | save_boundary_start      |   nan          |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | save_graph_end           |     0.00495958 |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | save_graph_start         |   nan          |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | select_nodes_end         |     1.58314    |
| pu_retrain      | antmaze-large-diverse-v2  | graph_build | select_nodes_start       |   nan          |


## Training Evidence

| condition       | env                       |   seed | event       | phase        |   step |       loss |
|:----------------|:--------------------------|-------:|:------------|:-------------|-------:|-----------:|
| loaded_baseline | antmaze-medium-diverse-v2 |      0 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-medium-diverse-v2 |      0 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-medium-diverse-v2 |      0 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-medium-diverse-v2 |      1 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-medium-diverse-v2 |      1 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-medium-diverse-v2 |      1 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-medium-diverse-v2 |      2 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-medium-diverse-v2 |      2 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-medium-diverse-v2 |      2 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-medium-play-v2    |      1 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-medium-play-v2    |      1 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-medium-play-v2    |      1 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-medium-play-v2    |      2 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-medium-play-v2    |      2 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-medium-play-v2    |      2 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-medium-play-v2    |      0 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-medium-play-v2    |      0 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-medium-play-v2    |      0 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-large-play-v2     |      0 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-large-play-v2     |      0 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-large-play-v2     |      0 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-large-play-v2     |      2 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-large-play-v2     |      2 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-large-play-v2     |      2 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-large-play-v2     |      1 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-large-play-v2     |      1 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-large-play-v2     |      1 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-large-diverse-v2  |      2 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-large-diverse-v2  |      2 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-large-diverse-v2  |      2 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-large-diverse-v2  |      1 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-large-diverse-v2  |      1 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-large-diverse-v2  |      1 | loaded      | reachability |    nan | nan        |
| loaded_baseline | antmaze-large-diverse-v2  |      0 | loaded      | tdr          |    nan | nan        |
| loaded_baseline | antmaze-large-diverse-v2  |      0 | loaded      | policy       |    nan | nan        |
| loaded_baseline | antmaze-large-diverse-v2  |      0 | loaded      | reachability |    nan | nan        |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | loaded      | tdr          |    nan | nan        |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | loaded      | policy       |    nan | nan        |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | train_start | reachability |    nan | nan        |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |      0 |   1.86074  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |    200 |   1.16845  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |    400 |   1.13472  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |    600 |   1.12848  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |    800 |   1.09647  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   1000 |   1.09607  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   1200 |   1.04343  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   1400 |   1.01586  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   1600 |   1.01177  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   1800 |   1.02697  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   2000 |   1.01207  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   2200 |   1.00942  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   2400 |   1.01874  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   2600 |   0.997573 |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   2800 |   1.0313   |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | nan         | reachability |   2999 |   1.01244  |
| pu_retrain      | antmaze-medium-diverse-v2 |      0 | saved       | reachability |    nan | nan        |
| pu_retrain      | antmaze-medium-play-v2    |      0 | loaded      | tdr          |    nan | nan        |
| pu_retrain      | antmaze-medium-play-v2    |      0 | loaded      | policy       |    nan | nan        |
| pu_retrain      | antmaze-medium-play-v2    |      0 | train_start | reachability |    nan | nan        |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |      0 |   1.86417  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |    200 |   1.16586  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |    400 |   1.1408   |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |    600 |   1.1208   |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |    800 |   1.09998  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   1000 |   1.10193  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   1200 |   1.04688  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   1400 |   1.01573  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   1600 |   1.00971  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   1800 |   1.01921  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   2000 |   1.01185  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   2200 |   0.998352 |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   2400 |   0.998109 |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   2600 |   1.00198  |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   2800 |   0.988676 |
| pu_retrain      | antmaze-medium-play-v2    |      0 | nan         | reachability |   2999 |   0.980455 |
| pu_retrain      | antmaze-medium-play-v2    |      0 | saved       | reachability |    nan | nan        |
| pu_retrain      | antmaze-large-play-v2     |      0 | loaded      | tdr          |    nan | nan        |
| pu_retrain      | antmaze-large-play-v2     |      0 | loaded      | policy       |    nan | nan        |
| pu_retrain      | antmaze-large-play-v2     |      0 | train_start | reachability |    nan | nan        |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |      0 |   1.89028  |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |    200 |   1.04926  |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |    400 |   1.02843  |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |    600 |   1.02558  |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |    800 |   1.03368  |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   1000 |   0.972448 |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   1200 |   0.95186  |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   1400 |   0.995769 |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   1600 |   0.945908 |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   1800 |   0.962826 |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   2000 |   0.889644 |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   2200 |   0.910072 |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   2400 |   0.952082 |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   2600 |   0.940572 |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   2800 |   0.908174 |
| pu_retrain      | antmaze-large-play-v2     |      0 | nan         | reachability |   2999 |   0.898386 |
| pu_retrain      | antmaze-large-play-v2     |      0 | saved       | reachability |    nan | nan        |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | loaded      | tdr          |    nan | nan        |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | loaded      | policy       |    nan | nan        |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | train_start | reachability |    nan | nan        |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |      0 |   1.895    |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |    200 |   1.07945  |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |    400 |   1.07089  |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |    600 |   1.04235  |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |    800 |   1.01394  |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   1000 |   0.991581 |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   1200 |   0.988863 |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   1400 |   0.977592 |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   1600 |   0.946246 |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   1800 |   0.961334 |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   2000 |   0.93459  |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   2200 |   0.919949 |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   2400 |   0.933412 |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   2600 |   0.900598 |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   2800 |   0.93979  |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | nan         | reachability |   2999 |   0.940062 |
| pu_retrain      | antmaze-large-diverse-v2  |      0 | saved       | reachability |    nan | nan        |

