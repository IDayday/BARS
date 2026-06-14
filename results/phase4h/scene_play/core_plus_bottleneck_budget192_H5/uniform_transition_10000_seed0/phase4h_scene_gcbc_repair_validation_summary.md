# Phase 4H Stronger Scene GCBC Direct Repair Validation

Phase 4H checks whether the Scene Phase 4G direct repair-edge policy
evidence survives a GCBC model trained longer than the prior 200-step
smoke run. The metric remains offline supervised action fitting and is
not rollout success or closed-loop option execution.

## Training

- `run_dir`: `results/phase4h/scene_play/core_plus_bottleneck_budget192_H5/uniform_transition_10000_seed0/gcbc`
- `last_step`: `10000`
- `final_val_action_mse`: `0.0051270271651446`
- `best_val_action_mse`: `0.0051270271651446`
- `num_train_segments`: `132995`

## Direct Repair Diagnostics

- `mean_direct_edge_action_mse`: baseline `0.038237718896338785`, candidate `0.011463576496073543`
- `median_direct_edge_action_mse`: baseline `0.023924724622206253`, candidate `0.005752590353949927`
- `mean_direct_policy_support_score`: baseline `0.5613987912850971`, candidate `0.8332988952585879`
- `median_direct_policy_support_score`: baseline `0.6190438053660963`, candidate `0.8911249055165809`
- `direct_certified_rate`: baseline `0.87`, candidate `0.894`
- `transfer_certified_rate`: baseline `0.794`, candidate `0.794`
- `mean_policy_score_delta_direct_minus_transfer`: baseline `0.21998491071931187`, candidate `0.4918850146928027`
- `spearman_transfer_vs_direct_policy_score`: baseline `0.5467250685239224`, candidate `0.49820359303897216`
- `mean_direct_reliability`: baseline `0.3665760088594466`, candidate `0.40188976399671994`
- `mean_transfer_reliability`: baseline `0.3200390060463293`, candidate `0.3200390060463293`

## Planner Deltas

- `calibrated_compat_penalized` `path_coverage`: baseline `0.51`, candidate `0.51`, delta `0.0`
- `calibrated_compat_penalized` `mean_min_edge_proxy_score`: baseline `0.17300133454025365`, candidate `0.18206866864991042`, delta `0.009067334109656766`
- `calibrated_compat_penalized` `mean_uncertified_edge_fraction`: baseline `0.1964052287581699`, candidate `0.1849673202614379`, delta `-0.011437908496732013`
- `calibrated_compat_penalized` `mean_pair_incompatible_fraction`: baseline `0.707516339869281`, candidate `0.7009803921568627`, delta `-0.006535947712418277`
- `calibrated_compat_threshold` `path_coverage`: baseline `0.48`, candidate `0.48`, delta `0.0`
- `calibrated_compat_threshold` `mean_min_edge_proxy_score`: baseline `0.24361123623951766`, candidate `0.24757374342760316`, delta `0.003962507188085496`
- `calibrated_compat_threshold` `mean_uncertified_edge_fraction`: baseline `0.03733540764790765`, candidate `0.033168740981240986`, delta `-0.004166666666666666`
- `calibrated_compat_threshold` `mean_pair_incompatible_fraction`: baseline `0.0`, candidate `0.0`, delta `0.0`
- `compat_threshold` `path_coverage`: baseline `0.48`, candidate `0.48`, delta `0.0`
- `compat_threshold` `mean_min_edge_proxy_score`: baseline `0.14841365797825998`, candidate `0.14935070662291722`, delta `0.0009370486446572401`
- `compat_threshold` `mean_uncertified_edge_fraction`: baseline `0.16448412698412698`, candidate `0.159796626984127`, delta `-0.004687499999999983`
- `compat_threshold` `mean_pair_incompatible_fraction`: baseline `0.0`, candidate `0.0`, delta `0.0`
- `support_shortest_path` `path_coverage`: baseline `0.51`, candidate `0.51`, delta `0.0`
- `support_shortest_path` `mean_min_edge_proxy_score`: baseline `0.11847457942373496`, candidate `0.1269506396619673`, delta `0.00847606023823233`
- `support_shortest_path` `mean_uncertified_edge_fraction`: baseline `0.3568627450980392`, candidate `0.34542483660130713`, delta `-0.011437908496732097`
- `support_shortest_path` `mean_pair_incompatible_fraction`: baseline `0.9084967320261437`, candidate `0.9084967320261437`, delta `0.0`

## Interpretation

The comparison isolates model-training strength while reusing the same
support-bank repair edges, Phase 4F calibration, and Phase 4G planning
queries. Improvements in direct repair-edge MSE or certification rate
make the offline proxy more credible, but still do not establish rollout
success or arbitrary-reset executability.

Related work reviewed: GCSL, RvS, and the GCSL reference implementation.
