# Round 005 GAS 3-Seed Self-Training Launch

- Evidence class while running: E4_FULL_BUDGET_TRAINED_METHOD pending completion.
- Official weights used: false.
- From scratch: FORCE=1.
- GPU wait: WAIT_FOR_GPU_FREE=0.
- Environments: antmaze-giant-stitch-v0,antmaze-large-explore-v0,scene-play-v0.
- Seeds: 0,1,2.
- GPUs: 0,1,2,3,4,5.
- Train steps: 1000000 for TDR and policy.
- Dataset root: /root/remote/datasets/ogbench.
- Artifact root: artifacts/gas_selftrain_round005.
- Run root: runs_round005_gas_selftrain.
- Initial job assignment: reports/round_005_gas_selftrain_3seed_jobs.tsv.
- Initial worker table: reports/round_005_gas_selftrain_3seed_workers.tsv.
- Direct full-open scene launch table: reports/round_005_gas_selftrain_direct_scene_jobs.tsv.
- Launch command: commands/round_005_gas_selftrain_3seed_launch.sh.

Scheduling update: the initial launcher queued `scene-play-v0` behind the GPU0-2 giant jobs. At user request, the GPU0-2 queue parent processes were stopped while their current giant child jobs continued, and `scene-play-v0` seeds 0/1/2 were launched immediately on GPU0/1/2. Round 005 therefore now runs all nine TDR jobs concurrently where GPU0/1/2 each host both one giant and one scene job; GPU3/4/5 each host one large job. External GPU occupancy is not waited on.

No p_bridge, integrated BARS-v3, oracle-headroom, or failure taxonomy interpretation is run by this launcher.
