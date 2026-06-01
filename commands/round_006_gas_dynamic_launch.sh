#!/usr/bin/env bash
set -euo pipefail
python scripts/round006_gas_dynamic_orchestrator.py --round 006 --dataset-dir /mnt/project/offlinerl_datasets/ogbench --run-root runs_round006_gas_dynamic --out-root artifacts/gas_selftrain_round006 --seeds 42,43,44,45,46 --gpus 0,1 --gpu-slots-per-gpu 8 --poll-seconds 60 --download-poll-seconds 30 --jobs-tsv rounds/round_006/gas_dynamic_remaining_jobs.tsv
