#!/usr/bin/env bash
set -euo pipefail
bash scripts/round005_train_gas_from_scratch_3seed_queue.sh \
  ROUND=005 \
  ENVS=antmaze-giant-stitch-v0,antmaze-large-explore-v0,scene-play-v0 \
  SEEDS=0,1,2 \
  GPUS=0,1,2,3,4,5 \
  TRAIN_STEPS=1000000 \
  LOG_INTERVAL=5000 \
  SAVE_INTERVAL=100000 \
  TE_THRESHOLD=0.99 \
  OUT_ROOT=artifacts/gas_selftrain_round005 \
  RUN_ROOT=runs_round005_gas_selftrain \
  GAS_REPO_PATH=external_src/GAS \
  WAIT_FOR_GPU_FREE=0 \
  FORCE=1
