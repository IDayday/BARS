#!/usr/bin/env bash
set -euo pipefail
source scripts/stage24_env_mirrors.sh
export PATH=/root/anaconda3/envs/gcrlo/bin:$PATH
export PYTHONPATH=$PWD
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export D4RL_SUPPRESS_IMPORT_ERROR=1

bash scripts/stage24_run_reachability_confirm.sh \
  CONFIG=configs/stage24_reachability_confirm.json \
  ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 \
  SEEDS=0,1,2 EPISODES=100 \
  GPUS=0,1,2,3,4,5 MAX_PARALLEL_EVAL=18 WAIT=1 \
  PREFER_PRETRAINED=1 TRAIN_IF_MISSING=0 QUICK=0 EVAL_ON_CPU=0 \
  SKIP_PREPARE=1 SKIP_REACHABILITY_TRAIN=1 ANALYZE_MIN_EPISODES=100

bash scripts/stage24_run_reachability_confirm.sh \
  CONFIG=configs/stage24_reachability_confirm.json \
  ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 \
  SEEDS=0,1,2 EPISODES=100 \
  VARIANTS=gas_shortest_replan_on_local_drift,gas_shortest_adaptive_subgoal_horizon,gas_reachability_budget_replan_on_local_drift \
  STAGE24_ROOT=runs_stage24_local_drift LOG_ROOT=runs_stage24_local_drift_logs \
  GPUS=0,1,2,3,4,5 MAX_PARALLEL_EVAL=18 WAIT=1 \
  PREFER_PRETRAINED=1 TRAIN_IF_MISSING=0 QUICK=0 EVAL_ON_CPU=0 \
  SKIP_PREPARE=1 SKIP_REACHABILITY_TRAIN=1 ANALYZE_MIN_EPISODES=100 MAX_STEPS=400

bash scripts/stage24_oracle_headroom_scan.sh \
  ENVS=scene-play-v0 SEEDS=0 GPUS=5 EDGE_EXEC_PILOT=1 TOP_K_BRIDGE=4 MAX_SOURCES=200 WAIT=1 \
  PREFER_PRETRAINED=1 TRAIN_IF_MISSING=0 QUICK=0 \
  PREPARE_LOG_ROOT=runs_stage24_prepare_oracle_scene_retry \
  ORACLE_REPORTS_ROOT=reports/stage24_oracle_scan_tmp_scene_retry

python scripts/stage24_local_drift_diagnostic.py \
  --eval-roots runs_stage24_reachability_confirm,runs_stage24_local_drift \
  --out reports/stage24_local_drift.csv

python scripts/stage24_analyze.py \
  --reachability-roots runs_stage24_reachability_confirm \
  --local-drift-roots runs_stage24_reachability_confirm,runs_stage24_local_drift \
  --oracle-reports-root reports/stage24_oracle_scan_tmp_scene_retry \
  --oracle-artifact-root artifacts/stage24 \
  --reports-root reports \
  --min-episodes 100
