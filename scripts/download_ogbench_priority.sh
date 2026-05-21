#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/project/offlinerl_datasets}"
OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$DATA_ROOT/ogbench}"
OGBENCH_BASE_URL="${OGBENCH_BASE_URL:-https://rail.eecs.berkeley.edu/datasets/ogbench}"
LOG_DIR="${LOG_DIR:-$DATA_ROOT/logs}"
QUEUE_DIR="${QUEUE_DIR:-$DATA_ROOT/queues}"
ARIA2_CONCURRENT="${ARIA2_CONCURRENT:-6}"
ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-16}"
ARIA2_SPLITS="${ARIA2_SPLITS:-16}"
ARIA2_SUMMARY_INTERVAL="${ARIA2_SUMMARY_INTERVAL:-60}"

NO_PROXY_ENV=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy)

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/download_ogbench_priority.sh [gas|state|locomaze|manip|visual|all] [--background]

Presets:
  gas       Datasets shown in the GAS tables: state AntMaze/Scene and visual AntMaze/Scene.
  state     Priority non-visual OGBench datasets not already in the core AntMaze set.
  locomaze  PointMaze, extra AntMaze, AntSoccer, HumanoidMaze.
  manip     Cube, Scene, Puzzle, PowderWorld.
  visual    Visual OGBench datasets. Large and lower priority for state-based CRL.
  all       state + visual.
USAGE
}

is_domestic_url() {
  case "$1" in
    *hf-mirror.com*|*modelscope.cn*|*opendatalab.com*|*aliyun*|*tuna.tsinghua.edu.cn*|*ustc.edu.cn*|*bfsu.edu.cn*|*ctyun.cn*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_for_url() {
  local url="$1"
  shift
  if is_domestic_url "$url"; then
    "${NO_PROXY_ENV[@]}" "$@"
  else
    "$@"
  fi
}

valid_npz() {
  python - "$1" <<'PY'
import sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as zf:
        ok = zf.testzip() is None
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

append_dataset() {
  local dataset="$1"
  local queue="$2"
  local suffix file path
  for suffix in "" "-val"; do
    file="${dataset}${suffix}.npz"
    path="$OGBENCH_DATASET_DIR/$file"
    if [[ -s "$path" ]] && valid_npz "$path"; then
      echo "ready $file"
      continue
    fi
    rm -f "$path" "$path.aria2"
    printf '%s/%s\n  out=%s\n' "${OGBENCH_BASE_URL%/}" "$file" "$file" >> "$queue"
    echo "queued $file"
  done
}

download_queue() {
  local queue="$1"
  if [[ ! -s "$queue" ]]; then
    echo "No missing files for this preset."
    return 0
  fi

  mkdir -p "$OGBENCH_DATASET_DIR"
  run_for_url "$OGBENCH_BASE_URL" aria2c \
    --console-log-level=notice \
    --summary-interval="$ARIA2_SUMMARY_INTERVAL" \
    --continue=true \
    --allow-overwrite=false \
    --auto-file-renaming=false \
    --file-allocation=none \
    --max-tries=10 \
    --retry-wait=10 \
    --connect-timeout=20 \
    --timeout=90 \
    -x "$ARIA2_CONNECTIONS" \
    -s "$ARIA2_SPLITS" \
    -k 1M \
    --max-concurrent-downloads="$ARIA2_CONCURRENT" \
    -d "$OGBENCH_DATASET_DIR" \
    -i "$queue"
}

verify_selected() {
  local dataset
  for dataset in "$@"; do
    valid_npz "$OGBENCH_DATASET_DIR/${dataset}.npz"
    valid_npz "$OGBENCH_DATASET_DIR/${dataset}-val.npz"
    echo "verified $dataset"
  done
}

start_background() {
  local preset="$1"
  mkdir -p "$LOG_DIR"
  local stamp log
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/ogbench_${preset}_${stamp}.log"
  setsid nohup "$0" "$preset" >"$log" 2>&1 < /dev/null &
  echo "Started OGBench $preset download in background."
  echo "PID: $!"
  echo "Log: $log"
}

tier_gas_state=(
  antmaze-medium-navigate-v0
  antmaze-large-navigate-v0
  antmaze-giant-navigate-v0
  antmaze-medium-stitch-v0
  antmaze-large-stitch-v0
  antmaze-giant-stitch-v0
  antmaze-medium-explore-v0
  antmaze-large-explore-v0
  scene-play-v0
)

tier_gas_pixel=(
  visual-antmaze-medium-navigate-v0
  visual-antmaze-large-navigate-v0
  visual-antmaze-giant-navigate-v0
  visual-antmaze-medium-stitch-v0
  visual-antmaze-large-stitch-v0
  visual-antmaze-giant-stitch-v0
  visual-antmaze-medium-explore-v0
  visual-antmaze-large-explore-v0
  visual-scene-play-v0
)

tier_pointmaze=(
  pointmaze-medium-stitch-v0
  pointmaze-medium-navigate-v0
  pointmaze-large-stitch-v0
  pointmaze-large-navigate-v0
  pointmaze-giant-stitch-v0
  pointmaze-giant-navigate-v0
  pointmaze-teleport-stitch-v0
  pointmaze-teleport-navigate-v0
)

tier_ant_extra=(
  antmaze-medium-explore-v0
  antmaze-large-explore-v0
  antmaze-teleport-stitch-v0
  antmaze-teleport-navigate-v0
  antmaze-teleport-explore-v0
)

tier_humanoid=(
  humanoidmaze-medium-stitch-v0
  humanoidmaze-medium-navigate-v0
  humanoidmaze-large-stitch-v0
  humanoidmaze-large-navigate-v0
  humanoidmaze-giant-stitch-v0
  humanoidmaze-giant-navigate-v0
)

tier_antsoccer=(
  antsoccer-arena-stitch-v0
  antsoccer-arena-navigate-v0
  antsoccer-medium-stitch-v0
  antsoccer-medium-navigate-v0
)

tier_cube=(
  cube-single-play-v0
  cube-single-noisy-v0
  cube-double-play-v0
  cube-double-noisy-v0
  cube-triple-play-v0
  cube-triple-noisy-v0
  cube-quadruple-play-v0
  cube-quadruple-noisy-v0
)

tier_scene_puzzle=(
  scene-play-v0
  scene-noisy-v0
  puzzle-3x3-play-v0
  puzzle-3x3-noisy-v0
  puzzle-4x4-play-v0
  puzzle-4x4-noisy-v0
  puzzle-4x5-play-v0
  puzzle-4x5-noisy-v0
  puzzle-4x6-play-v0
  puzzle-4x6-noisy-v0
  powderworld-easy-play-v0
  powderworld-medium-play-v0
  powderworld-hard-play-v0
)

tier_visual=(
  visual-antmaze-medium-stitch-v0
  visual-antmaze-medium-navigate-v0
  visual-antmaze-large-stitch-v0
  visual-antmaze-large-navigate-v0
  visual-antmaze-giant-stitch-v0
  visual-antmaze-giant-navigate-v0
  visual-antmaze-medium-explore-v0
  visual-antmaze-large-explore-v0
  visual-antmaze-teleport-stitch-v0
  visual-antmaze-teleport-navigate-v0
  visual-antmaze-teleport-explore-v0
  visual-humanoidmaze-medium-stitch-v0
  visual-humanoidmaze-medium-navigate-v0
  visual-humanoidmaze-large-stitch-v0
  visual-humanoidmaze-large-navigate-v0
  visual-humanoidmaze-giant-stitch-v0
  visual-humanoidmaze-giant-navigate-v0
  visual-cube-single-play-v0
  visual-cube-single-noisy-v0
  visual-cube-double-play-v0
  visual-cube-double-noisy-v0
  visual-cube-triple-play-v0
  visual-cube-triple-noisy-v0
  visual-cube-quadruple-play-v0
  visual-cube-quadruple-noisy-v0
  visual-scene-play-v0
  visual-scene-noisy-v0
  visual-puzzle-3x3-play-v0
  visual-puzzle-3x3-noisy-v0
  visual-puzzle-4x4-play-v0
  visual-puzzle-4x4-noisy-v0
  visual-puzzle-4x5-play-v0
  visual-puzzle-4x5-noisy-v0
  visual-puzzle-4x6-play-v0
  visual-puzzle-4x6-noisy-v0
)

preset="${1:-state}"
background="false"
if [[ "${2:-}" == "--background" ]]; then
  background="true"
elif [[ "${1:-}" == "--background" ]]; then
  preset="state"
  background="true"
fi

case "$preset" in
  gas)
    selected=("${tier_gas_state[@]}" "${tier_gas_pixel[@]}")
    ;;
  state)
    selected=("${tier_pointmaze[@]}" "${tier_ant_extra[@]}" "${tier_humanoid[@]}" "${tier_antsoccer[@]}" "${tier_cube[@]}" "${tier_scene_puzzle[@]}")
    ;;
  locomaze)
    selected=("${tier_pointmaze[@]}" "${tier_ant_extra[@]}" "${tier_humanoid[@]}" "${tier_antsoccer[@]}")
    ;;
  manip)
    selected=("${tier_cube[@]}" "${tier_scene_puzzle[@]}")
    ;;
  visual)
    selected=("${tier_visual[@]}")
    ;;
  all)
    selected=("${tier_pointmaze[@]}" "${tier_ant_extra[@]}" "${tier_humanoid[@]}" "${tier_antsoccer[@]}" "${tier_cube[@]}" "${tier_scene_puzzle[@]}" "${tier_visual[@]}")
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ "$background" == "true" ]]; then
  start_background "$preset"
  exit 0
fi

mkdir -p "$QUEUE_DIR" "$OGBENCH_DATASET_DIR"
stamp="$(date +%Y%m%d_%H%M%S)"
queue="$QUEUE_DIR/ogbench_${preset}_${stamp}.txt"
: > "$queue"

echo "Preset: $preset"
echo "Dataset dir: $OGBENCH_DATASET_DIR"
echo "Source: ${OGBENCH_BASE_URL%/}"
echo "Queue: $queue"
for dataset in "${selected[@]}"; do
  append_dataset "$dataset" "$queue"
done

download_queue "$queue"
verify_selected "${selected[@]}"
