#!/usr/bin/env bash
set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PKG="${PKG:-stage25_results_config_package_${STAMP}}"
OUT_DIR="${OUT_DIR:-packages}"
PKG_DIR="${PKG_DIR:-${OUT_DIR}/${PKG}}"
TAR="${TAR:-${OUT_DIR}/${PKG}.tar.gz}"

mkdir -p "$OUT_DIR"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"/{configs,scripts,reports,logs,code,commands}

find configs -maxdepth 1 -type f -name 'stage25_*.json' -exec cp {} "$PKG_DIR/configs/" \;
find scripts -maxdepth 1 -type f \( -name 'stage25_*.py' -o -name 'stage25_*.sh' \) -exec cp {} "$PKG_DIR/scripts/" \;
find reports -maxdepth 1 -type f \( -name 'stage25_*.csv' -o -name 'stage25_*.json' -o -name 'stage25_*.md' \) -exec cp {} "$PKG_DIR/reports/" \; 2>/dev/null || true

for d in runs_stage25_*_logs; do
  [[ -d "$d" ]] || continue
  mkdir -p "$PKG_DIR/logs/$d"
  [[ -f "$d/jobs.tsv" ]] && cp "$d/jobs.tsv" "$PKG_DIR/logs/$d/"
  [[ -f "$d/failed_jobs.csv" ]] && cp "$d/failed_jobs.csv" "$PKG_DIR/logs/$d/"
done
for d in reports/stage25_oracle_scan_tmp*; do
  [[ -d "$d" ]] || continue
  out="$PKG_DIR/reports/${d#reports/}"
  mkdir -p "$out"
  find "$d" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.md' -o -name 'failed_jobs.csv' \) -exec cp {} "$out/" \;
done

git status --short > "$PKG_DIR/code/git_status.txt" || true
git rev-parse --short HEAD > "$PKG_DIR/code/git_head.txt" || true
git diff -- . \
  ':(exclude)*.npz' \
  ':(exclude)*.pt' \
  ':(exclude)*.pth' \
  ':(exclude)*.ckpt' \
  ':(exclude)*.pkl' \
  ':(exclude)__pycache__/**' \
  ':(exclude)artifacts/**' \
  ':(exclude)_data/**' \
  ':(exclude)packages/**' \
  ':(exclude)stage24_results_config_package_*/**' \
  ':(exclude)stage25_results_config_package_*/**' \
  > "$PKG_DIR/code/stage25_code_diff.patch" || true

cat > "$PKG_DIR/commands/stage25_commands.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source scripts/stage24_env_mirrors.sh 2>/dev/null || true
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONPATH=$PWD

python scripts/stage25_enrich_failure_atlas_all_variants.py \
  --eval-roots runs_stage24_reachability_confirm,runs_stage24_local_drift \
  --out reports/stage25_failure_atlas_all_variants.csv \
  --summary-out reports/stage25_failure_atlas_summary.csv \
  --integrity-out reports/stage25_label_integrity.json \
  --min-episodes 100

bash scripts/stage25_reachability_closing.sh \
  ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 \
  SEEDS=0,1,2 EPISODES=300 GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1

bash scripts/stage25_run_local_drift_v2.sh \
  ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 \
  SEEDS=0,1,2 EPISODES=100 GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1

bash scripts/stage25_oracle_headroom_scan_v2.sh \
  CONFIG=configs/stage25_oracle_scan_matrix.json \
  ENVS=scene-play-v0 SEEDS=0 GRAPH_IDS=G3 GPUS=${GPUS:-0} WAIT=1

python scripts/stage25_boundary_coverage_repair.py \
  --oracle-artifact-root artifacts/stage25 \
  --oracle-reports-root reports/stage25_oracle_scan_tmp \
  --out reports/stage25_boundary_coverage.csv \
  --summary-out reports/stage25_boundary_coverage.md

python scripts/stage25_d4rl_protocol_audit.py \
  --envs antmaze-medium-play-v2,antmaze-medium-diverse-v2,antmaze-large-play-v2,antmaze-large-diverse-v2 \
  --gas-artifact-root artifacts/gas \
  --out-md reports/stage25_d4rl_protocol_audit.md \
  --out-json reports/stage25_d4rl_protocol_audit.json

python scripts/stage25_analyze.py \
  --reports-root reports \
  --reachability-roots runs_stage25_reachability_closing \
  --local-drift-roots runs_stage25_local_drift_v2 \
  --oracle-reports-root reports/stage25_oracle_scan_tmp \
  --oracle-artifact-root artifacts/stage25 \
  --failure-atlas reports/stage25_failure_atlas_all_variants.csv \
  --min-episodes 100
EOF
chmod +x "$PKG_DIR/commands/stage25_commands.sh"

python - "$PKG_DIR" <<'PY'
import json, sys
from pathlib import Path
pkg=Path(sys.argv[1])
status_path=pkg/"reports/stage25_gate_status.json"
status=json.loads(status_path.read_text()) if status_path.exists() else {}
keys=[
  "reachability_closing","label_integrity","local_drift_v2","oracle_headroom",
  "p_bridge","boundary","d4rl_protocol","integrated"
]
lines=["# Stage25 Results Package","", "## Gates", ""]
for k in keys:
    lines.append(f"- {k}: {status.get(k, 'unknown')}")
lines.extend([
    "",
    "## Contents",
    "",
    "- `reports/`: Stage25 CSV/JSON/Markdown outputs.",
    "- `configs/`: Stage25 experiment configs.",
    "- `scripts/`: Stage25 entrypoints and analyzers.",
    "- `logs/`: lightweight jobs and failed job files.",
    "- `code/`: git status, head, and code diff.",
    "- `commands/stage25_commands.sh`: replay command sequence.",
    "",
    "Datasets, checkpoints, cache files, large pickle files, and full debug JSONL traces are excluded.",
])
(pkg/"README.md").write_text("\\n".join(lines)+"\\n")
PY

(
  cd "$PKG_DIR"
  find . -type f | sort > MANIFEST.txt
  sha256sum $(find . -type f ! -name CHECKSUMS.sha256 | sort) > CHECKSUMS.sha256
)

mkdir -p "$(dirname "$TAR")"
tar --exclude='*.npz' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='__pycache__' --exclude='*.jsonl' -czf "$TAR" -C "$(dirname "$PKG_DIR")" "$(basename "$PKG_DIR")"
echo "[package_stage25] wrote $TAR"
sha256sum "$TAR"
