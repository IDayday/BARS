#!/usr/bin/env bash
set -euo pipefail

ROUND=""
for arg in "$@"; do
  case "$arg" in
    --round) shift; ROUND="${1:-}" ;;
    --round=*) ROUND="${arg#--round=}" ;;
  esac
done

if [[ -z "$ROUND" ]]; then
  ROUND="$(python - <<'PY'
import json
from pathlib import Path
state=json.loads(Path('research_state/bars_research_state.json').read_text())
print(state.get('round', 1))
PY
)"
fi
ROUND_NUM="$(printf '%03d' "$ROUND")"
RN="round_${ROUND_NUM}"
OUT_DIR="packages"
PKG_DIR="${OUT_DIR}/bars_autoresearch_${RN}"
TAR="${OUT_DIR}/bars_autoresearch_${RN}.tar.gz"
mkdir -p "$OUT_DIR"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"/{round,reports,research_state,scripts,code}

cp -a "rounds/${RN}/." "$PKG_DIR/round/" 2>/dev/null || true
find reports -maxdepth 1 -type f -name "${RN}_*" -exec cp {} "$PKG_DIR/reports/" \; 2>/dev/null || true
cp research_state/*.json research_state/*.jsonl "$PKG_DIR/research_state/" 2>/dev/null || true
cp scripts/autoresearch_*.py scripts/autoresearch_*.sh "$PKG_DIR/scripts/" 2>/dev/null || true
cp scripts/stage25_enrich_failure_atlas_all_variants.py scripts/stage25_analyze.py "$PKG_DIR/scripts/" 2>/dev/null || true
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
  ':(exclude)rounds/**' \
  ':(exclude)packages/**' \
  ':(exclude)stage24_results_config_package_*/**' \
  ':(exclude)stage25_results_config_package_*/**' \
  > "$PKG_DIR/code/git_diff.patch" || true

(
  cd "$PKG_DIR"
  find . -type f | sort > MANIFEST.txt
  sha256sum $(find . -type f ! -name CHECKSUMS.sha256 | sort) > CHECKSUMS.sha256
)
cp "$PKG_DIR/MANIFEST.txt" "rounds/${RN}/package_manifest.txt" 2>/dev/null || true
tar --exclude='*.jsonl.gz' --exclude='*.npz' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='__pycache__' -czf "$TAR" -C "$OUT_DIR" "bars_autoresearch_${RN}"

python - "$ROUND" "$TAR" <<'PY'
import json, sys
from pathlib import Path
round_num=int(sys.argv[1])
rn=f"round_{round_num:03d}"
tar=sys.argv[2]
gate=json.loads((Path('rounds')/rn/'gate_status.json').read_text()) if (Path('rounds')/rn/'gate_status.json').exists() else {}
state=json.loads(Path('research_state/bars_research_state.json').read_text())
failed=Path('rounds')/rn/'failed_jobs.csv'
failed_jobs=max(0, len(failed.read_text().splitlines())-1) if failed.exists() else 0
summary={
  "ROUND": rn,
  "PRIMARY_QUESTION": gate.get("primary_question",""),
  "EXPERIMENTS_COMPLETED": "protocol_repair_all_variant_failure_atlas",
  "FAILED_JOBS": failed_jobs,
  "PRIMARY_RESULTS": gate.get("details",{}),
  "GATES": gate.get("gates",{}),
  "DECISION": state.get("global_decision"),
  "NEXT_ROUND": state.get("next_round_plan"),
  "PACKAGE": tar,
}
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
PY
