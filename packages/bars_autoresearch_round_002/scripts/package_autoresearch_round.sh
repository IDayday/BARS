#!/usr/bin/env bash
set -euo pipefail

ROUND="${ROUND:-}"
OUT="${OUT:-}"
for arg in "$@"; do
  case "$arg" in
    ROUND=*) ROUND="${arg#ROUND=}" ;;
    OUT=*) OUT="${arg#OUT=}" ;;
    --round=*) ROUND="${arg#--round=}" ;;
    --out=*) OUT="${arg#--out=}" ;;
  esac
done

if [[ -z "$ROUND" ]]; then
  ROUND="002"
fi
ROUND_NUM="$(printf '%03d' "${ROUND#0}")"
RN="round_${ROUND_NUM}"
OUT="${OUT:-packages/bars_autoresearch_${RN}.tar.gz}"
PKG_ROOT="packages/bars_autoresearch_${RN}"

rm -rf "$PKG_ROOT"
mkdir -p "$PKG_ROOT"/{configs,scripts,reports,rounds,research_state,commands,logs,code}

cp -a configs/. "$PKG_ROOT/configs/" 2>/dev/null || true
cp scripts/autoresearch_*.py "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/autoresearch_*.sh "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/build_baseline_registry.py "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/fetch_public_baseline_targets.py "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/verify_baseline_card.py "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/certify_gas_baseline.sh "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/run_official_gas_eval.py "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/compare_official_vs_bars_adapter.py "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/reclassify_prior_evidence.py "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/analyze_baseline_certification.py "$PKG_ROOT/scripts/" 2>/dev/null || true
cp scripts/package_autoresearch_round.sh "$PKG_ROOT/scripts/" 2>/dev/null || true
find reports -maxdepth 1 -type f -name "${RN}_*" -exec cp {} "$PKG_ROOT/reports/" \; 2>/dev/null || true
cp -a "rounds/${RN}" "$PKG_ROOT/rounds/" 2>/dev/null || true
cp research_state/*.json "$PKG_ROOT/research_state/" 2>/dev/null || true
cp research_state/*.jsonl "$PKG_ROOT/research_state/" 2>/dev/null || true
cp -a research_state/baseline_cards "$PKG_ROOT/research_state/" 2>/dev/null || true
cp "rounds/${RN}/commands.sh" "$PKG_ROOT/commands/${RN}_commands.sh" 2>/dev/null || true
cp "rounds/${RN}/failed_jobs.csv" "$PKG_ROOT/logs/${RN}_failed_jobs.csv" 2>/dev/null || true
cp "rounds/${RN}/jobs.tsv" "$PKG_ROOT/logs/${RN}_jobs.tsv" 2>/dev/null || true

git status --short > "$PKG_ROOT/code/${RN}_git_status.txt" || true
git rev-parse HEAD > "$PKG_ROOT/code/${RN}_git_head.txt" || true
git diff -- . \
  ':(exclude)artifacts/**' \
  ':(exclude)runs_stage*/**' \
  ':(exclude)packages/**' \
  ':(exclude)__pycache__/**' \
  ':(exclude)*.npz' \
  ':(exclude)*.pt' \
  ':(exclude)*.pth' \
  ':(exclude)*.ckpt' \
  ':(exclude)*.pkl' \
  > "$PKG_ROOT/code/${RN}_code_diff.patch" || true

(
  cd "$PKG_ROOT"
  find . -type f | sort > MANIFEST.txt
  sha256sum $(find . -type f ! -name CHECKSUMS.sha256 | sort) > CHECKSUMS.sha256
)

mkdir -p "$(dirname "$OUT")"
tar --exclude='*.npz' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.pkl' --exclude='__pycache__' -czf "$OUT" -C packages "bars_autoresearch_${RN}"
cp "$PKG_ROOT/MANIFEST.txt" "rounds/${RN}/package_manifest.txt" 2>/dev/null || true
echo "$OUT"
