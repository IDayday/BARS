#!/usr/bin/env bash
set -Eeuo pipefail

ROOTS="${ROOTS:-runs_stage19_protocol_ablation_medium50 runs_stage19_protocol_ablation_medium100_core}"
CONTEXT_ROOTS="${CONTEXT_ROOTS:-runs_stage18_medium100_best_loaded runs_stage18_online_tuned_h50_thr1_lb01 runs_stage18_online_tuned_h50_thr1_lb03_fullbars runs_stage17_online_quick_loaded}"
OUT_ROOT="${OUT_ROOT:-analysis_packages}"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-5242880}"
TAIL_LINES="${TAIL_LINES:-2000}"
TS="$(date +%Y%m%d_%H%M%S)"
PKG_NAME="bars_stage19_protocol_results_${TS}"
TMP_ROOT="$(mktemp -d /tmp/bars_stage19_pack.XXXXXX)"
PKG_DIR="${TMP_ROOT}/${PKG_NAME}"
mkdir -p "$PKG_DIR" "$OUT_ROOT" reports

copy_file() { local f="$1"; local dest="${PKG_DIR}/${f}"; mkdir -p "$(dirname "$dest")"; cp -a "$f" "$dest"; }
copy_log_file() { local f="$1"; local dest="${PKG_DIR}/${f}"; mkdir -p "$(dirname "$dest")"; local size; size="$(stat -c%s "$f" 2>/dev/null || echo 0)"; if [ "$size" -le "$MAX_LOG_BYTES" ]; then cp -a "$f" "$dest"; else tail -n "$TAIL_LINES" "$f" > "${dest}.tail${TAIL_LINES}"; echo "TRUNCATED original_size_bytes=${size}" > "${dest}.TRUNCATED"; fi; }

META="${PKG_DIR}/metadata"; mkdir -p "$META"
date > "$META/packaged_at.txt"; pwd > "$META/project_path.txt"; hostname > "$META/hostname.txt" 2>&1 || true
python --version > "$META/python_version.txt" 2>&1 || true; which python > "$META/which_python.txt" 2>&1 || true
pip freeze > "$META/pip_freeze.txt" 2>&1 || true; nvidia-smi > "$META/nvidia_smi.txt" 2>&1 || true; df -h > "$META/disk_free.txt" 2>&1 || true
git status > "$META/git_status.txt" 2>&1 || true; git status --short > "$META/git_status_short.txt" 2>&1 || true; git branch --show-current > "$META/git_branch.txt" 2>&1 || true; git rev-parse HEAD > "$META/git_head.txt" 2>&1 || true; git log --oneline -40 > "$META/git_log_oneline_40.txt" 2>&1 || true; git diff > "$META/git_diff.patch" 2>&1 || true
pgrep -af "python -m bars.cli run" > "$META/current_bars_processes.txt" 2>&1 || true

for r in $ROOTS $CONTEXT_ROOTS; do
  [ -d "$r" ] || continue
  python scripts/collect_csv.py --log-root "$r" > "reports/package_collect_${r}_${TS}.out" 2> "reports/package_collect_${r}_${TS}.err" || true
  stage="stage3"
  python scripts/analyze_bars_results.py --log-root "$r" --stage "$stage" --out "reports/${r}_stage3_autopack_${TS}.md" --force-collect > "reports/package_analyze_${r}_${TS}.out" 2> "reports/package_analyze_${r}_${TS}.err" || true
done

COMPARE_ARGS=()
for r in $ROOTS $CONTEXT_ROOTS; do
  [ -d "$r" ] || continue
  label="$(basename "$r" | sed 's/^runs_//')"
  COMPARE_ARGS+=(--root "${label}=${r}")
done
if [ "${#COMPARE_ARGS[@]}" -gt 0 ]; then
  python scripts/compare_protocol_ablation.py "${COMPARE_ARGS[@]}" --out "reports/stage19_protocol_ablation_compare_${TS}.md" --csv-out "reports/stage19_protocol_ablation_compare_${TS}.csv" --force-collect > "reports/package_compare_protocol_${TS}.out" 2> "reports/package_compare_protocol_${TS}.err" || true
fi

for root in $ROOTS $CONTEXT_ROOTS; do
  [ -d "$root" ] || continue
  du -sh "$root" > "$META/${root}_du.txt" 2>&1 || true
  find "$root" -maxdepth 8 -type f -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' | sort -k3 > "$META/${root}_file_inventory.tsv" 2>/dev/null || true
  if [ -d "$root/_analysis" ]; then find "$root/_analysis" -maxdepth 3 -type f \( -name "*.csv" -o -name "*.md" -o -name "*.txt" -o -name "*.json" \) -print0 | while IFS= read -r -d '' f; do copy_file "$f"; done; fi
  for sub in _jobs _monitor; do [ -d "$root/$sub" ] && find "$root/$sub" -maxdepth 4 -type f -print0 | while IFS= read -r -d '' f; do copy_log_file "$f"; done; done
  find "$root" -type f \( -path "*/logs/*.csv" -o -name "config.json" -o -name "job.json" -o -name "manifest.json" -o -name "stdout.log" -o -name "stderr.log" \) ! -path "*/cache/*" ! -path "*/checkpoints/*" ! -path "*/archives/*" -print0 | while IFS= read -r -d '' f; do case "$(basename "$f")" in stdout.log|stderr.log) copy_log_file "$f";; *) copy_file "$f";; esac; done
done

if [ -d reports ]; then find reports -maxdepth 4 -type f \( -name "stage19*" -o -name "stage18*" -o -name "stage17*" -o -name "*.md" -o -name "*.txt" -o -name "*.csv" -o -name "*.patch" -o -name "*.out" -o -name "*.err" \) -print0 | while IFS= read -r -d '' f; do copy_file "$f"; done; fi
if [ -d configs ]; then find configs -maxdepth 5 -type f \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" \) -print0 | while IFS= read -r -d '' f; do copy_file "$f"; done; fi

tar --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.npz' --exclude='*.npy' --exclude='*.hdf5' --exclude='*.tar.gz' -czf "${PKG_DIR}/source_snapshot.tar.gz" bars scripts configs README.md setup.py .gitignore 2>/dev/null || true
cat > "${PKG_DIR}/UPLOAD_README.txt" <<EOF
BARS Stage19 protocol ablation results package.

Primary files:
  reports/stage19_protocol_ablation_compare_*.md
  reports/stage19_protocol_ablation_compare_*.csv
  each root/_analysis/eval_all.csv
  per-run logs/eval.csv
  metadata/git_status.txt, git_diff.patch, nvidia_smi.txt

Excluded:
  cache/, checkpoints/, archives/, *.pt/*.npz/*.npy/*.hdf5
EOF
OUT_TAR="${OUT_ROOT}/${PKG_NAME}.tar.gz"
tar -C "$TMP_ROOT" -czf "$OUT_TAR" "$PKG_NAME"
echo "$PKG_DIR" > "${OUT_ROOT}/${PKG_NAME}.tmpdir.txt"
echo "PACKAGE_CREATED=${OUT_TAR}"
du -h "$OUT_TAR" || true
