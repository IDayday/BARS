#!/usr/bin/env bash
set -Eeuo pipefail

BASE_ROOT="${BASE_ROOT:-runs_stage16_full12}"
PU_ROOT="${PU_ROOT:-runs_stage17_pu_retrain4}"
OUT_ROOT="${OUT_ROOT:-analysis_packages}"
TS="$(date +%Y%m%d_%H%M%S)"
PKG_NAME="bars_stage17_pu_analysis_${TS}"
TMP_ROOT="$(mktemp -d /tmp/bars_stage17_pack.XXXXXX)"
PKG_DIR="${TMP_ROOT}/${PKG_NAME}"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-5242880}"
TAIL_LINES="${TAIL_LINES:-2000}"

mkdir -p "$PKG_DIR" "$OUT_ROOT" reports

copy_file() {
  local f="$1"
  local dest="${PKG_DIR}/${f}"
  mkdir -p "$(dirname "$dest")"
  cp -a "$f" "$dest"
}

copy_log_file() {
  local f="$1"
  local dest="${PKG_DIR}/${f}"
  mkdir -p "$(dirname "$dest")"
  local size
  size="$(stat -c%s "$f" 2>/dev/null || echo 0)"
  if [ "$size" -le "$MAX_LOG_BYTES" ]; then
    cp -a "$f" "$dest"
  else
    tail -n "$TAIL_LINES" "$f" > "${dest}.tail${TAIL_LINES}"
    echo "TRUNCATED original_size_bytes=${size}" > "${dest}.TRUNCATED"
  fi
}

# Best-effort collect/analyze.
if [ -f scripts/collect_csv.py ]; then
  python scripts/collect_csv.py --log-root "$BASE_ROOT" > "reports/package_collect_base_${TS}.out" 2> "reports/package_collect_base_${TS}.err" || true
  python scripts/collect_csv.py --log-root "$PU_ROOT" > "reports/package_collect_pu_${TS}.out" 2> "reports/package_collect_pu_${TS}.err" || true
fi

if [ -f scripts/analyze_bars_results.py ]; then
  python scripts/analyze_bars_results.py --log-root "$BASE_ROOT" --stage stage1 --out "reports/stage17_loaded_baseline_reanalysis_${TS}.md" --force-collect > "reports/package_analyze_base_${TS}.out" 2> "reports/package_analyze_base_${TS}.err" || true
  python scripts/analyze_bars_results.py --log-root "$PU_ROOT" --stage stage1 --out "reports/stage17_pu_retrain4_report_${TS}.md" --force-collect > "reports/package_analyze_pu_${TS}.out" 2> "reports/package_analyze_pu_${TS}.err" || true
fi

if [ -f scripts/compare_stage17_pu_vs_loaded.py ]; then
  python scripts/compare_stage17_pu_vs_loaded.py \
    --loaded-root "$BASE_ROOT" \
    --pu-root "$PU_ROOT" \
    --out "reports/stage17_pu_vs_loaded_compare_${TS}.md" \
    --csv-out-dir "reports/stage17_compare_csv_${TS}" \
    > "reports/package_compare_${TS}.out" \
    2> "reports/package_compare_${TS}.err" || true
fi

META="${PKG_DIR}/metadata"
mkdir -p "$META"

date > "$META/packaged_at.txt"
pwd > "$META/project_path.txt"
hostname > "$META/hostname.txt" 2>&1 || true
uname -a > "$META/uname.txt" 2>&1 || true
python --version > "$META/python_version.txt" 2>&1 || true
which python > "$META/which_python.txt" 2>&1 || true
pip freeze > "$META/pip_freeze.txt" 2>&1 || true
nvidia-smi > "$META/nvidia_smi.txt" 2>&1 || true
df -h > "$META/disk_free.txt" 2>&1 || true

git status > "$META/git_status.txt" 2>&1 || true
git status --short > "$META/git_status_short.txt" 2>&1 || true
git branch --show-current > "$META/git_branch.txt" 2>&1 || true
git rev-parse HEAD > "$META/git_head.txt" 2>&1 || true
git log --oneline -30 > "$META/git_log_oneline_30.txt" 2>&1 || true
git diff > "$META/git_diff.patch" 2>&1 || true
git diff --staged > "$META/git_diff_staged.patch" 2>&1 || true

pgrep -af "python -m bars.cli run" > "$META/current_bars_processes.txt" 2>&1 || true

for root in "$BASE_ROOT" "$PU_ROOT"; do
  if [ -d "$root" ]; then
    du -sh "$root" > "$META/${root}_du.txt" 2>&1 || true
    find "$root" -maxdepth 8 -type f -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' | sort -k3 > "$META/${root}_file_inventory.tsv" 2>/dev/null || true

    if [ -d "$root/_analysis" ]; then
      find "$root/_analysis" -maxdepth 3 -type f \( -name "*.csv" -o -name "*.md" -o -name "*.txt" -o -name "*.json" \) -print0 \
      | while IFS= read -r -d '' f; do copy_file "$f"; done
    fi

    for sub in _jobs _monitor; do
      if [ -d "$root/$sub" ]; then
        find "$root/$sub" -maxdepth 4 -type f -print0 \
        | while IFS= read -r -d '' f; do copy_log_file "$f"; done
      fi
    done

    find "$root" -type f \( \
        -path "*/logs/*.csv" -o \
        -name "config.json" -o \
        -name "job.json" -o \
        -name "manifest.json" -o \
        -name "stdout.log" -o \
        -name "stderr.log" \
      \) \
      ! -path "*/cache/*" \
      ! -path "*/checkpoints/*" \
      ! -path "*/archives/*" \
      -print0 \
      | while IFS= read -r -d '' f; do
          case "$(basename "$f")" in
            stdout.log|stderr.log) copy_log_file "$f" ;;
            *) copy_file "$f" ;;
          esac
        done
  fi
done

# Reports.
if [ -d reports ]; then
  find reports -maxdepth 3 -type f \( \
      -name "stage17*" -o \
      -name "stage16*" -o \
      -name "*.md" -o \
      -name "*.txt" -o \
      -name "*.csv" -o \
      -name "*.patch" -o \
      -name "*.out" -o \
      -name "*.err" \
    \) -print0 \
  | while IFS= read -r -d '' f; do copy_file "$f"; done
fi

# Source/config snapshot, excluding large artifacts.
for d in configs bars scripts examples; do
  if [ -d "$d" ]; then
    mkdir -p "${PKG_DIR}/$d"
    tar \
      --exclude='__pycache__' \
      --exclude='.pytest_cache' \
      --exclude='*.pyc' \
      --exclude='*.pt' \
      --exclude='*.pth' \
      --exclude='*.ckpt' \
      --exclude='*.npz' \
      --exclude='*.npy' \
      --exclude='*.hdf5' \
      -cf - "$d" | tar -C "$PKG_DIR" -xf -
  fi
done

for f in README.md README.rst pyproject.toml setup.py setup.cfg .gitignore; do
  [ -f "$f" ] && copy_file "$f"
done

cat > "${PKG_DIR}/UPLOAD_README.txt" <<EOF
BARS Stage17 PU analysis package.

Includes:
  ${BASE_ROOT}/_analysis/*.csv
  ${PU_ROOT}/_analysis/*.csv
  per-run logs/*.csv
  config/job/manifest/stdout/stderr tails
  reports/stage17*.md
  reports/stage17_compare_csv*
  source/config snapshot
  metadata/git/nvidia/pip info

Excluded:
  cache/
  checkpoints/
  archives/
  *.pt/*.pth/*.ckpt/*.npz/*.npy/*.hdf5
EOF

OUT_TAR="${OUT_ROOT}/${PKG_NAME}.tar.gz"
tar -C "$TMP_ROOT" -czf "$OUT_TAR" "$PKG_NAME"

echo "$PKG_DIR" > "${OUT_ROOT}/${PKG_NAME}.tmpdir.txt"
echo "PACKAGE_CREATED=${OUT_TAR}"
du -h "$OUT_TAR" || true
