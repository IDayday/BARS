#!/usr/bin/env bash
set -Eeuo pipefail

OUT_ROOT="${1:-analysis_packages}"
TS="$(date +%Y%m%d_%H%M%S)"
PKG_NAME="bars_stage16_results_${TS}"
TMP_ROOT="$(mktemp -d /tmp/bars_stage16_pack.XXXXXX)"
PKG_DIR="${TMP_ROOT}/${PKG_NAME}"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-5242880}"
TAIL_LINES="${TAIL_LINES:-2000}"

mkdir -p "${PKG_DIR}" "${OUT_ROOT}" reports

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

for root in runs_stage16_graph_timing4 runs_stage16_full12; do
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

if [ -d reports ]; then
  find reports -maxdepth 2 -type f \( \
      -name "stage16*" -o \
      -name "*.md" -o \
      -name "*.txt" -o \
      -name "*.patch" -o \
      -name "*.out" -o \
      -name "*.err" \
    \) -print0 \
    | while IFS= read -r -d '' f; do copy_file "$f"; done
fi

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
BARS Stage 1.6 results package.

Primary files:
  reports/stage16_graph_timing4_report.md
  reports/stage16_graph_timing4_profile_summary.txt
  reports/stage16_graph_timing4_decision.md
  reports/stage16_full12_report.md
  reports/stage16_full12_profile_summary.txt
  reports/stage16_full12_decision.md
  runs_stage16_graph_timing4/_analysis/*.csv
  runs_stage16_full12/_analysis/*.csv
  per-run logs/*.csv
  metadata/git_status.txt
  metadata/git_diff.patch
  metadata/nvidia_smi.txt

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
