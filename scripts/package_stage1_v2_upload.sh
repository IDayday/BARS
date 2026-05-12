#!/usr/bin/env bash
set -Eeuo pipefail

LOG_ROOT="${1:-runs_stage1_diag_v2}"
OUT_ROOT="${2:-analysis_packages}"
GPUS="${GPUS:-0,1,2,3}"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-5242880}"   # 5MB
TAIL_LINES="${TAIL_LINES:-2000}"
INCLUDE_SOURCE="${INCLUDE_SOURCE:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
PKG_NAME="bars_stage1_v2_upload_${TS}"
TMP_ROOT="$(mktemp -d /tmp/bars_stage1_v2_pack.XXXXXX)"
PKG_DIR="${TMP_ROOT}/${PKG_NAME}"

mkdir -p "${PKG_DIR}" "${OUT_ROOT}" reports

echo "[INFO] Project root: $(pwd)"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] Package dir=${PKG_DIR}"

if [ ! -d "${LOG_ROOT}" ]; then
  echo "[ERROR] LOG_ROOT does not exist: ${LOG_ROOT}" >&2
  exit 1
fi

# ---------------------------------------------------------------------
# 0. Best-effort aggregate analysis before packaging
# ---------------------------------------------------------------------
echo "[INFO] Running collect/analyze if scripts exist..."

if [ -f scripts/collect_csv.py ]; then
  python scripts/collect_csv.py --log-root "${LOG_ROOT}" \
    > "reports/package_collect_stage1_v2_${TS}.out" \
    2> "reports/package_collect_stage1_v2_${TS}.err" || true
fi

if [ -f scripts/analyze_bars_results.py ]; then
  python scripts/analyze_bars_results.py \
    --log-root "${LOG_ROOT}" \
    --stage stage1 \
    --out "reports/stage1_diagnostics_v2_autopack_${TS}.md" \
    > "reports/package_analyze_stage1_v2_${TS}.out" \
    2> "reports/package_analyze_stage1_v2_${TS}.err" || true
fi

# ---------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------
META_DIR="${PKG_DIR}/metadata"
mkdir -p "${META_DIR}"

echo "[INFO] Collecting metadata..."

date > "${META_DIR}/packaged_at.txt"
pwd > "${META_DIR}/project_path.txt"
hostname > "${META_DIR}/hostname.txt" 2>&1 || true
uname -a > "${META_DIR}/uname.txt" 2>&1 || true
python --version > "${META_DIR}/python_version.txt" 2>&1 || true
which python > "${META_DIR}/which_python.txt" 2>&1 || true
pip freeze > "${META_DIR}/pip_freeze.txt" 2>&1 || true

nvidia-smi > "${META_DIR}/nvidia_smi.txt" 2>&1 || true
df -h > "${META_DIR}/disk_free.txt" 2>&1 || true
du -sh "${LOG_ROOT}" > "${META_DIR}/log_root_du.txt" 2>&1 || true

git status --short > "${META_DIR}/git_status_short.txt" 2>&1 || true
git status > "${META_DIR}/git_status.txt" 2>&1 || true
git branch --show-current > "${META_DIR}/git_branch.txt" 2>&1 || true
git log --oneline -30 > "${META_DIR}/git_log_oneline_30.txt" 2>&1 || true
git rev-parse HEAD > "${META_DIR}/git_head.txt" 2>&1 || true
git diff > "${META_DIR}/git_diff.patch" 2>&1 || true
git diff --staged > "${META_DIR}/git_diff_staged.patch" 2>&1 || true

pgrep -af "python -m bars.cli run" > "${META_DIR}/current_bars_processes.txt" 2>&1 || true

python -m bars.sched.jobctl status \
  --log-root "${LOG_ROOT}" \
  --gpus "${GPUS}" \
  > "${META_DIR}/jobctl_status.txt" 2>&1 || true

find "${LOG_ROOT}" -maxdepth 8 -type f \
  -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' \
  | sort -k3 \
  > "${META_DIR}/log_root_file_inventory.tsv" 2>/dev/null || true

find "${LOG_ROOT}" -maxdepth 6 -type d \
  > "${META_DIR}/log_root_dirs.txt" 2>/dev/null || true

# ---------------------------------------------------------------------
# 2. Copy configs
# ---------------------------------------------------------------------
echo "[INFO] Copying configs..."

if [ -d configs ]; then
  mkdir -p "${PKG_DIR}/configs"
  find configs -maxdepth 4 -type f \( \
      -name "*.json" -o \
      -name "*.yaml" -o \
      -name "*.yml" -o \
      -name "*.toml" \
    \) -print0 \
    | while IFS= read -r -d '' f; do
        dest="${PKG_DIR}/${f}"
        mkdir -p "$(dirname "${dest}")"
        cp -a "${f}" "${dest}"
      done
fi

# ---------------------------------------------------------------------
# 3. Copy reports
# ---------------------------------------------------------------------
echo "[INFO] Copying reports..."

if [ -d reports ]; then
  mkdir -p "${PKG_DIR}/reports"
  find reports -maxdepth 2 -type f \( \
      -name "*.md" -o \
      -name "*.txt" -o \
      -name "*.csv" -o \
      -name "*.out" -o \
      -name "*.err" -o \
      -name "*.log" \
    \) -print0 \
    | while IFS= read -r -d '' f; do
        dest="${PKG_DIR}/${f}"
        mkdir -p "$(dirname "${dest}")"
        cp -a "${f}" "${dest}"
      done
fi

# ---------------------------------------------------------------------
# 4. Copy _analysis
# ---------------------------------------------------------------------
echo "[INFO] Copying ${LOG_ROOT}/_analysis..."

if [ -d "${LOG_ROOT}/_analysis" ]; then
  mkdir -p "${PKG_DIR}/${LOG_ROOT}/_analysis"
  find "${LOG_ROOT}/_analysis" -maxdepth 3 -type f \( \
      -name "*.csv" -o \
      -name "*.md" -o \
      -name "*.txt" -o \
      -name "*.json" \
    \) -print0 \
    | while IFS= read -r -d '' f; do
        dest="${PKG_DIR}/${f}"
        mkdir -p "$(dirname "${dest}")"
        cp -a "${f}" "${dest}"
      done
else
  echo "[WARN] ${LOG_ROOT}/_analysis does not exist." | tee -a "${META_DIR}/warnings.txt"
fi

# ---------------------------------------------------------------------
# 5. Copy scheduler / monitor logs
# ---------------------------------------------------------------------
echo "[INFO] Copying _jobs and _monitor..."

for subdir in "_jobs" "_monitor"; do
  if [ -d "${LOG_ROOT}/${subdir}" ]; then
    find "${LOG_ROOT}/${subdir}" -maxdepth 4 -type f -print0 \
      | while IFS= read -r -d '' f; do
          dest="${PKG_DIR}/${f}"
          mkdir -p "$(dirname "${dest}")"
          size="$(stat -c%s "${f}" 2>/dev/null || echo 0)"
          if [ "${size}" -le "${MAX_LOG_BYTES}" ]; then
            cp -a "${f}" "${dest}"
          else
            tail -n "${TAIL_LINES}" "${f}" > "${dest}.tail${TAIL_LINES}"
            echo "TRUNCATED original_size_bytes=${size}" > "${dest}.TRUNCATED"
          fi
        done
  fi
done

# ---------------------------------------------------------------------
# 6. Copy per-run logs and metadata.
#    Do NOT copy cache/checkpoints/archives by default.
# ---------------------------------------------------------------------
echo "[INFO] Copying per-run logs/configs/manifests/stdout/stderr..."

find "${LOG_ROOT}" -type f \( \
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
      dest="${PKG_DIR}/${f}"
      mkdir -p "$(dirname "${dest}")"
      base="$(basename "${f}")"
      size="$(stat -c%s "${f}" 2>/dev/null || echo 0)"

      if [[ "${base}" == "stdout.log" || "${base}" == "stderr.log" ]]; then
        if [ "${size}" -le "${MAX_LOG_BYTES}" ]; then
          cp -a "${f}" "${dest}"
        else
          tail -n "${TAIL_LINES}" "${f}" > "${dest}.tail${TAIL_LINES}"
          echo "TRUNCATED original_size_bytes=${size}" > "${dest}.TRUNCATED"
        fi
      else
        cp -a "${f}" "${dest}"
      fi
    done

# Archive inventory only.
find "${LOG_ROOT}" -type f -path "*/archives/*" \
  -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' \
  | sort -k3 \
  > "${META_DIR}/archive_inventory.tsv" 2>/dev/null || true

# Cache/checkpoint inventory only.
find "${LOG_ROOT}" -type f \( \
    -path "*/cache/*" -o \
    -path "*/checkpoints/*" \
  \) \
  -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' \
  | sort -k3 \
  > "${META_DIR}/excluded_cache_checkpoint_inventory.tsv" 2>/dev/null || true

# ---------------------------------------------------------------------
# 7. Quick summary script
# ---------------------------------------------------------------------
echo "[INFO] Creating quick summary..."

cat > "${META_DIR}/make_quick_summary.py" <<'PY'
from pathlib import Path
import json
import os
import traceback
import pandas as pd

log_root = Path(os.environ["BARS_LOG_ROOT"])
pkg_dir = Path(os.environ["BARS_PKG_DIR"])
out_path = pkg_dir / "metadata" / "stage1_v2_quick_summary.txt"

lines = []
lines.append(f"LOG_ROOT={log_root}")
lines.append(f"PKG_DIR={pkg_dir}")

def block(title, body):
    lines.append("")
    lines.append("=" * 100)
    lines.append(title)
    lines.append("=" * 100)
    lines.append(str(body))

def safe_read_csv(path):
    if not path.exists():
        block(str(path), "MISSING")
        return None
    try:
        df = pd.read_csv(path)
        block(f"{path} columns", list(df.columns))
        block(f"{path} shape", df.shape)
        return df
    except Exception:
        block(f"{path} read error", traceback.format_exc())
        return None

try:
    analysis = log_root / "_analysis"

    summary = safe_read_csv(analysis / "summary_all.csv")
    diag = safe_read_csv(analysis / "diagnostics_all.csv")
    graph = safe_read_csv(analysis / "graph_all.csv")
    eval_df = safe_read_csv(analysis / "eval_all.csv")

    if summary is not None and len(summary):
        for group_cols in [["env", "status"], ["status"], ["env", "phase"], ["phase"]]:
            cols = [c for c in group_cols if c in summary.columns]
            if cols:
                block(f"summary groupby {cols}", summary.groupby(cols).size())

        tail_cols = [c for c in [
            "run_id", "env", "seed", "variant", "node_method",
            "phase", "status", "message", "timestamp"
        ] if c in summary.columns]
        if tail_cols:
            block("summary tail", summary[tail_cols].tail(80).to_string(index=False))

    if diag is not None and len(diag):
        if "phase" in diag.columns:
            block("diagnostics phase counts", diag["phase"].value_counts(dropna=False))

        edge = diag[diag["phase"].eq("edge_diag")].copy() if "phase" in diag.columns else pd.DataFrame()
        if len(edge):
            cols = [c for c in [
                "reach_auc_proxy",
                "reach_auprc_proxy",
                "false_positive_proxy_rate",
                "reachable_edge_coverage_proxy",
                "selected_edges",
                "num_edges"
            ] if c in edge.columns]
            groups = [c for c in ["env"] if c in edge.columns]
            if cols and groups:
                block("edge_diag mean by env", edge.groupby(groups)[cols].mean(numeric_only=True).round(4))
                block("edge_diag std by env", edge.groupby(groups)[cols].std(numeric_only=True).round(4))

        path = diag[diag["phase"].eq("path_diag")].copy() if "phase" in diag.columns else pd.DataFrame()
        if len(path):
            cols = [c for c in [
                "found",
                "num_edges",
                "num_subgoals",
                "total_cost",
                "total_risk",
                "total_boundary",
                "objective"
            ] if c in path.columns]
            groups = [c for c in ["env", "variant"] if c in path.columns]
            if cols and groups:
                block("path_diag mean by env/variant", path.groupby(groups)[cols].mean(numeric_only=True).round(4))
                block("path_diag std by env/variant", path.groupby(groups)[cols].std(numeric_only=True).round(4))

        boundary = diag[diag["phase"].eq("boundary_diag")].copy() if "phase" in diag.columns else pd.DataFrame()
        if len(boundary):
            cols = [c for c in [
                "psi_mean", "psi_p10", "psi_p50", "psi_p90",
                "supported_pair_rate", "num_pairs"
            ] if c in boundary.columns]
            groups = [c for c in ["env"] if c in boundary.columns]
            if cols and groups:
                block("boundary_diag mean by env", boundary.groupby(groups)[cols].mean(numeric_only=True).round(4))
                block("boundary_diag std by env", boundary.groupby(groups)[cols].std(numeric_only=True).round(4))

    if graph is not None and len(graph):
        for group_cols in [["env", "event"], ["env", "phase"], ["event"], ["phase"]]:
            cols_group = [c for c in group_cols if c in graph.columns]
            metric_cols = [c for c in [
                "num_nodes", "num_edges", "num_candidate_edges", "num_kept_edges",
                "duration_sec"
            ] if c in graph.columns]
            if cols_group and metric_cols:
                block(f"graph groupby {cols_group}", graph.groupby(cols_group)[metric_cols].mean(numeric_only=True).round(4))

    if eval_df is not None and len(eval_df):
        metric_cols = [c for c in [
            "success", "return", "steps", "replans",
            "no_path_count", "last_plan_edges", "goal_distance_final"
        ] if c in eval_df.columns]
        group_cols = [c for c in ["env", "variant"] if c in eval_df.columns]
        if metric_cols and group_cols:
            block("eval mean by env/variant", eval_df.groupby(group_cols)[metric_cols].mean(numeric_only=True).round(4))

    manifests = sorted(log_root.glob("**/manifest.json"))
    block("manifest_count", len(manifests))
    rows = []
    for p in manifests:
        try:
            m = json.loads(p.read_text())
            rows.append({
                "path": str(p),
                "status": m.get("status"),
                "env": m.get("env"),
                "seed": m.get("seed"),
                "variant": m.get("variant"),
                "node_method": m.get("node_method"),
                "run_id": m.get("run_id"),
            })
        except Exception as exc:
            rows.append({"path": str(p), "error": repr(exc)})
    if rows:
        df = pd.DataFrame(rows)
        block("manifest rows", df.to_string(index=False))
        if "status" in df.columns:
            block("manifest status counts", df["status"].value_counts(dropna=False))

except Exception:
    block("quick summary error", traceback.format_exc())

out_path.write_text("\n".join(lines) + "\n")
print(out_path)
PY

BARS_LOG_ROOT="${LOG_ROOT}" \
BARS_PKG_DIR="${PKG_DIR}" \
python "${META_DIR}/make_quick_summary.py" || true

# ---------------------------------------------------------------------
# 8. Source snapshot
# ---------------------------------------------------------------------
if [ "${INCLUDE_SOURCE}" = "1" ]; then
  echo "[INFO] Creating source snapshot..."

  declare -a SRC_ITEMS=()
  for item in bars configs scripts examples reports LICENSE README.md README.rst pyproject.toml setup.py setup.cfg .gitignore; do
    if [ -e "${item}" ]; then
      SRC_ITEMS+=("${item}")
    fi
  done

  if [ "${#SRC_ITEMS[@]}" -gt 0 ]; then
    tar \
      --exclude='__pycache__' \
      --exclude='.pytest_cache' \
      --exclude='.ipynb_checkpoints' \
      --exclude='*.pyc' \
      --exclude='*.pt' \
      --exclude='*.pth' \
      --exclude='*.ckpt' \
      --exclude='*.npz' \
      --exclude='*.npy' \
      --exclude='*.hdf5' \
      --exclude='*.tar.gz' \
      -czf "${PKG_DIR}/source_snapshot.tar.gz" \
      "${SRC_ITEMS[@]}" 2>/dev/null || true
  fi
fi

# ---------------------------------------------------------------------
# 9. Upload README
# ---------------------------------------------------------------------
cat > "${PKG_DIR}/UPLOAD_README.txt" <<EOF
This package is for BARS Stage 1 v2 analysis.

Primary files to inspect:
  metadata/stage1_v2_quick_summary.txt
  metadata/jobctl_status.txt
  metadata/git_status.txt
  metadata/git_diff.patch
  ${LOG_ROOT}/_analysis/*.csv
  ${LOG_ROOT}/_jobs/*
  ${LOG_ROOT}/_monitor/*
  per-run logs/*.csv
  per-run stdout.log / stderr.log or .tail${TAIL_LINES}
  reports/stage1*.md

Excluded by default:
  cache/
  checkpoints/
  archives/
  *.pt
  *.npz
  *.npy
  *.hdf5
EOF

# ---------------------------------------------------------------------
# 10. Final tarball
# ---------------------------------------------------------------------
OUT_TAR="${OUT_ROOT}/${PKG_NAME}.tar.gz"

echo "[INFO] Creating final tarball: ${OUT_TAR}"
tar -C "${TMP_ROOT}" -czf "${OUT_TAR}" "${PKG_NAME}"

echo "${PKG_DIR}" > "${OUT_ROOT}/${PKG_NAME}.tmpdir.txt"

echo
echo "============================================================"
echo "Package created:"
echo "  ${OUT_TAR}"
echo
echo "Size:"
du -h "${OUT_TAR}" || true
echo
echo "Quick summary inside package:"
echo "  ${PKG_NAME}/metadata/stage1_v2_quick_summary.txt"
echo
echo "Temporary unpacked package dir:"
echo "  ${PKG_DIR}"
echo
echo "Upload this .tar.gz to ChatGPT."
echo "============================================================"
