#!/usr/bin/env bash
set -Eeuo pipefail

# Main route-A log roots. Missing roots are skipped automatically.
ROOTS="${ROOTS:-runs_stage17_online_quick_loaded runs_stage18_online_tuned_h50_thr1_lb01 runs_stage18_online_tuned_h50_thr1_lb03_fullbars runs_stage18_medium100_best_loaded runs_stage18_large_quick_bars_lite}"

# Optional context roots. These help me compare online results to offline diagnostics / edge rollout / PU retrain.
CONTEXT_ROOTS="${CONTEXT_ROOTS:-runs_stage16_full12 runs_stage17_pu_retrain4}"

OUT_ROOT="${OUT_ROOT:-analysis_packages}"
GPUS="${GPUS:-0,1,2,3,4,5,6}"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-5242880}"   # 5 MB
TAIL_LINES="${TAIL_LINES:-2000}"
INCLUDE_SOURCE="${INCLUDE_SOURCE:-1}"
INCLUDE_CONTEXT="${INCLUDE_CONTEXT:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
PKG_NAME="bars_stage18_routeA_results_${TS}"
TMP_ROOT="$(mktemp -d /tmp/bars_stage18_pack.XXXXXX)"
PKG_DIR="${TMP_ROOT}/${PKG_NAME}"

mkdir -p "${PKG_DIR}" "${OUT_ROOT}" reports

echo "[INFO] Project: $(pwd)"
echo "[INFO] ROOTS=${ROOTS}"
echo "[INFO] CONTEXT_ROOTS=${CONTEXT_ROOTS}"
echo "[INFO] Package dir: ${PKG_DIR}"

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

root_exists() {
  local r="$1"
  [ -d "$r" ]
}

# ---------------------------------------------------------------------
# 0. Build root list.
# ---------------------------------------------------------------------
declare -a MAIN_ROOT_ARRAY=()
declare -a CONTEXT_ROOT_ARRAY=()
declare -a ALL_ROOT_ARRAY=()

read -r -a _tmp_main <<< "$ROOTS"
for r in "${_tmp_main[@]}"; do
  if root_exists "$r"; then
    MAIN_ROOT_ARRAY+=("$r")
    ALL_ROOT_ARRAY+=("$r")
  else
    echo "[WARN] Missing main root, skipped: $r"
  fi
done

if [ "$INCLUDE_CONTEXT" = "1" ]; then
  read -r -a _tmp_ctx <<< "$CONTEXT_ROOTS"
  for r in "${_tmp_ctx[@]}"; do
    if root_exists "$r"; then
      CONTEXT_ROOT_ARRAY+=("$r")
      ALL_ROOT_ARRAY+=("$r")
    else
      echo "[WARN] Missing context root, skipped: $r"
    fi
  done
fi

if [ "${#MAIN_ROOT_ARRAY[@]}" -eq 0 ]; then
  echo "[ERROR] No main Stage18/online roots found. Set ROOTS='...' and retry." >&2
  exit 1
fi

printf "%s\n" "${MAIN_ROOT_ARRAY[@]}" > reports/stage18_routeA_main_roots_${TS}.txt
printf "%s\n" "${CONTEXT_ROOT_ARRAY[@]:-}" > reports/stage18_routeA_context_roots_${TS}.txt || true
printf "%s\n" "${ALL_ROOT_ARRAY[@]}" > reports/stage18_routeA_all_roots_${TS}.txt

# ---------------------------------------------------------------------
# 1. Metadata.
# ---------------------------------------------------------------------
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
free -h > "$META/memory_free.txt" 2>&1 || true
nproc > "$META/nproc.txt" 2>&1 || true

git status > "$META/git_status.txt" 2>&1 || true
git status --short > "$META/git_status_short.txt" 2>&1 || true
git branch --show-current > "$META/git_branch.txt" 2>&1 || true
git rev-parse HEAD > "$META/git_head.txt" 2>&1 || true
git log --oneline -40 > "$META/git_log_oneline_40.txt" 2>&1 || true
git diff > "$META/git_diff.patch" 2>&1 || true
git diff --staged > "$META/git_diff_staged.patch" 2>&1 || true

pgrep -af "python -m bars.cli run" > "$META/current_bars_processes.txt" 2>&1 || true

# ---------------------------------------------------------------------
# 2. Best-effort collect/analyze.
# ---------------------------------------------------------------------
echo "[INFO] Running collect/analyze..."

for r in "${MAIN_ROOT_ARRAY[@]}"; do
  echo "[INFO] Collect/analyze main root: $r"
  if [ -f scripts/collect_csv.py ]; then
    python scripts/collect_csv.py --log-root "$r" \
      > "reports/package_collect_${r}_${TS}.out" \
      2> "reports/package_collect_${r}_${TS}.err" || true
  fi

  if [ -f scripts/analyze_bars_results.py ]; then
    python scripts/analyze_bars_results.py \
      --log-root "$r" \
      --stage stage3 \
      --out "reports/${r}_stage3_autopack_${TS}.md" \
      --force-collect \
      > "reports/package_analyze_${r}_${TS}.out" \
      2> "reports/package_analyze_${r}_${TS}.err" || true
  fi
done

for r in "${CONTEXT_ROOT_ARRAY[@]:-}"; do
  echo "[INFO] Collect/analyze context root: $r"
  if [ -f scripts/collect_csv.py ]; then
    python scripts/collect_csv.py --log-root "$r" \
      > "reports/package_collect_${r}_${TS}.out" \
      2> "reports/package_collect_${r}_${TS}.err" || true
  fi

  # Context roots are usually Stage1 diagnostics or PU retrain; stage1 is safest.
  if [ -f scripts/analyze_bars_results.py ]; then
    python scripts/analyze_bars_results.py \
      --log-root "$r" \
      --stage stage1 \
      --out "reports/${r}_stage1_context_autopack_${TS}.md" \
      --force-collect \
      > "reports/package_analyze_${r}_${TS}.out" \
      2> "reports/package_analyze_${r}_${TS}.err" || true
  fi
done

# Compare online roots if script exists.
if [ -f scripts/compare_online_results.py ]; then
  COMPARE_ARGS=()
  for r in "${MAIN_ROOT_ARRAY[@]}"; do
    label="$(basename "$r" | sed 's/^runs_//')"
    COMPARE_ARGS+=(--root "${label}=${r}")
  done
  if [ "${#COMPARE_ARGS[@]}" -gt 0 ]; then
    python scripts/compare_online_results.py \
      "${COMPARE_ARGS[@]}" \
      --out "reports/stage18_routeA_online_compare_${TS}.md" \
      --csv-out "reports/stage18_routeA_online_compare_${TS}.csv" \
      > "reports/package_compare_online_${TS}.out" \
      2> "reports/package_compare_online_${TS}.err" || true
  fi
fi

# ---------------------------------------------------------------------
# 3. Create quick summary.
# ---------------------------------------------------------------------
cat > "${META}/make_routeA_quick_summary.py" <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import traceback

roots = os.environ.get("BARS_ROOTS", "").split()
context_roots = os.environ.get("BARS_CONTEXT_ROOTS", "").split()
pkg_dir = Path(os.environ["BARS_PKG_DIR"])
out = pkg_dir / "metadata" / "stage18_routeA_quick_summary.txt"

lines = []

def block(title, body=""):
    lines.append("")
    lines.append("=" * 100)
    lines.append(title)
    lines.append("=" * 100)
    lines.append(str(body))

def read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        block(f"READ ERROR {path}", traceback.format_exc())
        return None

def summarize_online_root(root: Path):
    block(f"ONLINE ROOT: {root}")

    summary = read_csv(root / "_analysis" / "summary_all.csv")
    eval_df = read_csv(root / "_analysis" / "eval_all.csv")
    diag = read_csv(root / "_analysis" / "diagnostics_all.csv")

    if summary is not None:
        block(f"{root} summary shape", summary.shape)
        if "status" in summary.columns:
            block(f"{root} status counts", summary["status"].value_counts(dropna=False))
        group_cols = [c for c in ["env", "variant", "status"] if c in summary.columns]
        if group_cols:
            block(f"{root} status by env/variant", summary.groupby(group_cols).size())
        tail_cols = [c for c in ["env", "seed", "variant", "phase", "status", "message", "timestamp"] if c in summary.columns]
        if tail_cols:
            block(f"{root} summary tail", summary[tail_cols].tail(80).to_string(index=False))

    if eval_df is not None:
        block(f"{root} eval shape", eval_df.shape)
        block(f"{root} eval columns", list(eval_df.columns))

        metrics = [c for c in [
            "success",
            "return",
            "steps",
            "replans",
            "no_path_count",
            "last_plan_edges",
            "goal_distance_final",
            "fallback_used",
            "fallback_count",
            "initial_plan_failed_count",
            "plan_failed_initial",
            "first_plan_edges",
            "max_plan_edges",
            "mean_plan_edges",
            "num_plan_calls",
            "num_subgoal_attempts",
            "num_subgoal_reached",
            "subgoal_reach_rate",
        ] if c in eval_df.columns]

        group_cols = [c for c in ["env", "variant"] if c in eval_df.columns]
        if metrics and group_cols:
            block(f"{root} eval mean by env/variant", eval_df.groupby(group_cols)[metrics].mean(numeric_only=True).round(4))
            block(f"{root} eval std by env/variant", eval_df.groupby(group_cols)[metrics].std(numeric_only=True).round(4))

        if "success" in eval_df.columns and "variant" in eval_df.columns:
            block(f"{root} overall success by variant", eval_df.groupby("variant")["success"].mean().round(4))
            if "seed" in eval_df.columns:
                block(f"{root} per-seed success", eval_df.groupby([c for c in ["env", "variant", "seed"] if c in eval_df.columns])["success"].mean().round(4))

        if "no_path_count" in eval_df.columns and "success" in eval_df.columns:
            x = eval_df.copy()
            x["has_no_path"] = x["no_path_count"].fillna(0) > 0
            block(f"{root} success by has_no_path", x.groupby([c for c in ["env", "variant", "has_no_path"] if c in x.columns])["success"].mean().round(4))

        if "last_plan_edges" in eval_df.columns and "success" in eval_df.columns:
            x = eval_df.copy()
            x["edge_bucket"] = pd.cut(
                x["last_plan_edges"].fillna(-1),
                bins=[-2, 0, 1, 2, 1000],
                labels=["0", "1", "2", "3+"],
            )
            block(f"{root} success by last_plan_edges bucket", x.groupby([c for c in ["variant", "edge_bucket"] if c in x.columns])["success"].mean().round(4))

    if diag is not None:
        if "phase" in diag.columns:
            block(f"{root} diagnostics phases", diag["phase"].value_counts(dropna=False))

def summarize_context_root(root: Path):
    block(f"CONTEXT ROOT: {root}")
    for name in ["diagnostics_all.csv", "eval_all.csv", "summary_all.csv", "profile_all.csv", "graph_all.csv"]:
        df = read_csv(root / "_analysis" / name)
        if df is not None:
            block(f"{root} {name} shape/columns", f"shape={df.shape}\ncolumns={list(df.columns)}")

    diag = read_csv(root / "_analysis" / "diagnostics_all.csv")
    if diag is not None and "phase" in diag.columns:
        block(f"{root} diagnostic phases", diag["phase"].value_counts(dropna=False))
        for phase, cols in {
            "edge_rollout_diag": [
                "edge_rollout_auc", "edge_rollout_auprc", "selected_edge_success_rate",
                "unselected_edge_success_rate", "success_rate_selected_supported",
                "success_rate_selected_hard_neg_proxy"
            ],
            "balanced_edge_diag": [
                "edge_auc_balanced", "edge_auprc_balanced", "selected_supported_rate",
                "selected_hard_neg_proxy_rate", "score_supported_mean", "score_hard_neg_proxy_mean"
            ],
            "boundary_diag": [
                "supported_pair_rate", "psi_mean", "psi_p10", "psi_p50", "psi_p90"
            ],
            "path_diag": [
                "found", "total_risk", "total_cost", "total_boundary", "objective", "lambda_risk"
            ],
        }.items():
            x = diag[diag["phase"].eq(phase)].copy()
            cols = [c for c in cols if c in x.columns]
            groups = [c for c in ["env", "variant", "lambda_risk"] if c in x.columns]
            if len(x) and cols:
                block(f"{root} {phase} mean", x.groupby(groups)[cols].mean(numeric_only=True).round(4) if groups else x[cols].mean(numeric_only=True).round(4))

try:
    block("MAIN ONLINE ROOTS", roots)
    for r in roots:
        p = Path(r)
        if p.exists():
            summarize_online_root(p)
        else:
            block(f"MISSING ONLINE ROOT {r}")

    block("CONTEXT ROOTS", context_roots)
    for r in context_roots:
        p = Path(r)
        if p.exists():
            summarize_context_root(p)
        else:
            block(f"MISSING CONTEXT ROOT {r}")

except Exception:
    block("ERROR", traceback.format_exc())

out.write_text("\n".join(lines) + "\n")
print(out)
PY

BARS_ROOTS="${MAIN_ROOT_ARRAY[*]}" \
BARS_CONTEXT_ROOTS="${CONTEXT_ROOT_ARRAY[*]:-}" \
BARS_PKG_DIR="${PKG_DIR}" \
python "${META}/make_routeA_quick_summary.py" || true

# ---------------------------------------------------------------------
# 4. Copy root contents: _analysis, jobs, monitor, per-run CSV logs.
# ---------------------------------------------------------------------
echo "[INFO] Copying log roots..."

for root in "${ALL_ROOT_ARRAY[@]}"; do
  if [ ! -d "$root" ]; then
    continue
  fi

  du -sh "$root" > "$META/${root}_du.txt" 2>&1 || true
  find "$root" -maxdepth 8 -type f \
    -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' \
    | sort -k3 \
    > "$META/${root}_file_inventory.tsv" 2>/dev/null || true

  if [ -d "$root/_analysis" ]; then
    find "$root/_analysis" -maxdepth 3 -type f \( \
        -name "*.csv" -o -name "*.md" -o -name "*.txt" -o -name "*.json" \
      \) -print0 \
      | while IFS= read -r -d '' f; do
          copy_file "$f"
        done
  fi

  for sub in _jobs _monitor; do
    if [ -d "$root/$sub" ]; then
      find "$root/$sub" -maxdepth 4 -type f -print0 \
        | while IFS= read -r -d '' f; do
            copy_log_file "$f"
          done
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
done

# ---------------------------------------------------------------------
# 5. Copy reports/configs/source snapshot.
# ---------------------------------------------------------------------
echo "[INFO] Copying reports/configs/source..."

if [ -d reports ]; then
  find reports -maxdepth 4 -type f \( \
      -name "stage18*" -o \
      -name "stage17*" -o \
      -name "stage16*" -o \
      -name "baseline_alignment*" -o \
      -name "*.md" -o \
      -name "*.txt" -o \
      -name "*.csv" -o \
      -name "*.patch" -o \
      -name "*.out" -o \
      -name "*.err" \
    \) -print0 \
    | while IFS= read -r -d '' f; do
        copy_file "$f"
      done
fi

# Copy configs fully; they are small and critical.
if [ -d configs ]; then
  find configs -maxdepth 5 -type f \( \
      -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" \
    \) -print0 \
    | while IFS= read -r -d '' f; do
        copy_file "$f"
      done
fi

if [ "$INCLUDE_SOURCE" = "1" ]; then
  SRC_TAR="${PKG_DIR}/source_snapshot.tar.gz"
  declare -a SRC_ITEMS=()
  for item in bars scripts examples configs README.md README.rst pyproject.toml setup.py setup.cfg .gitignore; do
    if [ -e "$item" ]; then
      SRC_ITEMS+=("$item")
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
      -czf "$SRC_TAR" \
      "${SRC_ITEMS[@]}" 2>/dev/null || true
  fi
fi

cat > "${PKG_DIR}/UPLOAD_README.txt" <<EOF
BARS Stage18 Route-A result package.

Main roots:
$(printf '  %s\n' "${MAIN_ROOT_ARRAY[@]}")

Context roots:
$(printf '  %s\n' "${CONTEXT_ROOT_ARRAY[@]:-}")

Primary files:
  metadata/stage18_routeA_quick_summary.txt
  reports/stage18_routeA_online_compare_*.md
  reports/stage18_routeA_online_compare_*.csv
  each root/_analysis/*.csv
  per-run logs/*.csv
  config.json / job.json / manifest.json
  stdout.log / stderr.log tails
  configs/
  source_snapshot.tar.gz

Excluded:
  cache/
  checkpoints/
  archives/
  *.pt/*.pth/*.ckpt/*.npz/*.npy/*.hdf5
EOF

# ---------------------------------------------------------------------
# 6. Final tarball.
# ---------------------------------------------------------------------
OUT_TAR="${OUT_ROOT}/${PKG_NAME}.tar.gz"
tar -C "$TMP_ROOT" -czf "$OUT_TAR" "$PKG_NAME"

echo "$PKG_DIR" > "${OUT_ROOT}/${PKG_NAME}.tmpdir.txt"

echo
echo "============================================================"
echo "PACKAGE_CREATED=${OUT_TAR}"
du -h "$OUT_TAR" || true
echo "Quick summary inside:"
echo "  ${PKG_NAME}/metadata/stage18_routeA_quick_summary.txt"
echo "Upload this .tar.gz to ChatGPT."
echo "============================================================"
