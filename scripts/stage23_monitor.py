#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd


ERROR_CLASSES = [
    ("dependency/import", ["modulenotfounderror", "importerror", "no module named"]),
    ("CUDA/JAX/XLA", ["cuda", "jax", "xla", "cudnn"]),
    ("MuJoCo/EGL", ["mujoco", "egl", "glfw", "osmesa"]),
    ("OGBench/D4RL env", ["ogbench", "d4rl", "no registered env", "environment"]),
    ("checkpoint/artifact path", ["checkpoint", "artifact", "keygraph", "params_", "pickle"]),
    ("GAS protocol mismatch", ["task_id", "goal", "reset", "horizon"]),
    ("policy adapter/skill mismatch", ["sample_action", "actor", "skill"]),
    ("planner/no path", ["planner", "no_path", "budget_infeasible", "disconnected"]),
    ("edge reset unsupported", ["set_state", "arbitrary", "reset unsupported"]),
    ("analysis/reporting bug", ["pandas", "csv", "to_markdown", "keyerror"]),
]


def classify_error(text: str) -> str:
    low = text.lower()
    for label, needles in ERROR_CLASSES:
        if any(n in low for n in needles):
            return label
    return "unknown"


def tail(path: Path, n: int = 80) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except Exception:
        return ""


def gpu_util() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return "nvidia-smi unavailable"


def scan_status(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    now = time.time()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("status.json"):
            try:
                row = json.loads(path.read_text())
            except Exception:
                row = {"status": "unknown"}
            row["status_path"] = str(path)
            rows.append(row)
        for failed in root.rglob("failed_jobs.csv"):
            try:
                for row in csv.DictReader(failed.open()):
                    row["status"] = "failed"
                    rows.append(row)
            except Exception:
                pass
        for log in list(root.rglob("*.log"))[-500:]:
            mtime = log.stat().st_mtime
            has_completed_eval = any(log.parent.rglob("eval.csv"))
            if now - mtime > 1800 and not has_completed_eval:
                rows.append({"status": "stalled_log", "log": str(log), "error_class": classify_error(tail(log, 60))})
    completed = {
        (str(r.get("env", "")), str(r.get("seed", "")), str(r.get("stage", "")))
        for r in rows
        if str(r.get("status")) == "completed"
    }
    for row in rows:
        key = (str(row.get("env", "")), str(row.get("seed", "")), str(row.get("stage", "")))
        if str(row.get("status")) == "failed" and key in completed:
            row["status"] = "recovered"
    return rows


def scan_eval(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("eval.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if not {"env", "seed", "success"}.issubset(df.columns):
                continue
            if len(df):
                df["eval_path"] = str(path)
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_reports(summary_md: Path, failed_csv: Path, gate_json: Path, statuses: list[dict[str, Any]], eval_df: pd.DataFrame) -> None:
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    reports = summary_md.parent
    cached_eval_df = pd.DataFrame()
    cached_eval_path = reports / "stage23_live_grouped.csv"
    if len(eval_df) == 0 and cached_eval_path.exists():
        try:
            cached_eval_df = pd.read_csv(cached_eval_path)
        except Exception:
            cached_eval_df = pd.DataFrame()
    counts: dict[str, int] = {}
    for row in statuses:
        counts[str(row.get("status", "unknown"))] = counts.get(str(row.get("status", "unknown")), 0) + 1
    failed = [s for s in statuses if str(s.get("status")) in {"failed", "stalled_log"}]
    if failed:
        pd.DataFrame(failed).to_csv(failed_csv, index=False)
    else:
        pd.DataFrame(columns=["status", "log", "error_class"]).to_csv(failed_csv, index=False)
    gates = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jobs": counts,
        "repro": "unknown",
        "bridge_existence": "unknown",
        "oracle": "unknown",
        "p_bridge": "unknown",
        "boundary": "unknown",
        "integrated": "unknown",
    }
    repro = reports / "stage23_gas_reproduction_matrix.csv"
    if repro.exists():
        df = pd.read_csv(repro)
        failed_repro = df[(df["status"] == "failed") | (df["status"] == "skipped")]
        gaps = []
        if {"env", "seed", "route", "success"}.issubset(df.columns):
            for _, sub in df.groupby(["env", "seed"], dropna=False):
                b = pd.to_numeric(sub[sub["route"].astype(str).eq("B_official_our_checkpoint")]["success"], errors="coerce").dropna()
                c = pd.to_numeric(sub[sub["route"].astype(str).eq("C_adapter_same_checkpoint")]["success"], errors="coerce").dropna()
                if len(b) and len(c):
                    gaps.append(abs(float(c.iloc[-1] - b.iloc[-1])))
        repair_ok = False
        repair = reports / "stage23_adapter_protocol_repair.csv"
        if repair.exists():
            rdf = pd.read_csv(repair)
            vals = pd.to_numeric(rdf.get("official_control_minus_official_pp", pd.Series(dtype=float)), errors="coerce").abs().dropna()
            repair_ok = len(vals) > 0 and float(vals.max()) <= 3.0
        hard_gap_ok = gaps and max(gaps) <= 0.03
        hard_no_skipped = len(failed_repro) == 0
        gates["repro"] = "GO_REPRO_REPAIRED" if repair_ok and len(df[df["status"] == "failed"]) == 0 else ("GO_REPRO" if hard_no_skipped and hard_gap_ok else "HOLD_REPRO")
    bridge = reports / "stage23_bridge_graph_summary.csv"
    if bridge.exists():
        df = pd.read_csv(bridge)
        headroom = df[(df.get("graph_id", "") != "G0") & ((df.get("shorter_path_rate", 0) > 0) | (df.get("bridge_usage_rate", 0) > 0))]
        gates["bridge_existence"] = "PASS_BRIDGE_EXISTENCE" if len(headroom) else "NO_BRIDGE_HEADROOM"
    oracle = reports / "stage23_oracle_bridge_summary.csv"
    if oracle.exists():
        df = pd.read_csv(oracle)
        oracle_rows = df[df.get("graph_id", "").astype(str).eq("G_oracle")]
        reduction = pd.to_numeric(oracle_rows.get("mean_path_cost_reduction", pd.Series(dtype=float)), errors="coerce").fillna(0)
        shorter = pd.to_numeric(oracle_rows.get("shorter_path_rate", pd.Series(dtype=float)), errors="coerce").fillna(0)
        usage = pd.to_numeric(oracle_rows.get("bridge_usage_rate", pd.Series(dtype=float)), errors="coerce").fillna(0)
        ok = (len(reduction) and float(reduction.max()) >= 0.5) or (len(shorter) and float(shorter.max()) >= 0.2 and len(usage) and float(usage.max()) >= 0.2)
        gates["oracle"] = "PASS_ORACLE" if ok else "NO_ORACLE_UPPER_BOUND"
    pbridge = reports / "stage23_p_bridge_metrics.csv"
    if pbridge.exists():
        df = pd.read_csv(pbridge)
        val = pd.to_numeric(df.get("selected_bridge_AUROC", pd.Series(dtype=float)), errors="coerce")
        fp = pd.to_numeric(df.get("false_positive_bridge_relative_reduction@0.6", pd.Series(dtype=float)), errors="coerce")
        if len(val.dropna()) and val.max() >= 0.65 and len(fp.dropna()) and fp.max() >= 0.20:
            gates["p_bridge"] = "PASS_P_BRIDGE"
        elif len(val.dropna()) and val.max() >= 0.65:
            gates["p_bridge"] = "PARTIAL_P_BRIDGE_HOLD_FP_REDUCTION"
        else:
            gates["p_bridge"] = "HOLD_P_BRIDGE"
    boundary = reports / "stage23_boundary_junction_metrics.csv"
    if boundary.exists():
        df = pd.read_csv(boundary)
        au = pd.to_numeric(df.get("psi_AUROC_for_conditional_success", pd.Series(dtype=float)), errors="coerce")
        gap = pd.to_numeric(df.get("supported_gap", pd.Series(dtype=float)), errors="coerce")
        coverage = pd.to_numeric(df.get("coverage", pd.Series(dtype=float)), errors="coerce")
        gates["boundary"] = "PASS_BOUNDARY" if len(au.dropna()) and au.max() >= 0.60 and len(gap.dropna()) and gap.max() >= 0.10 and len(coverage.dropna()) and coverage.max() > 0.01 else "HOLD_BOUNDARY"
    eval_for_summary = eval_df
    if len(eval_for_summary) == 0 and len(cached_eval_df):
        eval_for_summary = cached_eval_df
    if len(eval_df):
        grouped = eval_df.groupby([c for c in ["env", "seed", "variant"] if c in eval_df], dropna=False).agg(episodes=("success", "count"), success=("success", "mean")).reset_index()
        grouped.to_csv(reports / "stage23_live_grouped.csv", index=False)
    gate_json.write_text(json.dumps(gates, indent=2, sort_keys=True))
    lines = ["# Stage23 Live Summary", "", f"Updated: {gates['updated']}", "", "## Jobs"]
    for k in sorted(counts):
        lines.append(f"- {k}: {counts[k]}")
    lines.extend(["", "## Gates"])
    for k, v in gates.items():
        if k != "updated" and k != "jobs":
            lines.append(f"- {k}: {v}")
    lines.extend(["", "## Eval"])
    if len(eval_for_summary):
        if len(eval_df) == 0:
            lines.append("Cached summary from `reports/stage23_live_grouped.csv`; raw run directories are not present in this checkout.")
        group_cols = [c for c in ["env", "seed", "variant", "budget", "fallback_mode"] if c in eval_for_summary]
        agg = {"success": ("success", "mean")}
        if "episodes" in eval_for_summary:
            # Already grouped report: keep its episode counts instead of
            # treating each summary row as one episode.
            grouped = eval_for_summary.copy()
        else:
            agg["episodes"] = ("success", "count")
            if "steps" in eval_for_summary:
                agg["steps"] = ("steps", "mean")
            grouped = eval_for_summary.groupby(group_cols, dropna=False).agg(**agg).reset_index()
        try:
            lines.append(grouped.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + grouped.to_csv(index=False).strip() + "\n```")
    else:
        lines.append("No eval rows yet.")
    lines.extend(["", "## GPU", "```", gpu_util(), "```"])
    summary_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", default="runs_stage23_repro,runs_stage23_atlas,runs_stage23_bridge,runs_stage23_integrated")
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--summary-md", default="reports/stage23_live_summary.md")
    p.add_argument("--failed-csv", default="reports/stage23_failed_jobs.csv")
    p.add_argument("--gate-json", default="reports/stage23_gate_status.json")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()
    roots = [Path(x) for x in args.roots.split(",") if x]
    while True:
        statuses = scan_status(roots)
        eval_df = scan_eval(roots)
        write_reports(Path(args.summary_md), Path(args.failed_csv), Path(args.gate_json), statuses, eval_df)
        running = [s for s in statuses if s.get("status") == "running"]
        print(json.dumps({"running": len(running), "failed": sum(1 for s in statuses if s.get("status") == "failed"), "eval_rows": int(len(eval_df))}, indent=2))
        if not args.wait or not running:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
