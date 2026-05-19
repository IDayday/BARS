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


def classify_error(text: str) -> str:
    low = text.lower()
    if "modulenotfounderror" in low or "importerror" in low:
        return "dependency/import"
    if "cuda" in low or "jax" in low or "mujoco" in low or "egl" in low:
        return "CUDA/JAX/MuJoCo"
    if "checkpoint" in low or "artifact" in low or "keygraph" in low or "params_" in low:
        return "checkpoint/artifact missing"
    if "goal" in low or "task_id" in low or "reset" in low:
        return "goal protocol"
    if "budget_infeasible" in low or "no_start_connection" in low or "planner" in low:
        return "planner infeasible"
    if "boundary" in low:
        return "boundary feasibility"
    if "p_exec" in low or "reachability" in low:
        return "p_exec scoring"
    if "fallback" in low:
        return "fallback protocol"
    return "unknown"


def tail(path: Path, n: int = 100) -> str:
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
    for root in roots:
        for path in root.rglob("status.json"):
            try:
                row = json.loads(path.read_text())
            except Exception:
                row = {"status": "unknown"}
            row["status_path"] = str(path)
            rows.append(row)
        failed_csv = root / "failed_jobs.csv"
        if failed_csv.exists():
            try:
                with failed_csv.open() as f:
                    for row in csv.DictReader(f):
                        row["status"] = "failed"
                        rows.append(row)
            except Exception:
                pass
    return rows


def scan_eval(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        for path in root.rglob("eval.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if len(df):
                df["eval_path"] = str(path)
                if "fallback_mode" not in df:
                    df["fallback_mode"] = path.parent.name.replace("fallback_", "")
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_summary(path: Path, statuses: list[dict[str, Any]], df: pd.DataFrame) -> None:
    counts = {}
    for row in statuses:
        counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1
    lines = ["# Stage23 Live Summary", "", f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "", "## Jobs"]
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")
    lines.append("")
    lines.append("## Eval")
    if len(df):
        keys = [k for k in ["env", "seed", "variant", "budget", "fallback_mode"] if k in df.columns]
        grouped = df.groupby(keys, dropna=False).agg(episodes=("success", "count"), success=("success", "mean")).reset_index() if keys else pd.DataFrame()
        grouped.to_csv(path.with_name("stage23_live_grouped.csv"), index=False)
        try:
            lines.append(grouped.to_markdown(index=False))
        except ImportError:
            lines.append("```csv\n" + grouped.to_csv(index=False).strip() + "\n```")
        if "budget_reject_count" in df:
            br = df.groupby(keys, dropna=False).agg(budget_reject_rate=("budget_reject_count", lambda x: float((x > 0).mean()))).reset_index()
            if len(br[br["budget_reject_rate"] > 0.5]):
                lines.append("")
                lines.append("- HOLD_BOUNDARY signal: at least one setting has budget_reject_rate > 50%.")
        if "fallback_used" in df and float(df["fallback_used"].mean()) > 0.7:
            lines.append("- FALLBACK_DOMINATED signal: fallback_used > 70%.")
    else:
        lines.append("No eval rows yet.")
    lines.append("")
    lines.append("## GPU")
    lines.append("```")
    lines.append(gpu_util())
    lines.append("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", default="runs_stage23_key_claim,runs_stage23_boundary,runs_stage23_d4rl")
    p.add_argument("--summary-md", default="reports/stage23_live_summary.md")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()
    roots = [Path(x) for x in args.roots.split(",") if x]
    while True:
        statuses = scan_status(roots)
        df = scan_eval(roots)
        write_summary(Path(args.summary_md), statuses, df)
        running = [s for s in statuses if s.get("status") == "running"]
        print(json.dumps({"running": len(running), "completed": sum(1 for s in statuses if s.get("status") == "completed"), "failed": sum(1 for s in statuses if s.get("status") == "failed"), "eval_rows": int(len(df))}, indent=2))
        if not args.wait or not running:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
