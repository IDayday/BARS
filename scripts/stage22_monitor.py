#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd


def classify_error(text: str) -> str:
    low = text.lower()
    if "modulenotfounderror" in low or "importerror" in low or "no module named" in low:
        return "dependency/import"
    if "cuda" in low or "jaxlib" in low or "xla" in low or "cudnn" in low:
        return "CUDA/JAX"
    if "mujoco" in low or "egl" in low or "glfw" in low:
        return "MuJoCo/EGL"
    if "ogbench" in low or "d4rl" in low or "dataset" in low or "no registered env" in low:
        return "env/dataset"
    if "keygraph" in low or "params_" in low or "checkpoint" in low or "artifact" in low:
        return "checkpoint/artifact"
    if "planner" in low or "budget_infeasible" in low:
        return "planner"
    if "actor" in low or "sample_actions" in low or "skill" in low:
        return "policy adapter"
    return "unknown"


def tail(path: Path, n: int = 100) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def gpu_util() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return "nvidia-smi unavailable"


def scan_status(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("status.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {"status": "unknown"}
        data["status_path"] = str(path)
        data["root"] = str(root)
        rows.append(data)
    return rows


def scan_eval(eval_root: Path) -> pd.DataFrame:
    frames = []
    for path in eval_root.rglob("eval.csv"):
        try:
            df = pd.read_csv(path)
            if len(df):
                df["eval_path"] = str(path)
                frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def latest_train_steps(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        for path in root.rglob("*.log"):
            text = tail(path, 40)
            step = ""
            for line in reversed(text.splitlines()):
                if "Training TDR" in line or "Training Policy" in line or "step" in line:
                    step = line[-180:]
                    break
            rows.append({"log": str(path), "mtime": path.stat().st_mtime, "latest_step": step})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:20]


def write_summary_md(path: Path, statuses: list[dict[str, Any]], eval_df: pd.DataFrame, failed: list[dict[str, Any]], stalled: list[Path]) -> None:
    counts: dict[str, int] = {}
    for s in statuses:
        counts[s.get("status", "unknown")] = counts.get(s.get("status", "unknown"), 0) + 1
    lines = ["# Stage22 Live Summary", ""]
    lines.append(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Jobs")
    for k in sorted(counts):
        lines.append(f"- {k}: {counts[k]}")
    if failed:
        lines.append(f"- failed classified: {len(failed)}")
    if stalled:
        lines.append(f"- stalled logs: {len(stalled)}")
    lines.append("")
    lines.append("## Eval")
    if len(eval_df):
        keys = [k for k in ["env", "seed", "variant", "budget", "fallback_mode"] if k in eval_df.columns]
        if keys:
            grouped = eval_df.groupby(keys, dropna=False).agg(episodes=("success", "count"), success=("success", "mean")).reset_index()
            try:
                lines.append(grouped.to_markdown(index=False))
            except ImportError:
                lines.append("```csv")
                lines.append(grouped.to_csv(index=False).strip())
                lines.append("```")
        else:
            lines.append(f"Rows: {len(eval_df)}")
    else:
        lines.append("No eval rows yet.")
    if failed:
        lines.append("")
        lines.append("## Failures")
        for f in failed[:20]:
            lines.append(f"- {f.get('error_class', 'unknown')}: {f.get('log', f.get('status_path', ''))}")
    if stalled:
        lines.append("")
        lines.append("## Stalled")
        for p in stalled[:20]:
            lines.append(f"- {p}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def monitor_once(roots: list[Path], summary_md: Path | None, kill_stalled: bool) -> dict[str, Any]:
    statuses: list[dict[str, Any]] = []
    for root in roots:
        statuses.extend(scan_status(root))
    failed = [s for s in statuses if s.get("status") == "failed"]
    for f in failed:
        log = f.get("log")
        if not log:
            sp = Path(f["status_path"])
            logs = list(sp.parent.glob("*.log"))
            log = str(logs[0]) if logs else ""
        f["log"] = log
        if log and Path(log).exists() and "error_class" not in f:
            f["error_class"] = classify_error(tail(Path(log), 200))
    logs = []
    for root in roots:
        logs.extend(root.rglob("*.log"))
    now = time.time()
    stalled = [p for p in logs if now - p.stat().st_mtime > 30 * 60 and "running" in tail(p.parent / "status.json", 10)]
    if kill_stalled:
        print("--kill-stalled requested, but automatic killing is intentionally conservative; no processes were killed.")
    eval_roots = [r for r in roots if "eval" in r.name]
    eval_df = pd.concat([scan_eval(r) for r in eval_roots], ignore_index=True) if eval_roots else pd.DataFrame()
    for root in roots:
        failed_csv = root / "failed_jobs.csv"
        if failed_csv.exists():
            try:
                with open(failed_csv) as f:
                    for row in csv.DictReader(f):
                        failed.append(dict(row))
            except Exception:
                pass
    if summary_md:
        write_summary_md(summary_md, statuses, eval_df, failed, stalled)
    print(json.dumps({
        "job_counts": {k: sum(1 for s in statuses if s.get("status") == k) for k in sorted(set(s.get("status", "unknown") for s in statuses))},
        "eval_rows": int(len(eval_df)),
        "eval_success": float(eval_df["success"].mean()) if len(eval_df) and "success" in eval_df else None,
        "failed": len(failed),
        "stalled": len(stalled),
        "gpu": gpu_util(),
    }, indent=2))
    if stalled:
        for p in stalled[:5]:
            print(f"\n--- stalled tail: {p} ---\n{tail(p, 100)}")
    return {"statuses": statuses, "failed": failed, "stalled": stalled, "eval_df": eval_df}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", default="runs_stage22_prepare,runs_stage22_eval")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--summary-md", default="")
    p.add_argument("--kill-stalled", action="store_true")
    args = p.parse_args()
    roots = [Path(x) for x in args.roots.split(",") if x]
    summary_md = Path(args.summary_md) if args.summary_md else None
    while True:
        result = monitor_once(roots, summary_md, args.kill_stalled)
        running = [s for s in result["statuses"] if s.get("status") == "running"]
        if not args.wait or not running:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
