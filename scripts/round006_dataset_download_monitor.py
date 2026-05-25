#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def tail_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-n:]


def render_snapshot(args: argparse.Namespace) -> tuple[str, dict]:
    table = Path(args.table)
    download_log = Path(args.download_log)
    rows = read_rows(table)
    ready = [row for row in rows if str(row.get("ready", "")).lower() == "true"]
    pending = [row for row in rows if str(row.get("ready", "")).lower() != "true"]
    total_bytes = 0
    for row in rows:
        for key in ("train_size", "val_size"):
            try:
                total_bytes += int(row.get(key) or 0)
            except ValueError:
                pass
    failed_lines = [line for line in tail_lines(download_log, 200) if " FAIL " in line]
    log_tail = tail_lines(download_log, args.tail_lines)
    payload = {
        "generated": now_iso(),
        "ready": len(ready),
        "total": len(rows),
        "pending": [row.get("env", "") for row in pending],
        "total_size_bytes": total_bytes,
        "failed_tail_count": len(failed_lines),
    }
    lines = [
        "# Round 006 Dataset Download Monitor",
        "",
        f"- Generated: {payload['generated']}",
        f"- Ready: `{payload['ready']}/{payload['total']}`",
        f"- Total local size: `{human_size(total_bytes)}`",
        f"- Failures in recent log tail: `{len(failed_lines)}`",
        f"- Table: `{table}`",
        f"- Download log: `{download_log}`",
        "",
        "## Pending",
        "",
    ]
    if pending:
        for row in pending:
            lines.append(f"- {row.get('env', '')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Recent Download Log", "", "```text"])
    lines.extend(log_tail or ["<no download log yet>"])
    lines.append("```")
    return "\n".join(lines) + "\n", payload


def emit(args: argparse.Namespace) -> None:
    snapshot, payload = render_snapshot(args)
    latest = Path(args.latest_report)
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(snapshot, encoding="utf-8")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(snapshot)
        f.write("\n")
    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    print(snapshot, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(REPO_ROOT / "reports" / "round_006_ogbench_download_status.tsv"))
    parser.add_argument("--download-log", default=str(REPO_ROOT / "runs_round006_gas_dynamic" / "_orchestrator" / "download.log"))
    parser.add_argument("--latest-report", default=str(REPO_ROOT / "reports" / "round_006_dataset_download_monitor_latest.md"))
    parser.add_argument("--log", default=str(REPO_ROOT / "runs_round006_gas_dynamic" / "_monitor" / "dataset_download_monitor.log"))
    parser.add_argument("--jsonl", default=str(REPO_ROOT / "runs_round006_gas_dynamic" / "_monitor" / "dataset_download_monitor.jsonl"))
    parser.add_argument("--tail-lines", type=int, default=14)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args()
    while True:
        emit(args)
        if not args.watch:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
