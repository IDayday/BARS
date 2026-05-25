#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "runs_round006_gas_dynamic"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def log_tail(path: Path, limit: int = 30000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-limit:]


def tensorboard_file_failure_logs(run_dir: Path) -> list[str]:
    matches: list[str] = []
    for name in ("pretrain_tdr.log", "train_policy.log", "evaluate_gas.log", "worker_supervisor.log"):
        path = run_dir / name
        tail = log_tail(path)
        if "FileNotFoundError" in tail and "events.out.tfevents" in tail:
            matches.append(rel(path))
    return matches


def checkpoint_summary(run_dir: Path) -> dict:
    try:
        env = run_dir.parent.name
        seed_dir = run_dir.name
        out_root = REPO_ROOT / "artifacts" / "gas_selftrain_round006" / env / seed_dir
        checkpoints = sorted(out_root.glob("**/params_*.pkl"))
        latest = None
        latest_step = -1
        for path in checkpoints:
            try:
                step = int(path.stem.split("_")[-1])
            except Exception:
                step = -1
            if step > latest_step:
                latest_step = step
                latest = path
        return {
            "checkpoint_count": len(checkpoints),
            "latest_checkpoint": rel(latest) if latest is not None else None,
            "latest_checkpoint_step": latest_step if latest_step >= 0 else None,
        }
    except Exception as exc:
        return {"checkpoint_summary_error": repr(exc)}


def requeue(args: argparse.Namespace) -> list[dict]:
    run_root = Path(args.run_root)
    rows: list[dict] = []
    for status_path in sorted(run_root.glob("*/seed*/status.json")):
        status = read_json(status_path)
        if status.get("status") != "failed":
            continue
        run_dir = status_path.parent
        evidence_logs = tensorboard_file_failure_logs(run_dir)
        if not evidence_logs:
            continue
        count = int(status.get("logging_failure_requeue_count") or 0)
        row = {
            "env": status.get("env") or run_dir.parent.name,
            "seed": status.get("seed") or run_dir.name.removeprefix("seed"),
            "status_file": rel(status_path),
            "previous_failed_at": status.get("failed_at"),
            "requeue_count": count,
            "evidence_logs": ", ".join(evidence_logs),
            **checkpoint_summary(run_dir),
        }
        if count >= args.max_requeues:
            row["action"] = "skipped_max_requeues"
            rows.append(row)
            continue
        row["action"] = "dry_run" if args.dry_run else "requeued"
        rows.append(row)
        if args.dry_run:
            continue
        history = list(status.get("failure_history") or [])
        history.append(
            {
                "status": "failed",
                "failed_at": status.get("failed_at"),
                "error": status.get("error"),
                "evidence_logs": evidence_logs,
                "requeued_at": now_iso(),
                "requeue_reason": "tensorboard_event_file_missing_non_algorithmic",
            }
        )
        status.update(
            {
                "status": "retry_pending",
                "last_failed_at": status.get("failed_at"),
                "last_error": status.get("error"),
                "failed_at": None,
                "error": None,
                "failure_history": history,
                "logging_failure_requeue_count": count + 1,
                "requeue_reason": "tensorboard_event_file_missing_non_algorithmic",
                "requeued_at": now_iso(),
            }
        )
        write_json(status_path, status)
    return rows


def write_report(path: Path, rows: list[dict], dry_run: bool) -> None:
    lines = [
        "# Round 006 Logging Failure Requeue",
        "",
        f"Generated: {now_iso()}.",
        f"Dry run: `{dry_run}`.",
        "",
        "Only failures with both `FileNotFoundError` and `events.out.tfevents` in the phase logs are eligible.",
        "These are logging-path failures, not algorithm-result evidence.",
        "",
        "| env | seed | action | requeues_before | latest_ckpt_step | latest_ckpt | status_file |",
        "| --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    if not rows:
        lines.append("| none |  |  |  |  |  |  |")
    for row in rows:
        lines.append(
            "| {env} | {seed} | {action} | {requeue_count} | {latest_checkpoint_step} | {latest_checkpoint} | {status_file} |".format(
                env=row.get("env"),
                seed=row.get("seed"),
                action=row.get("action"),
                requeue_count=row.get("requeue_count"),
                latest_checkpoint_step=row.get("latest_checkpoint_step"),
                latest_checkpoint=row.get("latest_checkpoint"),
                status_file=row.get("status_file"),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--max-requeues", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=str(REPO_ROOT / "reports" / "round_006_logging_failure_requeue.md"))
    args = parser.parse_args()
    rows = requeue(args)
    write_report(Path(args.report), rows, args.dry_run)
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
