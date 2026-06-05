#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


STATE_NAMES = [
    "NORMAL",
    "COMMITTING",
    "LOCAL_STALL",
    "PATH_DRIFT",
    "RECOVERY",
    "REPLAN_MISS",
    "FINAL_GOAL",
    "FINAL_GOAL_STALL",
    "CHURN_GUARD",
    "FALLBACK_TO_GAS",
    "RECOVERY_LOCKOUT",
]

SUMMARY_METRICS = [
    "success_rate",
    "normalized_score",
    "return",
    "target_switch_count",
    "early_switch_count",
    "stall_count",
    "drift_count",
    "recovery_attempt_count",
    "recovery_success_count",
    "recovery_success_rate",
    "global_replan_request_count",
    "global_replan_request_rate_per_100_steps",
    "max_consecutive_replan_burst",
    "segment_target_reach_rate",
    "mean_segment_progress",
    "mean_distance_to_path",
    "final_goal_switch_count",
    "final_goal_stall_count",
    "timeout_rate",
    "no_path_rate",
    "replan_churn_episode_rate",
    "zero_segment_reach_episode_rate",
    "failed_recovery_churn_episode_rate",
    "unstable_execution_episode_rate",
    "churn_guard_trigger_count",
    "fallback_to_gas_step_count",
    "replan_suppressed_by_cooldown_count",
    "replan_suppressed_by_budget_count",
    "recovery_suppressed_by_lockout_count",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze CAGE trace churn and failure modes.")
    parser.add_argument("--input_roots", nargs="+", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    return parser


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(root: Path) -> list[dict[str, Any]]:
    rows_by_job: dict[str, dict[str, Any]] = {}
    rows_without_job: list[dict[str, Any]] = []
    for path in sorted((root / "manifests").glob("*.jsonl")):
        for row in load_jsonl(path):
            job_id = row.get("job_id")
            if job_id:
                rows_by_job[str(job_id)] = row
            else:
                rows_without_job.append(row)
    return [*rows_by_job.values(), *rows_without_job]


def load_status(root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "status").glob("*.jsonl")):
        for row in load_jsonl(path):
            job_id = row.get("job_id")
            if job_id:
                latest[str(job_id)] = row
    return latest


def load_summary_jobs(root: Path) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "tables").glob("*summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in data.get("jobs", []):
            job_id = row.get("job_id")
            if job_id:
                jobs[str(job_id)] = row
    return jobs


def load_csv_metrics(path_value: str | None) -> dict[str, float]:
    if not path_value or not Path(path_value).exists():
        return {}
    with Path(path_value).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    row = rows[-1]
    metrics: dict[str, float] = {}
    for name, suffix in [
        ("success_rate", "episode.success"),
        ("normalized_score", "episode.normalized_return"),
        ("return", "episode.return"),
    ]:
        vals = [numeric(v) for k, v in row.items() if k.endswith(suffix)]
        vals = [v for v in vals if v is not None]
        if vals:
            metrics[name] = mean(vals)
    return metrics


def split_trace(path_value: str | None) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    rows = load_jsonl(Path(path_value)) if path_value else []
    episodes: list[dict[str, Any]] = []
    steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        record_type = row.get("record_type", "episode")
        if record_type == "step":
            idx = int(row.get("episode_idx", 0) or 0)
            steps[idx].append(row)
        elif record_type == "episode":
            episodes.append(row)
    return episodes, steps


def max_replan_burst(step_rows: list[dict[str, Any]]) -> int | None:
    if not step_rows:
        return None
    best = 0
    cur = 0
    for row in sorted(step_rows, key=lambda item: int(item.get("step", 0) or 0)):
        if bool(row.get("should_replan")):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def state_occupancy(step_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not step_rows:
        return {f"state_occupancy_{name}": None for name in STATE_NAMES}
    counts = Counter(str(row.get("cage_state")) for row in step_rows)
    total = sum(counts.values()) or 1
    return {f"state_occupancy_{name}": counts.get(name, 0) / total for name in STATE_NAMES}


def episode_record(
    root: Path,
    row: dict[str, Any],
    status: dict[str, Any] | None,
    episode: dict[str, Any],
    step_rows: list[dict[str, Any]],
    csv_metrics: dict[str, float],
) -> dict[str, Any]:
    steps = [numeric(step.get("step")) for step in step_rows]
    episode_steps = int(max(steps) + 1) if steps and max(steps) is not None else None
    replan_count = numeric(episode.get("global_replan_request_count")) or 0.0
    burst = numeric(episode.get("max_consecutive_replan_burst"))
    if burst is None:
        burst_val = max_replan_burst(step_rows)
        burst = float(burst_val) if burst_val is not None else None
    rate = numeric(episode.get("global_replan_request_rate_per_100_steps"))
    if rate is None and episode_steps:
        rate = 100.0 * replan_count / max(1, episode_steps)
    recovery_attempts = numeric(episode.get("recovery_attempt_count")) or 0.0
    recovery_successes = numeric(episode.get("recovery_success_count")) or 0.0
    segment_rate = numeric(episode.get("segment_target_reach_rate"))
    drift_count = numeric(episode.get("drift_count")) or 0.0
    target_switch_count = numeric(episode.get("target_switch_count")) or 0.0

    out = {
        "input_root": str(root),
        "job_id": row.get("job_id"),
        "env_name": row.get("env_name"),
        "variant": row.get("variant"),
        "seed": row.get("seed"),
        "episode_idx": episode.get("episode_idx"),
        "run_status": status.get("status") if status else None,
        "success_rate": numeric(episode.get("success", csv_metrics.get("success_rate"))),
        "normalized_score": numeric(episode.get("normalized_score", csv_metrics.get("normalized_score"))),
        "return": numeric(episode.get("return", csv_metrics.get("return"))),
        "target_switch_count": numeric(episode.get("target_switch_count")),
        "early_switch_count": numeric(episode.get("early_switch_count")),
        "stall_count": numeric(episode.get("stall_count")),
        "drift_count": numeric(episode.get("drift_count")),
        "recovery_attempt_count": recovery_attempts,
        "recovery_success_count": recovery_successes,
        "recovery_success_rate": recovery_successes / recovery_attempts if recovery_attempts else None,
        "global_replan_request_count": replan_count,
        "global_replan_request_rate_per_100_steps": rate,
        "max_consecutive_replan_burst": burst,
        "segment_target_reach_rate": segment_rate,
        "mean_segment_progress": numeric(episode.get("mean_segment_progress")),
        "mean_distance_to_path": numeric(episode.get("mean_distance_to_path")),
        "final_goal_on_step": numeric(episode.get("final_goal_on_step")),
        "final_goal_switch_count": numeric(episode.get("final_goal_switch_count")),
        "final_goal_stall_count": numeric(episode.get("final_goal_stall_count")),
        "timeout_rate": numeric(episode.get("timeout")),
        "no_path_rate": numeric(episode.get("no_path")),
        "churn_guard_trigger_count": numeric(episode.get("churn_guard_trigger_count")),
        "fallback_to_gas_step_count": numeric(episode.get("fallback_to_gas_step_count")),
        "replan_suppressed_by_cooldown_count": numeric(episode.get("replan_suppressed_by_cooldown_count")),
        "replan_suppressed_by_budget_count": numeric(episode.get("replan_suppressed_by_budget_count")),
        "recovery_suppressed_by_lockout_count": numeric(episode.get("recovery_suppressed_by_lockout_count")),
    }
    out.update(state_occupancy(step_rows))
    out["replan_churn_episode"] = bool(replan_count > 100 or (rate is not None and rate > 20))
    out["zero_segment_reach_episode"] = bool(segment_rate == 0)
    out["failed_recovery_churn_episode"] = bool(
        recovery_attempts > 0 and recovery_successes == 0 and replan_count > 20
    )
    out["unstable_execution_episode"] = bool(
        out["replan_churn_episode"]
        or out["zero_segment_reach_episode"]
        or out["failed_recovery_churn_episode"]
        or drift_count > 50
        or target_switch_count > 100
    )
    return out


def pseudo_episode_from_summary(row: dict[str, Any], csv_metrics: dict[str, float]) -> dict[str, Any]:
    episode = dict(row)
    if "success" not in episode and row.get("success_rate") is not None:
        episode["success"] = row.get("success_rate")
    if "return" not in episode and csv_metrics.get("return") is not None:
        episode["return"] = csv_metrics.get("return")
    return episode


def collect_episodes(input_roots: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root_value in input_roots:
        root = Path(root_value)
        manifest = load_manifest(root)
        status = load_status(root)
        summary_jobs = load_summary_jobs(root)
        for row in manifest:
            job_id = str(row.get("job_id"))
            row_status = status.get(job_id)
            csv_metrics = load_csv_metrics(row.get("result_path"))
            episodes, steps_by_episode = split_trace(row.get("cage_trace_path"))
            if not episodes:
                fallback = summary_jobs.get(job_id)
                if fallback:
                    episodes = [pseudo_episode_from_summary(fallback, csv_metrics)]
                elif csv_metrics:
                    episodes = [pseudo_episode_from_summary({}, csv_metrics)]
            for idx, episode in enumerate(episodes):
                ep_idx = int(episode.get("episode_idx", idx) or 0)
                records.append(episode_record(root, row, row_status, episode, steps_by_episode.get(ep_idx, []), csv_metrics))
    return records


def mean_std_max(values: list[Any]) -> tuple[float | None, float | None, float | None]:
    clean = [numeric(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return None, None, None
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0, max(clean)


def grouped(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row.get("env_name")), str(row.get("variant")))].append(row)
    out: list[dict[str, Any]] = []
    for (env_name, variant), rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "env_name": env_name,
            "variant": variant,
            "num_records": len(rows),
        }
        for metric in SUMMARY_METRICS:
            avg, sd, mx = mean_std_max([row.get(metric) for row in rows])
            record[f"{metric}_mean"] = avg
            record[f"{metric}_std"] = sd
            record[f"{metric}_max"] = mx
        for metric in [
            "replan_churn_episode",
            "zero_segment_reach_episode",
            "failed_recovery_churn_episode",
            "unstable_execution_episode",
        ]:
            avg, sd, mx = mean_std_max([row.get(metric) for row in rows])
            record[f"{metric}_rate"] = avg
            record[f"{metric}_max"] = mx
        for state in STATE_NAMES:
            avg, _, _ = mean_std_max([row.get(f"state_occupancy_{state}") for row in rows])
            record[f"state_occupancy_{state}_mean"] = avg
        out.append(record)
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(out_json: Path, out_md: Path, records: list[dict[str, Any]], grouped_rows: list[dict[str, Any]]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"episodes": records, "grouped": grouped_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fields = [
        "env_name",
        "variant",
        "num_records",
        "success_rate_mean",
        "normalized_score_mean",
        "target_switch_count_mean",
        "stall_count_mean",
        "drift_count_mean",
        "recovery_attempt_count_mean",
        "recovery_success_rate_mean",
        "global_replan_request_count_mean",
        "global_replan_request_rate_per_100_steps_mean",
        "max_consecutive_replan_burst_mean",
        "segment_target_reach_rate_mean",
        "replan_churn_episode_rate",
        "zero_segment_reach_episode_rate",
        "failed_recovery_churn_episode_rate",
        "unstable_execution_episode_rate",
        "churn_guard_trigger_count_mean",
        "fallback_to_gas_step_count_mean",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "# CAGE Churn Analysis\n\n"
        "## Grouped Metrics\n\n"
        + markdown_table(grouped_rows, fields)
        + "\n## Notes\n\n"
        "- `*_episode_rate` fields are means of per-episode boolean churn flags.\n"
        "- State occupancy is included in the JSON output when step-level traces are available.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    records = collect_episodes(args.input_roots)
    grouped_rows = grouped(records)
    write_outputs(Path(args.out_json), Path(args.out_md), records, grouped_rows)
    print(json.dumps({"episodes": len(records), "groups": len(grouped_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
