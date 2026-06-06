#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PRIMARY_JOB_METRICS = [
    "success_rate",
    "return",
    "normalized_score",
    "no_path_rate",
    "target_switch_count",
    "early_switch_count",
    "mean_commitment_length",
    "stall_count",
    "drift_count",
    "recovery_attempt_count",
    "recovery_success_rate",
    "global_replan_request_count",
    "global_replan_request_rate_per_100_steps",
    "max_consecutive_replan_burst",
    "segment_target_reach_rate",
    "mean_segment_progress",
    "mean_distance_to_path",
    "final_goal_on_rate",
    "final_goal_switch_count",
    "final_goal_stall_count",
    "timeout_rate",
    "path_changed_from_gas_rate",
    "path_min_reachability",
    "churn_guard_trigger_count",
    "fallback_to_gas_count",
    "fallback_to_gas_step_count",
    "replan_suppressed_by_cooldown_count",
    "replan_suppressed_by_budget_count",
    "recovery_suppressed_by_lockout_count",
    "churn_guard_active_on_timeout",
    "cage_safe_mode_enabled",
    "cage_trace_only",
    "cage_contract_commit",
    "contract_model_loaded",
    "contract_gate_pass_count",
    "contract_gate_reject_count",
    "contract_gate_reject_rate",
    "contract_fallback_to_gas_when_uncertain_count",
    "contract_recovery_reject_count",
    "contract_final_goal_reject_count",
]

PAIR_METRICS = [
    "success_rate",
    "normalized_score",
    "target_switch_count",
    "stall_count",
    "drift_count",
    "recovery_success_rate",
    "segment_target_reach_rate",
    "final_goal_stall_count",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate focused GAS/CAGE experiment outputs.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
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


def load_status_rows(input_root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted((input_root / "status").glob("*.jsonl")):
        for row in load_jsonl(path):
            job_id = row.get("job_id")
            if job_id:
                latest[str(job_id)] = row
    return latest


def load_episode_rows(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    rows = load_jsonl(Path(path_value))
    return [row for row in rows if row.get("record_type", "episode") == "episode"]


def load_csv_metrics(path_value: str | None) -> dict[str, float]:
    if not path_value or not Path(path_value).exists():
        return {}
    with Path(path_value).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return {}
    row = rows[-1]
    metrics: dict[str, float] = {}
    success_values = [numeric(v) for k, v in row.items() if k.endswith("episode.success")]
    success_values = [v for v in success_values if v is not None]
    normalized_values = [numeric(v) for k, v in row.items() if k.endswith("episode.normalized_return")]
    normalized_values = [v for v in normalized_values if v is not None]
    return_values = [numeric(v) for k, v in row.items() if k.endswith("episode.return")]
    return_values = [v for v in return_values if v is not None]
    if success_values:
        metrics["success_rate"] = mean(success_values)
    if normalized_values:
        metrics["normalized_score"] = mean(normalized_values)
    if return_values:
        metrics["return"] = mean(return_values)
    return metrics


def mean_values(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [numeric(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def summarize_job(row: dict[str, Any], status: dict[str, Any] | None) -> dict[str, Any]:
    episodes = load_episode_rows(row.get("cage_trace_path"))
    csv_metrics = load_csv_metrics(row.get("result_path"))
    summary: dict[str, Any] = {
        "job_id": row.get("job_id"),
        "env_name": row.get("env_name"),
        "seed": row.get("seed"),
        "variant": row.get("variant"),
        "manifest_status": row.get("status"),
        "run_status": status.get("status") if status else None,
        "num_episodes": len(episodes),
    }
    summary["succeeded"] = bool(status and status.get("status") == "succeeded")
    summary["success_rate"] = mean_values(episodes, "success")
    if summary["success_rate"] is None:
        summary["success_rate"] = csv_metrics.get("success_rate")
    summary["normalized_score"] = mean_values(episodes, "normalized_score")
    if summary["normalized_score"] is None:
        summary["normalized_score"] = csv_metrics.get("normalized_score")
    summary["return"] = mean_values(episodes, "return")
    if summary["return"] is None:
        summary["return"] = csv_metrics.get("return")
    summary["no_path_rate"] = mean_values(episodes, "no_path")
    summary["timeout_rate"] = mean_values(episodes, "timeout")
    if episodes:
        final_hits = [1.0 if ep.get("final_goal_on_step") is not None else 0.0 for ep in episodes]
        summary["final_goal_on_rate"] = mean(final_hits)
    else:
        summary["final_goal_on_rate"] = None
    for key in [
        "target_switch_count",
        "early_switch_count",
        "mean_commitment_length",
        "stall_count",
        "drift_count",
        "recovery_attempt_count",
        "global_replan_request_count",
        "global_replan_request_rate_per_100_steps",
        "max_consecutive_replan_burst",
        "segment_target_reach_rate",
        "mean_segment_progress",
        "mean_distance_to_path",
        "final_goal_switch_count",
        "final_goal_stall_count",
        "path_changed_from_gas_rate",
        "path_min_reachability",
        "churn_guard_trigger_count",
        "fallback_to_gas_count",
        "fallback_to_gas_step_count",
        "replan_suppressed_by_cooldown_count",
        "replan_suppressed_by_budget_count",
        "recovery_suppressed_by_lockout_count",
        "churn_guard_active_on_timeout",
        "cage_safe_mode_enabled",
        "cage_trace_only",
        "cage_contract_commit",
        "contract_model_loaded",
        "contract_gate_pass_count",
        "contract_gate_reject_count",
        "contract_gate_reject_rate",
        "contract_fallback_to_gas_when_uncertain_count",
        "contract_recovery_reject_count",
        "contract_final_goal_reject_count",
    ]:
        summary[key] = mean_values(episodes, key)
    attempts = [numeric(ep.get("recovery_attempt_count")) or 0.0 for ep in episodes]
    successes = [numeric(ep.get("recovery_success_count")) or 0.0 for ep in episodes]
    total_attempts = sum(attempts)
    summary["recovery_success_rate"] = sum(successes) / total_attempts if total_attempts else None
    return summary


def mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    clean = [numeric(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return None, None
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0


def group_rows(job_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in job_summaries:
        groups[(str(row.get("env_name")), str(row.get("variant")))].append(row)
    out: list[dict[str, Any]] = []
    for (env_name, variant), rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "env_name": env_name,
            "variant": variant,
            "num_jobs": len(rows),
            "num_succeeded": sum(1 for row in rows if row.get("succeeded")),
        }
        for metric in PRIMARY_JOB_METRICS:
            avg, sd = mean_std([row.get(metric) for row in rows])
            if metric == "success_rate":
                record["success_rate_mean"] = avg
                record["success_rate_std"] = sd
            elif metric == "normalized_score":
                record["normalized_score_mean"] = avg
                record["normalized_score_std"] = sd
            elif metric == "target_switch_count":
                record["target_switch_count_mean"] = avg
                record["target_switch_count_std"] = sd
            else:
                record[f"{metric}_mean"] = avg
        out.append(record)
    return out


def paired_rows(job_summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in job_summaries:
        by_env[str(row.get("env_name"))].append(row)
    paired: list[dict[str, Any]] = []
    for env_name, rows in sorted(by_env.items()):
        gas_by_seed = {row.get("seed"): row for row in rows if row.get("variant") == "gas"}
        variants = sorted({row.get("variant") for row in rows if row.get("variant") != "gas"})
        for variant in variants:
            var_rows = [row for row in rows if row.get("variant") == variant]
            deltas: dict[str, list[float]] = defaultdict(list)
            matched = 0
            for row in var_rows:
                gas = gas_by_seed.get(row.get("seed"))
                if not gas:
                    continue
                matched += 1
                for metric in PAIR_METRICS:
                    left = numeric(row.get(metric))
                    right = numeric(gas.get(metric))
                    if left is not None and right is not None:
                        deltas[metric].append(left - right)
            if matched == 0:
                warnings.append(f"{env_name}/{variant}: no matched gas seed; using aggregate difference")
                gas_rows = [row for row in rows if row.get("variant") == "gas"]
                for metric in PAIR_METRICS:
                    left, _ = mean_std([row.get(metric) for row in var_rows])
                    right, _ = mean_std([row.get(metric) for row in gas_rows])
                    if left is not None and right is not None:
                        deltas[metric].append(left - right)
            record: dict[str, Any] = {"env_name": env_name, "variant": variant, "matched_pairs": matched}
            for metric in PAIR_METRICS:
                record[f"delta_{metric}"] = mean(deltas[metric]) if deltas.get(metric) else None
            paired.append(record)
    return paired, warnings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    fields = sorted({key for row in rows for key in row.keys()})
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(args: argparse.Namespace, grouped: list[dict[str, Any]], paired: list[dict[str, Any]], warnings: list[str], jobs: list[dict[str, Any]]) -> None:
    out_csv = Path(args.out_csv)
    write_csv(out_csv, grouped)
    paired_csv = out_csv.with_name(out_csv.stem + "_paired.csv")
    write_csv(paired_csv, paired)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"grouped": grouped, "paired": paired, "warnings": warnings, "jobs": jobs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None"
    out_md.write_text(
        "# CAGE Focused Experiment Summary\n\n"
        "## Grouped\n\n"
        + markdown_table(grouped)
        + "\n## Paired Comparisons\n\n"
        + markdown_table(paired)
        + "\n## Warnings\n\n"
        + warning_text
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    manifest = load_jsonl(Path(args.manifest_path))
    status = load_status_rows(input_root)
    jobs = [summarize_job(row, status.get(str(row.get("job_id")))) for row in manifest]
    grouped = group_rows(jobs)
    paired, warnings = paired_rows(jobs)
    write_outputs(args, grouped, paired, warnings, jobs)
    print(json.dumps({"groups": len(grouped), "paired_rows": len(paired), "warnings": warnings}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
