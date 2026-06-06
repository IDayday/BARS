#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze deployed CAGE-v0.3 contract-rank traces and compact outputs.")
    parser.add_argument("--input_root", default="", help="Single experiment root.")
    parser.add_argument("--input_roots", nargs="*", default=[], help="One or more experiment roots.")
    parser.add_argument("--out_csv", default="results/cage_v03_contract_rank/deployment_analysis.csv")
    parser.add_argument("--out_md", default="reports/stage33_contract_rank_deployment.md")
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
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_manifest_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "manifests").glob("*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def load_status(root: Path) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "status").glob("*.jsonl")):
        for row in load_jsonl(path):
            job_id = row.get("job_id")
            if job_id:
                status[str(job_id)] = row
    return status


def load_eval_success(path_value: str | None) -> float | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    row = rows[-1]
    values = [numeric(value) for key, value in row.items() if key.endswith("episode.success")]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [numeric(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def source_distribution(step_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> dict[str, float]:
    counter: Counter[str] = Counter()
    for row in step_rows:
        source = row.get("contract_selected_source")
        if source:
            counter[str(source)] += 1
    if not counter:
        aggregate_keys = {
            "gas": "contract_rank_choose_gas_count",
            "cage": "contract_rank_choose_cage_count",
            "committed": "contract_rank_choose_committed_count",
        }
        for source, key in aggregate_keys.items():
            total = sum(int(numeric(row.get(key)) or 0) for row in episode_rows)
            if total:
                counter[source] += total
    total = sum(counter.values())
    if total == 0:
        return {}
    return {source: count / total for source, count in sorted(counter.items())}


def summarize_job(row: dict[str, Any], status: dict[str, Any] | None) -> dict[str, Any]:
    trace_rows = load_jsonl(Path(row.get("cage_trace_path") or ""))
    episodes = [item for item in trace_rows if item.get("record_type", "episode") == "episode"]
    steps = [item for item in trace_rows if item.get("record_type") == "step"]
    success = mean_key(episodes, "success")
    if success is None:
        success = load_eval_success(row.get("result_path"))
    dist = source_distribution(steps, episodes)
    coverage = mean_key(steps, "contract_candidate_coverage")
    if coverage is None:
        coverage = mean_key(episodes, "mean_contract_candidate_coverage")
    candidate_count = mean_key(steps, "contract_candidate_count")
    selected_score = mean_key(steps, "contract_selected_score")
    if selected_score is None:
        selected_score = mean_key(episodes, "mean_contract_selected_score")
    gas_score = mean_key(steps, "contract_gas_score")
    if gas_score is None:
        gas_score = mean_key(episodes, "mean_contract_gas_score")
    segment_reach = mean_key(episodes, "segment_target_reach_rate")
    stall = mean_key(episodes, "stall_count")
    final_goal_on = 0.0
    if episodes:
        final_goal_on = mean([1.0 if ep.get("final_goal_on_step") is not None else 0.0 for ep in episodes])
    local_safe_loop = bool(
        (segment_reach is not None and segment_reach >= 0.8)
        and (success is not None and success <= 0.2)
        and (final_goal_on <= 0.2)
    )
    return {
        "job_id": row.get("job_id"),
        "env_name": row.get("env_name"),
        "seed": row.get("seed"),
        "variant": row.get("variant"),
        "status": status.get("status") if status else None,
        "success_rate": success,
        "contract_candidate_coverage": coverage,
        "contract_candidate_count": candidate_count,
        "contract_selected_score": selected_score,
        "contract_gas_score": gas_score,
        "contract_best_non_gas_score": mean_key(steps, "contract_best_non_gas_score"),
        "rejected_count": mean_key(steps, "contract_rejected_count"),
        "extreme_negative_reject_count": mean_key(episodes, "contract_rank_extreme_reject_count"),
        "fallback_to_gas_step_count": mean_key(episodes, "fallback_to_gas_step_count"),
        "segment_target_reach_rate": segment_reach,
        "mean_segment_progress": mean_key(episodes, "mean_segment_progress"),
        "stall_count": stall,
        "global_replan_request_count": mean_key(episodes, "global_replan_request_count"),
        "final_goal_on_rate": final_goal_on if episodes else None,
        "selected_source_distribution": json.dumps(dist, sort_keys=True),
        "source_gas_rate": dist.get("gas"),
        "source_cage_rate": dist.get("cage"),
        "source_committed_rate": dist.get("committed"),
        "local_safe_loop": local_safe_loop,
    }


def group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("env_name")), str(row.get("variant")))].append(row)
    out: list[dict[str, Any]] = []
    for (env_name, variant), items in sorted(grouped.items()):
        record = {"env_name": env_name, "variant": variant, "num_jobs": len(items)}
        for key in [
            "success_rate",
            "contract_candidate_coverage",
            "contract_candidate_count",
            "contract_selected_score",
            "contract_gas_score",
            "contract_best_non_gas_score",
            "rejected_count",
            "extreme_negative_reject_count",
            "fallback_to_gas_step_count",
            "segment_target_reach_rate",
            "mean_segment_progress",
            "stall_count",
            "global_replan_request_count",
            "final_goal_on_rate",
            "source_gas_rate",
            "source_cage_rate",
            "source_committed_rate",
        ]:
            values = [numeric(item.get(key)) for item in items]
            values = [value for value in values if value is not None]
            record[f"{key}_mean"] = mean(values) if values else None
        record["local_safe_loop_count"] = sum(1 for item in items if item.get("local_safe_loop"))
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
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
    fields = sorted({key for row in rows for key in row})
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    roots = [Path(x) for x in args.input_roots]
    if args.input_root:
        roots.append(Path(args.input_root))
    if not roots:
        roots = [Path("results/cage_v03_contract_rank")]
    job_rows: list[dict[str, Any]] = []
    for root in roots:
        manifests = load_manifest_rows(root)
        status = load_status(root)
        for row in manifests:
            job_rows.append(summarize_job(row, status.get(str(row.get("job_id")))))
    grouped = group(job_rows)
    out_csv = Path(args.out_csv)
    write_csv(out_csv, grouped)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "# Stage33 Contract-Rank Deployment Analysis\n\n"
        "本表汇总 `cage_contract_rank` 部署后的 success、coverage、候选来源、拒绝计数和局部安全循环代理指标。缺失字段记为 NA。\n\n"
        "## Grouped\n\n"
        + markdown_table(grouped)
        + "\n## Job Rows\n\n"
        + markdown_table(job_rows),
        encoding="utf-8",
    )
    print(json.dumps({"out_csv": str(out_csv), "out_md": str(out_md), "num_jobs": len(job_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
