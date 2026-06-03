#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from stage30_official_gas_common import ARCHIVED_PRE_STAGE30_STATUS, write_csv
from stage31_official_gas_artifact_inventory import _classify_env


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row.setdefault("source_file", str(path))
    return rows


def _collect(run_root: Path, relative_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in relative_paths:
        for path in sorted(run_root.glob(rel)):
            rows.extend(_read_csv(path))
    return rows


def _collect_many(run_roots: list[Path], relative_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_root in run_roots:
        rows.extend(_collect(run_root, relative_paths))
    return rows


def _truthy(value: Any) -> bool:
    return str(value) in {"1", "1.0", "true", "True"}


def _float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - half, center + half


def _mean(values: Iterable[Any]) -> float:
    xs = [_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _pearson(xs_raw: Iterable[Any], ys_raw: Iterable[Any]) -> float:
    xs = [_float(x) for x in xs_raw]
    ys = [_float(y) for y in ys_raw]
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs2 = [x for x, _ in pairs]
    ys2 = [y for _, y in pairs]
    mx = sum(xs2) / len(xs2)
    my = sum(ys2) / len(ys2)
    vx = sum((x - mx) ** 2 for x in xs2)
    vy = sum((y - my) ** 2 for y in ys2)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def _inventory_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_csv(path) if path else []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        out[(str(row.get("env_name", "")), str(row.get("seed", "")))] = row
    return out


def _metadata_for(row: dict[str, Any], inventory: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    key = (str(row.get("env_name", "")), str(row.get("seed", "")))
    inv = inventory.get(key, {})
    if inv:
        return {
            "env_family": inv.get("env_family", ""),
            "env_size": inv.get("env_size", ""),
            "task_type": inv.get("task_type", ""),
            "dataset_type": inv.get("dataset_type", ""),
            "observation_type": inv.get("observation_type", ""),
            "target_tier": inv.get("target_tier", ""),
            "artifact_status": inv.get("artifact_status", ""),
        }
    return {**_classify_env(str(row.get("env_name", ""))), "artifact_status": ""}


def _assign_wide_labels(row: dict[str, Any]) -> dict[str, str]:
    if _truthy(row.get("success")):
        return {
            "outcome_label": "SUCCESS",
            "failure_label": "NOT_FAILED",
            "failure_phase": "not_failed",
            "stage31_label_evidence": "official episode success=1",
        }
    if _truthy(row.get("no_path")):
        return {
            "outcome_label": "FAILURE",
            "failure_label": "NO_OFFICIAL_GRAPH_PATH",
            "failure_phase": "no_official_graph_path",
            "stage31_label_evidence": "official episode no_path=1",
        }
    source = str(row.get("first_failed_edge_source", ""))
    if source == "final_goal_phase_no_keygraph_edge":
        return {
            "outcome_label": "FAILURE",
            "failure_label": "GOAL_INTERFACE_FAILURE",
            "failure_phase": "final_goal_phase",
            "stage31_label_evidence": "active official path reached final-goal phase before failure",
        }
    if _truthy(row.get("stuck")):
        return {
            "outcome_label": "FAILURE",
            "failure_label": "POLICY_LOCAL_FAILURE",
            "failure_phase": "keygraph_subgoal" if _truthy(row.get("first_failed_edge_reliable")) else "stuck",
            "stage31_label_evidence": "official episode stuck/progress heuristic triggered",
        }
    if _truthy(row.get("timeout")):
        return {
            "outcome_label": "FAILURE",
            "failure_label": "SUBGOAL_SEQUENCE_DRIFT",
            "failure_phase": "keygraph_subgoal" if _truthy(row.get("first_failed_edge_reliable")) else "timeout",
            "stage31_label_evidence": "official episode timeout without edge-probe evidence",
        }
    if _truthy(row.get("divergence")):
        return {
            "outcome_label": "FAILURE",
            "failure_label": "SUBGOAL_SEQUENCE_DRIFT",
            "failure_phase": "keygraph_subgoal" if _truthy(row.get("first_failed_edge_reliable")) else "divergence",
            "stage31_label_evidence": "official episode final goal distance increased",
        }
    if _truthy(row.get("first_failed_edge_reliable")):
        return {
            "outcome_label": "FAILURE",
            "failure_label": "UNRESOLVED_FAILURE",
            "failure_phase": "keygraph_subgoal",
            "stage31_label_evidence": "first failed official keygraph edge is reliable, but wide pass has no edge-probe evidence",
        }
    return {
        "outcome_label": "FAILURE",
        "failure_label": "UNRESOLVED_FAILURE",
        "failure_phase": "unresolved_failure",
        "stage31_label_evidence": "wide pass lacks sufficient episode evidence for a specific failure label",
    }


def _annotate_episodes(episodes: list[dict[str, Any]], inventory: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in episodes:
        meta = _metadata_for(row, inventory)
        labels = _assign_wide_labels(row)
        out.append(
            {
                "stage": "stage31_wide_official_gas_behavior_atlas",
                "evidence_class": "OFFICIAL_GAS_WIDE_EPISODE_TRACE",
                "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
                **meta,
                **row,
                **labels,
            }
        )
    return out


def _group(rows: list[dict[str, Any]], fields: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(field, "")) for field in fields)].append(row)
    return out


def _summary(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, part in sorted(_group(rows, fields).items()):
        n = len(part)
        success = sum(1 for r in part if _truthy(r.get("success")))
        failed = n - success
        no_path = sum(1 for r in part if _truthy(r.get("no_path")))
        timeout = sum(1 for r in part if _truthy(r.get("timeout")))
        stuck = sum(1 for r in part if _truthy(r.get("stuck")))
        divergence = sum(1 for r in part if _truthy(r.get("divergence")))
        lo, hi = _wilson(success, n)
        row = {field: value for field, value in zip(fields, key)}
        row.update(
            {
                "episodes": n,
                "success_count": success,
                "failure_count": failed,
                "success_rate": success / max(1, n),
                "success_ci95_low": lo,
                "success_ci95_high": hi,
                "no_path_rate": no_path / max(1, n),
                "timeout_rate": timeout / max(1, n),
                "stuck_rate": stuck / max(1, n),
                "divergence_rate": divergence / max(1, n),
                "mean_steps": _mean(r.get("steps", "") for r in part),
                "mean_final_goal_dist_phi": _mean(r.get("final_goal_dist_phi", "") for r in part),
                "mean_subgoal_reach_rate": _mean(r.get("subgoal_reach_rate", "") for r in part),
                "mean_path_update_count": _mean(r.get("path_update_count", "") for r in part),
                "mean_cached_path_miss_count": _mean(r.get("cached_path_miss_count", "") for r in part),
            }
        )
        out.append(row)
    return out


def _count_summary(rows: list[dict[str, Any]], fields: list[str], count_field: str, denominator: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, part in sorted(_group(rows, fields).items()):
        denom_rows = [r for r in part if str(r.get("outcome_label", "")) == denominator] if denominator else part
        denom = len(denom_rows)
        counts = Counter(str(r.get(count_field, "")) for r in denom_rows)
        for label, count in sorted(counts.items()):
            lo, hi = _wilson(count, denom)
            row = {field: value for field, value in zip(fields, key)}
            row.update({count_field: label, "count": count, "denominator": denom, "rate": count / max(1, denom), "ci95_low": lo, "ci95_high": hi})
            out.append(row)
    return out


def _path_dynamics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    fields = ["env_name", "env_family", "dataset_type", "observation_type"]
    for key, part in sorted(_group(rows, fields).items()):
        fail_flags = [0 if _truthy(r.get("success")) else 1 for r in part]
        row = {field: value for field, value in zip(fields, key)}
        row.update(
            {
                "episodes": len(part),
                "path_update_failure_corr": _pearson((r.get("path_update_count", "") for r in part), fail_flags),
                "cached_path_miss_failure_corr": _pearson((r.get("cached_path_miss_count", "") for r in part), fail_flags),
                "active_path_trace_failure_corr": _pearson((r.get("active_path_trace_count", "") for r in part), fail_flags),
                "mean_path_update_success": _mean(r.get("path_update_count", "") for r in part if _truthy(r.get("success"))),
                "mean_path_update_failure": _mean(r.get("path_update_count", "") for r in part if not _truthy(r.get("success"))),
                "mean_cached_path_miss_success": _mean(r.get("cached_path_miss_count", "") for r in part if _truthy(r.get("success"))),
                "mean_cached_path_miss_failure": _mean(r.get("cached_path_miss_count", "") for r in part if not _truthy(r.get("success"))),
            }
        )
        out.append(row)
    return out


def _compute_manifest(protocols: list[dict[str, Any]], episodes: list[dict[str, Any]], run_root: str, out_dir: Path, command_line: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in protocols:
        rows.append(
            {
                "stage": "stage31_wide_official_gas_behavior_atlas",
                "evidence_class": "OFFICIAL_GAS_WIDE_COMPUTE_MANIFEST",
                "run_root": str(run_root),
                "out_root": str(out_dir),
                "env_name": row.get("env_name", ""),
                "seed": row.get("seed", ""),
                "task_id": row.get("task_id", ""),
                "episode_count": row.get("episode_count", ""),
                "eval_on_cpu": row.get("eval_on_cpu", ""),
                "gpu": row.get("gpu", ""),
                "wrapper_status": row.get("wrapper_status", ""),
                "official_repo_sha": row.get("official_repo_sha", ""),
                "keygraph_sha256": row.get("keygraph_sha256", ""),
                "policy_sha256": row.get("policy_sha256", ""),
                "tdr_sha256": row.get("tdr_sha256", ""),
                "evaluation_command": row.get("evaluation_command", ""),
            }
        )
    rows.append(
        {
            "stage": "stage31_wide_official_gas_behavior_atlas",
            "evidence_class": "OFFICIAL_GAS_WIDE_COLLECTOR_COMMAND",
            "run_root": str(run_root),
            "out_root": str(out_dir),
            "episode_rows_collected": len(episodes),
            "collector_command": command_line,
        }
    )
    return rows


def _write_report(out_dir: Path, episodes: list[dict[str, Any]], by_env: list[dict[str, Any]], phase_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> None:
    failed = [r for r in episodes if str(r.get("outcome_label", "")) == "FAILURE"]
    unresolved_failed = sum(1 for r in failed if str(r.get("failure_label", "")) == "UNRESOLVED_FAILURE")
    failure_dense = sorted(by_env, key=lambda r: float(r.get("failure_count", 0) or 0) / max(1, float(r.get("episodes", 1) or 1)), reverse=True)[:10]
    phase_counts = Counter(str(r.get("failure_phase", "")) for r in failed)
    label_counts = Counter(str(r.get("failure_label", "")) for r in failed)
    lines = [
        "# Stage31 Wide Official GAS Behavior Atlas",
        "",
        "Status: OFFICIAL_GAS_WIDE_BEHAVIOR_ATLAS.",
        f"Pre-Stage30 BARS/Stage28/Stage29 evidence: {ARCHIVED_PRE_STAGE30_STATUS}.",
        "Official GAS graph, planner, policy, subgoal selection, and action outputs are unchanged.",
        "This wide pass uses episode-level instrumentation only; no same/cross/dt semantics are inferred without exact official provenance, and no edge-probe labels are assigned here.",
        "",
        "## Aggregate",
        "",
        f"- episodes: {len(episodes)}",
        f"- failed episodes: {len(failed)}",
        f"- unresolved failures / failed episodes: {unresolved_failed}/{max(1, len(failed))} ({unresolved_failed / max(1, len(failed)):.4f})",
        "",
        "## Failure-Dense Envs",
        "",
        "| env_name | episodes | success_rate | no_path_rate | timeout_rate | stuck_rate | divergence_rate | mean_subgoal_reach_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in failure_dense:
        lines.append(
            f"| {row.get('env_name', '')} | {row.get('episodes', '')} | {float(row.get('success_rate', 0) or 0):.4f} | {float(row.get('no_path_rate', 0) or 0):.4f} | {float(row.get('timeout_rate', 0) or 0):.4f} | {float(row.get('stuck_rate', 0) or 0):.4f} | {float(row.get('divergence_rate', 0) or 0):.4f} | {float(row.get('mean_subgoal_reach_rate', 0) or 0):.4f} |"
        )
    lines.extend(["", "## Failure Phase Counts", ""])
    for label, count in phase_counts.most_common():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Failure Label Counts", ""])
    for label, count in label_counts.most_common():
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- all episodes: `{out_dir / 'stage31_all_episode_traces.csv'}`",
            f"- all path edges: `{out_dir / 'stage31_all_path_edges.csv'}`",
            f"- summary by env: `{out_dir / 'stage31_success_by_env.csv'}`",
            f"- failure phases: `{out_dir / 'stage31_failure_phase_summary.csv'}`",
            f"- failure labels: `{out_dir / 'stage31_failure_label_summary.csv'}`",
            f"- path dynamics: `{out_dir / 'stage31_path_dynamics_by_env.csv'}`",
            f"- task sensitivity: `{out_dir / 'stage31_task_sensitivity.csv'}`",
            f"- compute manifest: `{out_dir / 'stage31_compute_manifest.csv'}`",
        ]
    )
    (out_dir / "stage31_wide_behavior_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Stage31 wide official GAS behavior atlas outputs.")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--run-roots", default="", help="Comma-separated run roots. Overrides --run-root when set.")
    parser.add_argument("--inventory-csv", default="")
    parser.add_argument("--out-root", default="")
    args = parser.parse_args()

    run_roots = [Path(x.strip()) for x in args.run_roots.split(",") if x.strip()] if args.run_roots else []
    if not run_roots and args.run_root:
        run_roots = [Path(args.run_root)]
    if not run_roots:
        raise SystemExit("--run-root or --run-roots is required")
    run_root_label = ",".join(str(x) for x in run_roots)
    out_dir = Path(args.out_root) if args.out_root else run_roots[0] / "global"
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = _inventory_index(Path(args.inventory_csv)) if args.inventory_csv else {}

    episodes_raw = _collect_many(run_roots, ["*/seed*/instrumentation/official_gas_episode_traces.csv"])
    path_edges = _collect_many(run_roots, ["*/seed*/instrumentation/official_gas_path_edges.csv"])
    protocols = _collect_many(run_roots, ["*/seed*/instrumentation/protocol_lock.csv"])
    episodes = _annotate_episodes(episodes_raw, inventory)

    by_env_seed_task = _summary(episodes, ["env_name", "seed", "task_id", "env_family", "dataset_type", "observation_type", "target_tier"])
    by_env = _summary(episodes, ["env_name", "env_family", "dataset_type", "observation_type", "target_tier"])
    by_family = _summary(episodes, ["env_family", "dataset_type", "observation_type", "target_tier"])
    phase_rows = _count_summary(episodes, ["env_name", "env_family", "dataset_type", "observation_type"], "failure_phase", "FAILURE")
    label_rows = _count_summary(episodes, ["env_name", "env_family", "dataset_type", "observation_type"], "failure_label", "FAILURE")
    task_rows = _summary(episodes, ["env_name", "task_id", "env_family", "dataset_type", "observation_type", "target_tier"])
    dynamics_rows = _path_dynamics(episodes)
    command_line = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    manifest_rows = _compute_manifest(protocols, episodes, run_root_label, out_dir, command_line)

    write_csv(out_dir / "stage31_all_episode_traces.csv", episodes)
    write_csv(out_dir / "stage31_all_path_edges.csv", path_edges)
    write_csv(out_dir / "stage31_all_protocol_lock.csv", protocols)
    write_csv(out_dir / "stage31_success_by_env_seed_task.csv", by_env_seed_task)
    write_csv(out_dir / "stage31_success_by_env.csv", by_env)
    write_csv(out_dir / "stage31_success_by_family.csv", by_family)
    write_csv(out_dir / "stage31_failure_phase_summary.csv", phase_rows)
    write_csv(out_dir / "stage31_failure_label_summary.csv", label_rows)
    write_csv(out_dir / "stage31_task_sensitivity.csv", task_rows)
    write_csv(out_dir / "stage31_path_dynamics_by_env.csv", dynamics_rows)
    write_csv(out_dir / "stage31_compute_manifest.csv", manifest_rows)
    _write_report(out_dir, episodes, by_env, phase_rows, label_rows)
    print(out_dir / "stage31_wide_behavior_report.md")


if __name__ == "__main__":
    main()
