#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage30_official_gas_common import (
    ARCHIVED_PRE_STAGE30_STATUS,
    gas_source_identity,
    parse_csv_list,
    parse_seed_list,
    protocol_lock_row,
    scan_official_artifacts,
    write_csv,
)


TAXONOMY = [
    "SUCCESS_WITH_CROSS_STITCHING",
    "NO_OFFICIAL_GRAPH_PATH",
    "KEYGRAPH_ABSTRACTION_LOST_SUPPORT",
    "CROSS_EDGE_EXECUTION_FAILURE",
    "TEMPORAL_EDGE_EXECUTION_FAILURE",
    "LONG_HOP_FAILURE",
    "SUBGOAL_SEQUENCE_DRIFT",
    "GOAL_INTERFACE_FAILURE",
    "POLICY_LOCAL_FAILURE",
    "LOW_TE_OR_NOISY_NODE_FAILURE",
    "UNRESOLVED",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_many_csvs(raw: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        rows.extend(_read_csv(Path(item)))
    return rows


def _edge_key_text(u: Any, v: Any) -> str:
    try:
        return f"{int(float(u))}->{int(float(v))}"
    except Exception:
        return ""


def _truthy(value: Any) -> bool:
    return str(value) in {"1", "1.0", "true", "True"}


def _edge_probe_index(probe_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in probe_rows:
        key = _edge_key_text(row.get("u"), row.get("v"))
        if key:
            out[key].append(row)
    return out


def _episode_edges(path_edges: list[dict[str, str]], env_name: str, seed: str, task_id: str, episode_id: str) -> list[dict[str, str]]:
    return [
        r
        for r in path_edges
        if r.get("env_name") == env_name and r.get("seed") == seed and r.get("task_id") == task_id and r.get("episode_id") == episode_id
    ]


def _assign_label(ep: dict[str, str], edges: list[dict[str, str]], probe_by_edge: dict[str, list[dict[str, str]]]) -> tuple[str, str]:
    if _truthy(ep.get("no_path")):
        return "NO_OFFICIAL_GRAPH_PATH", "official episode trace has no_path=1"
    if _truthy(ep.get("success")):
        used_cross = any(
            str(e.get("trajectory_semantics_valid", e.get("cross_trajectory_available"))) in {"1", "1.0"}
            and
            str(e.get("cross_trajectory_available")) in {"1", "1.0"}
            and (str(e.get("cross_trajectory")) in {"1", "1.0"} or str(e.get("edge_category")) == "cross_trajectory_keygraph_edge")
            for e in edges
        )
        if used_cross:
            return "SUCCESS_WITH_CROSS_STITCHING", "successful official episode used cross-trajectory path edge evidence"
        return "UNRESOLVED", "successful episode without cross-stitch evidence"

    failed_edge = ep.get("first_failed_edge_id", "") or ep.get("first_failed_edge", "")
    first_failed_edge_reliable = _truthy(ep.get("first_failed_edge_reliable", "1" if failed_edge else "0"))
    probes = probe_by_edge.get(failed_edge, []) if failed_edge and first_failed_edge_reliable else []
    valid_probes = [p for p in probes if _truthy(p.get("valid_probe"))]
    failed_probe = any(_truthy(p.get("valid_probe")) and not _truthy(p.get("reach")) for p in valid_probes)
    if valid_probes and failed_probe:
        edge_categories = {str(p.get("edge_category", "")) for p in valid_probes}
        sample_categories = {str(p.get("category", "")) for p in valid_probes}
        recoverable_cross_failure = any(
            str(p.get("trajectory_semantics_valid")) in {"1", "1.0"}
            and
            str(p.get("cross_trajectory_available")) in {"1", "1.0"}
            and (str(p.get("cross_trajectory")) in {"1", "1.0"} or str(p.get("edge_category")) == "cross_trajectory_keygraph_edge")
            for p in valid_probes
        )
        recoverable_temporal_failure = any(
            str(p.get("trajectory_semantics_valid")) in {"1", "1.0"} and "same_trajectory_temporal_like" in str(p.get("edge_category", ""))
            for p in valid_probes
        )
        if recoverable_cross_failure:
            return "CROSS_EDGE_EXECUTION_FAILURE", f"first_failed_edge={failed_edge} has failed official edge probe"
        if recoverable_temporal_failure:
            return "TEMPORAL_EDGE_EXECUTION_FAILURE", f"first_failed_edge={failed_edge} has failed official edge probe"
        if "long_hop_edges" in sample_categories or "high_cost_edges" in sample_categories:
            return "LONG_HOP_FAILURE", f"first_failed_edge={failed_edge} failed in official long-hop/high-cost probe category"
        if "low_te_edges" in sample_categories:
            return "LOW_TE_OR_NOISY_NODE_FAILURE", f"first_failed_edge={failed_edge} failed in official low-TE probe category"
        return "POLICY_LOCAL_FAILURE", f"first_failed_edge={failed_edge} has failed official edge probe"
    if _truthy(ep.get("timeout")):
        return "SUBGOAL_SEQUENCE_DRIFT", "episode timed out without edge-probe-supported first-edge label"
    if _truthy(ep.get("stuck")):
        return "POLICY_LOCAL_FAILURE", "episode stuck by progress heuristic without edge-probe-supported edge label"
    return "UNRESOLVED", "no sufficient episode + edge execution evidence for taxonomy label"


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - half, center + half


def _summary_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, part in sorted(grouped.items()):
        n = len(part)
        counts = Counter(str(r.get("taxonomy_label", "")) for r in part)
        for label in TAXONOMY:
            k = counts.get(label, 0)
            lo, hi = _wilson(k, n)
            row = {field: value for field, value in zip(group_fields, key)}
            row.update({"taxonomy_label": label, "count": k, "episodes": n, "rate": k / max(1, n), "ci95_low": lo, "ci95_high": hi})
            out.append(row)
    return out


def _dominant_label(rows: list[dict[str, Any]]) -> tuple[str, int, float]:
    labels = [str(r.get("taxonomy_label", "")) for r in rows if str(r.get("taxonomy_label", "")) not in {"", "UNRESOLVED"}]
    if not labels:
        return "NO_STABLE_DOMINANT_FAILURE_MODE", 0, 0.0
    label, count = Counter(labels).most_common(1)[0]
    return label, count, count / max(1, len(rows))


def _write_report(out_dir: Path, rows: list[dict[str, Any]], by_env_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage30 Official GAS Failure Taxonomy Report",
        "",
        "Status: OFFICIAL_GAS_ONLY_FAILURE_TAXONOMY.",
        f"Pre-Stage30 BARS/Stage28/Stage29 evidence: {ARCHIVED_PRE_STAGE30_STATUS}.",
        "Labels are assigned only from official episode traces plus official edge execution probes; otherwise the label remains UNRESOLVED.",
        "",
        "| env_name | label | count | rate | ci95_low | ci95_high |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in by_env_rows:
        if int(row.get("count", 0) or 0) == 0:
            continue
        lines.append(
            f"| {row.get('env_name', '')} | {row.get('taxonomy_label', '')} | {row.get('count', '')} | {float(row.get('rate', 0) or 0):.4f} | {float(row.get('ci95_low', 0) or 0):.4f} | {float(row.get('ci95_high', 0) or 0):.4f} |"
        )
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_env[str(row.get("env_name", ""))].append(row)
    overall_label, overall_count, overall_rate = _dominant_label(rows)
    lines.extend(["", "## Dominant Evidence-Backed Labels", ""])
    lines.append(f"- overall: `{overall_label}` count={overall_count}, rate={overall_rate:.4f}")
    for env_name, part in sorted(by_env.items()):
        label, count, rate = _dominant_label(part)
        lines.append(f"- {env_name}: `{label}` count={count}, rate={rate:.4f}")
    lines.extend(["", "## Files", "", f"- taxonomy CSV: `{out_dir / 'official_gas_failure_taxonomy.csv'}`"])
    lines.append(f"- by seed aggregate: `{out_dir / 'official_gas_failure_taxonomy_by_seed.csv'}`")
    lines.append(f"- by env aggregate: `{out_dir / 'official_gas_failure_taxonomy_by_env.csv'}`")
    lines.append(f"- protocol lock: `{out_dir / 'protocol_lock.csv'}`")
    (out_dir / "failure_taxonomy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _infer_envs(episodes: list[dict[str, str]]) -> list[str]:
    return sorted({str(r.get("env_name", "")) for r in episodes if r.get("env_name")})


def _infer_seeds(episodes: list[dict[str, str]]) -> list[int]:
    out: set[int] = set()
    for row in episodes:
        try:
            out.add(int(float(row.get("seed", ""))))
        except Exception:
            pass
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage30 official-GAS-only failure taxonomy.")
    parser.add_argument("--episode-csv", required=True)
    parser.add_argument("--path-edge-csv", required=True)
    parser.add_argument("--edge-probe-csv", default="")
    parser.add_argument("--out-root", default="runs_stage30_official_gas/taxonomy")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--envs", default="auto")
    parser.add_argument("--seeds", default="auto")
    args = parser.parse_args()
    episodes = _read_csv(Path(args.episode_csv))
    path_edges = _read_csv(Path(args.path_edge_csv))
    probes = _read_many_csvs(args.edge_probe_csv) if args.edge_probe_csv else []
    probe_by_edge = _edge_probe_index(probes)
    out_rows: list[dict[str, Any]] = []
    for ep in episodes:
        edges = _episode_edges(path_edges, ep.get("env_name", ""), ep.get("seed", ""), ep.get("task_id", ""), ep.get("episode_id", ""))
        label, evidence = _assign_label(ep, edges, probe_by_edge)
        out_rows.append(
            {
                "stage": "stage30_official_gas_failure_taxonomy",
                "evidence_class": "OFFICIAL_GAS_EPISODE_AND_EDGE_EVIDENCE",
                "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
                **ep,
                "taxonomy_label": label,
                "taxonomy_evidence": evidence,
            }
        )
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_seed_rows = _summary_rows(out_rows, ["env_name", "seed"])
    by_env_rows = _summary_rows(out_rows, ["env_name"])
    overall_rows = _summary_rows(out_rows, ["stage"])
    envs = parse_csv_list(args.envs) if args.envs.lower() != "auto" else _infer_envs(episodes)
    seeds = parse_seed_list(args.seeds) if args.seeds.lower() != "auto" else _infer_seeds(episodes)
    gas_repo = Path(args.gas_repo_path)
    source_identity = gas_source_identity(gas_repo)
    command_line = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    protocol_rows = [
        protocol_lock_row(
            art,
            gas_repo,
            stage="stage30_official_gas_failure_taxonomy",
            evidence_class="OFFICIAL_GAS_PROTOCOL_LOCK",
            wrapper_status="OFFICIAL_GAS_ONLY_FAILURE_TAXONOMY",
            command_line=command_line,
            task_id="taxonomy",
            episode_count=len([r for r in episodes if r.get("env_name") == art.env_name and str(r.get("seed")) in {str(art.seed), f"{float(art.seed):.1f}"}]),
            source_identity=source_identity,
            extra={
                "episode_csv": args.episode_csv,
                "path_edge_csv": args.path_edge_csv,
                "edge_probe_csv": args.edge_probe_csv,
            },
        )
        for art in scan_official_artifacts(Path(args.artifact_root), envs, seeds)
    ]
    write_csv(out_dir / "official_gas_failure_taxonomy.csv", out_rows)
    write_csv(out_dir / "official_gas_failure_taxonomy_by_seed.csv", by_seed_rows)
    write_csv(out_dir / "official_gas_failure_taxonomy_by_env.csv", by_env_rows)
    write_csv(out_dir / "official_gas_failure_taxonomy_overall.csv", overall_rows)
    write_csv(out_dir / "protocol_lock.csv", protocol_rows)
    _write_report(out_dir, out_rows, by_env_rows)
    print(out_dir / "failure_taxonomy_report.md")


if __name__ == "__main__":
    main()
