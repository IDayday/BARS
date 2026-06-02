#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage30_official_gas_common import ARCHIVED_PRE_STAGE30_STATUS, write_csv


TAXONOMY = [
    "NO_OFFICIAL_GRAPH_PATH",
    "KEYGRAPH_ABSTRACTION_LOST_SUPPORT",
    "LOW_TE_OR_NOISY_NODE_FAILURE",
    "CROSS_EDGE_EXECUTION_FAILURE",
    "TEMPORAL_EDGE_EXECUTION_FAILURE",
    "SUBGOAL_SEQUENCE_DRIFT",
    "LONG_HOP_FAILURE",
    "GOAL_INTERFACE_FAILURE",
    "POLICY_LOCAL_FAILURE",
    "SUCCESS_WITH_CROSS_STITCHING",
    "UNRESOLVED",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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
        used_cross = any(str(e.get("cross_trajectory")) in {"1", "1.0"} or str(e.get("edge_category")) == "cross_trajectory_keygraph_edge" for e in edges)
        if used_cross:
            return "SUCCESS_WITH_CROSS_STITCHING", "successful official episode used cross-trajectory path edge evidence"
        return "UNRESOLVED", "successful episode without cross-stitch evidence"

    failed_edge = ep.get("first_failed_edge", "")
    probes = probe_by_edge.get(failed_edge, []) if failed_edge else []
    valid_probes = [p for p in probes if _truthy(p.get("valid_probe"))]
    failed_probe = any(_truthy(p.get("valid_probe")) and not _truthy(p.get("reach")) for p in valid_probes)
    if valid_probes and failed_probe:
        edge_categories = {str(p.get("edge_category", "")) for p in valid_probes}
        if "cross_trajectory_keygraph_edge" in edge_categories:
            return "CROSS_EDGE_EXECUTION_FAILURE", f"first_failed_edge={failed_edge} has failed official edge probe"
        if any("temporal" in c for c in edge_categories):
            return "TEMPORAL_EDGE_EXECUTION_FAILURE", f"first_failed_edge={failed_edge} has failed official edge probe"
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


def _write_report(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_env[str(row.get("env_name", ""))].append(row)
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
    for env_name, part in sorted(by_env.items()):
        n = len(part)
        counts = Counter(str(r.get("taxonomy_label", "")) for r in part)
        for label in TAXONOMY:
            k = counts.get(label, 0)
            if k == 0:
                continue
            lo, hi = _wilson(k, n)
            lines.append(f"| {env_name} | {label} | {k} | {k / max(1, n):.4f} | {lo:.4f} | {hi:.4f} |")
    lines.extend(["", "## Files", "", f"- taxonomy CSV: `{out_dir / 'official_gas_failure_taxonomy.csv'}`"])
    (out_dir / "failure_taxonomy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage30 official-GAS-only failure taxonomy.")
    parser.add_argument("--episode-csv", required=True)
    parser.add_argument("--path-edge-csv", required=True)
    parser.add_argument("--edge-probe-csv", default="")
    parser.add_argument("--out-root", default="runs_stage30_official_gas/taxonomy")
    args = parser.parse_args()
    episodes = _read_csv(Path(args.episode_csv))
    path_edges = _read_csv(Path(args.path_edge_csv))
    probes = _read_csv(Path(args.edge_probe_csv)) if args.edge_probe_csv else []
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
    write_csv(out_dir / "official_gas_failure_taxonomy.csv", out_rows)
    _write_report(out_dir, out_rows)
    print(out_dir / "failure_taxonomy_report.md")


if __name__ == "__main__":
    main()
