#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage30_official_gas_analyze import TAXONOMY, _assign_label, _edge_probe_index, _summary_rows
from stage30_official_gas_common import ARCHIVED_PRE_STAGE30_STATUS, write_csv


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


def _episode_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("env_name", "")),
        str(row.get("seed", "")),
        str(row.get("task_id", "")),
        str(row.get("episode_id", "")),
    )


def _path_edge_index(path_edges: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in path_edges:
        out[_episode_key(row)].append(row)
    return out


def _taxonomy_rows(episodes: list[dict[str, Any]], path_edges: list[dict[str, Any]], probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probe_by_edge = _edge_probe_index(probes)
    edges_by_episode = _path_edge_index(path_edges)
    out: list[dict[str, Any]] = []
    for ep in episodes:
        edges = edges_by_episode.get(_episode_key(ep), [])
        label, evidence = _assign_label(ep, edges, probe_by_edge)
        out.append(
            {
                "stage": "stage30_official_gas_global_failure_taxonomy",
                "evidence_class": "OFFICIAL_GAS_EPISODE_PATH_AND_EDGE_EVIDENCE",
                "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
                **ep,
                "taxonomy_label": label,
                "taxonomy_evidence": evidence,
            }
        )
    return out


def _dominant_label(rows: list[dict[str, Any]]) -> tuple[str, int, float]:
    labels = [str(r.get("taxonomy_label", "")) for r in rows if str(r.get("taxonomy_label", "")) not in {"", "UNRESOLVED"}]
    if not labels:
        return "NO_STABLE_DOMINANT_FAILURE_MODE", 0, 0.0
    label, count = Counter(labels).most_common(1)[0]
    return label, count, count / max(1, len(rows))


def _write_report(out_dir: Path, episodes: list[dict[str, Any]], probes: list[dict[str, Any]], taxonomy: list[dict[str, Any]], by_env: list[dict[str, Any]]) -> None:
    by_probe_mode = Counter(str(r.get("probe_mode", "unavailable")) for r in probes)
    unresolved = sum(1 for r in taxonomy if str(r.get("taxonomy_label")) == "UNRESOLVED")
    dominant, dominant_count, dominant_rate = _dominant_label(taxonomy)
    lines = [
        "# Stage30 Official GAS Global Diagnosis Collector",
        "",
        "Status: OFFICIAL_GAS_GLOBAL_DIAGNOSIS_AGGREGATE.",
        f"Pre-Stage30 BARS/Stage28/Stage29 evidence: {ARCHIVED_PRE_STAGE30_STATUS}.",
        "No GAS graph, planner, policy, subgoal selection, or action outputs are modified by this collector.",
        "",
        "## Aggregate Counts",
        "",
        f"- episodes: {len(episodes)}",
        f"- edge probe rows: {len(probes)}",
        f"- unresolved taxonomy: {unresolved}/{len(taxonomy)} ({unresolved / max(1, len(taxonomy)):.4f})",
        f"- dominant evidence-backed label: `{dominant}` count={dominant_count}, rate={dominant_rate:.4f}",
        "",
        "## Probe Modes",
        "",
    ]
    for mode, count in sorted(by_probe_mode.items()):
        lines.append(f"- {mode}: {count}")
    lines.extend(
        [
            "",
            "## Taxonomy By Env",
            "",
            "| env_name | label | count | episodes | rate | ci95_low | ci95_high |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in by_env:
        if int(row.get("count", 0) or 0) == 0:
            continue
        lines.append(
            f"| {row.get('env_name', '')} | {row.get('taxonomy_label', '')} | {row.get('count', '')} | {row.get('episodes', '')} | {float(row.get('rate', 0) or 0):.4f} | {float(row.get('ci95_low', 0) or 0):.4f} | {float(row.get('ci95_high', 0) or 0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- all episode traces: `{out_dir / 'stage30_all_episode_traces.csv'}`",
            f"- all path edges: `{out_dir / 'stage30_all_path_edges.csv'}`",
            f"- all keygraph edges: `{out_dir / 'stage30_all_keygraph_edges.csv'}`",
            f"- all edge probes: `{out_dir / 'stage30_all_edge_probe.csv'}`",
            f"- global taxonomy: `{out_dir / 'stage30_failure_taxonomy.csv'}`",
        ]
    )
    (out_dir / "stage30_global_diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Stage30 official GAS layered diagnosis outputs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-root", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_root) if args.out_root else run_root / "global"
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = _collect(run_root, ["*/seed*/instrumentation/official_gas_episode_traces.csv"])
    path_edges = _collect(run_root, ["*/seed*/instrumentation/official_gas_path_edges.csv"])
    keygraph_edges = _collect(run_root, ["*/seed*/keygraph_audit/official_gas_keygraph_edges.csv"])
    probes = _collect(
        run_root,
        [
            "*/seed*/exact_semantic_probe/official_gas_edge_probe.csv",
            "*/seed*/nearest_execution_probe/official_gas_edge_probe.csv",
            "*/seed*/edge_probe/official_gas_edge_probe.csv",
        ],
    )
    protocols = _collect(run_root, ["*/seed*/**/protocol_lock.csv"])

    taxonomy = _taxonomy_rows(episodes, path_edges, probes)
    by_seed = _summary_rows(taxonomy, ["env_name", "seed"])
    by_env = _summary_rows(taxonomy, ["env_name"])
    overall = _summary_rows(taxonomy, ["stage"])

    write_csv(out_dir / "stage30_all_episode_traces.csv", episodes)
    write_csv(out_dir / "stage30_all_path_edges.csv", path_edges)
    write_csv(out_dir / "stage30_all_keygraph_edges.csv", keygraph_edges)
    write_csv(out_dir / "stage30_all_edge_probe.csv", probes)
    write_csv(out_dir / "stage30_all_protocol_lock.csv", protocols)
    write_csv(out_dir / "stage30_failure_taxonomy.csv", taxonomy)
    write_csv(out_dir / "stage30_failure_taxonomy_by_seed.csv", by_seed)
    write_csv(out_dir / "stage30_failure_taxonomy_by_env.csv", by_env)
    write_csv(out_dir / "stage30_failure_taxonomy_overall.csv", overall)
    write_csv(
        out_dir / "stage30_collector_command.csv",
        [
            {
                "stage": "stage30_official_gas_global_collector",
                "evidence_class": "OFFICIAL_GAS_GLOBAL_DIAGNOSIS_AGGREGATE",
                "run_root": str(run_root),
                "out_root": str(out_dir),
                "command": " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]),
            }
        ],
    )
    _write_report(out_dir, episodes, probes, taxonomy, by_env)
    print(out_dir / "stage30_global_diagnosis_report.md")


if __name__ == "__main__":
    main()
