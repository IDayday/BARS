#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--out", default="reports/stage22r_decisions.md")
    args = p.parse_args()
    root = Path(args.reports_dir)
    decisions: list[str] = []
    evidence: list[str] = []

    comp_path = root / "stage22_variant_comparison.csv"
    reach_overall_hold = False
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        reach = comp[comp["variant"].astype(str).str.contains("reachability", na=False)]
        if len(reach) and reach["success_delta"].max() >= 0.05:
            decisions.append("GO_REACHABILITY_CONFIRM")
            evidence.append(f"Reachability paired delta max={reach['success_delta'].max():.3f}.")
        elif len(reach) and reach["success_delta"].max() < 0.02:
            reach_overall_hold = True
            evidence.append(f"Reachability paired delta max={reach['success_delta'].max():.3f}, below 2pp.")
    fs_path = root / "stage22r_failure_slices.csv"
    if fs_path.exists():
        fs = pd.read_csv(fs_path)
        hard = fs[(fs["variant"].astype(str).str.contains("reachability", na=False)) & (fs["success_delta_vs_shortest"] >= 0.05)]
        if len(hard):
            if "GO_REACHABILITY_CONFIRM" not in decisions:
                decisions.append("GO_REACHABILITY_CONFIRM")
            evidence.append(f"Reachability has {len(hard)} hard-slice grouped rows with >=5pp gain.")
    edge_path = root / "stage22r_path_edge_overlap.csv"
    if edge_path.exists():
        ov = pd.read_csv(edge_path)
        reach_ov = ov[ov["variant"].astype(str).str.contains("reachability", na=False)]
        if len(reach_ov):
            mean_ov = float(reach_ov["edge_overlap_vs_shortest"].mean())
            evidence.append(f"Mean reachability edge overlap vs shortest={mean_ov:.3f}.")
            if mean_ov > 0.9 and "GO_REACHABILITY_CONFIRM" not in decisions:
                decisions.append("REPAIR_SCORING")
    if reach_overall_hold and "GO_REACHABILITY_CONFIRM" not in decisions and "REPAIR_SCORING" not in decisions:
        decisions.append("HOLD_REACHABILITY")
    bf_path = root / "stage22r_boundary_feasibility.csv"
    if bf_path.exists():
        bf = pd.read_csv(bf_path)
        boundary = bf[bf["variant"].astype(str).str.contains("boundary", na=False)]
        reject_rate = float((boundary["budget_reject_count"] > 0).mean()) if len(boundary) else 1.0
        evidence.append(f"Boundary budget reject rate={reject_rate:.3f}.")
        if reject_rate >= 0.5:
            decisions.append("HOLD_BOUNDARY")
        else:
            decisions.append("GO_BOUNDARY_REENTRY")
    if "GO_REACHABILITY_CONFIRM" not in decisions and "HOLD_REACHABILITY" not in decisions and "REPAIR_SCORING" not in decisions:
        decisions.append("HOLD_REACHABILITY")
    if "HOLD_BOUNDARY" not in decisions and "GO_BOUNDARY_REENTRY" not in decisions:
        decisions.append("HOLD_BOUNDARY")
    decisions = list(dict.fromkeys(decisions))
    lines = ["# Stage22R Decisions", "", "## Decisions"]
    for d in decisions:
        lines.append(f"- {d}")
    lines.append("")
    lines.append("## Evidence")
    for e in evidence:
        lines.append(f"- {e}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
