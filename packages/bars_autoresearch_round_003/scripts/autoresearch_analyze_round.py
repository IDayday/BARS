#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


STATE_PATH = Path("research_state/bars_research_state.json")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def round_name(n: int) -> str:
    return f"round_{n:03d}"


def analyze_failure_quality(round_num: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rn = round_name(round_num)
    atlas_path = Path(f"reports/{rn}_failure_atlas_all_variants.csv")
    integrity_path = Path(f"reports/{rn}_label_integrity.json")
    integrity = read_json(integrity_path, {})
    atlas = pd.read_csv(atlas_path) if atlas_path.exists() else pd.DataFrame()
    failed = atlas[pd.to_numeric(atlas.get("success", pd.Series(dtype=float)), errors="coerce").fillna(0) < 0.5] if len(atlas) else pd.DataFrame()
    unclassified = float(failed.get("primary_failure_type", pd.Series(dtype=str)).astype(str).eq("UNCLASSIFIED_FAILURE").mean()) if len(failed) else 0.0
    missing = int(integrity.get("missing_primary_failure_type_failed_rows", 0) or 0)
    complete_cells = int(integrity.get("complete_cells", 0) or 0)
    gate = "PASS_FAILURE_LABEL_QUALITY" if missing == 0 and complete_cells > 0 and unclassified <= 0.20 else "FAIL_FAILURE_LABEL_QUALITY"
    variants = sorted(atlas["variant"].dropna().astype(str).unique().tolist()) if len(atlas) and "variant" in atlas else []
    by_variant = (
        atlas.groupby(["variant", "primary_failure_type"], dropna=False)
        .agg(episodes=("success", "count"), success=("success", "mean"))
        .reset_index()
        if len(atlas)
        else pd.DataFrame(columns=["variant", "primary_failure_type", "episodes", "success"])
    )
    metrics = pd.DataFrame(
        [
            {
                "metric": "total_rows",
                "value": int(len(atlas)),
            },
            {
                "metric": "failed_rows",
                "value": int(len(failed)),
            },
            {
                "metric": "missing_primary_failure_type_failed_rows",
                "value": missing,
            },
            {
                "metric": "unclassified_failure_rate",
                "value": unclassified,
            },
            {
                "metric": "complete_cells",
                "value": complete_cells,
            },
            {
                "metric": "variants",
                "value": "|".join(variants),
            },
        ]
    )
    detail = {
        "failure_label_quality": gate,
        "label_integrity": integrity.get("gate", "unknown"),
        "total_rows": int(len(atlas)),
        "failed_rows": int(len(failed)),
        "missing_failed_labels": missing,
        "unclassified_failure_rate": unclassified,
        "complete_cells": complete_cells,
        "variants": variants,
    }
    return metrics, detail | {"by_variant_rows": int(len(by_variant))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, default=0)
    args = p.parse_args()
    state = read_json(STATE_PATH, {})
    round_num = int(args.round or state.get("round", 1) or 1)
    rn = round_name(round_num)
    round_dir = Path("rounds") / rn
    round_dir.mkdir(parents=True, exist_ok=True)
    metrics, detail = analyze_failure_quality(round_num)
    metrics.to_csv(round_dir / "metrics_summary.csv", index=False)
    gate_status = {
        "round": rn,
        "primary_question": "Are all Stage24 variants labeled consistently enough for autonomous decisions?",
        "gates": {
            "failure_label_quality": detail["failure_label_quality"],
            "label_integrity": detail["label_integrity"],
            "planner_evidence_fallback_mode": "none_only",
            "p_bridge": "SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM",
            "integrated": "SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE",
            "boundary": "HOLD_BOUNDARY_DIAGNOSTIC_ONLY",
            "d4rl": "HOLD_D4RL_PROTOCOL_AUDIT",
        },
        "details": detail,
    }
    write_json(round_dir / "gate_status.json", gate_status)
    write_json(Path(f"reports/{rn}_gate_status.json"), gate_status)
    metrics.to_csv(f"reports/{rn}_metrics_summary.csv", index=False)
    reflection = [
        f"# {rn} Reflection",
        "",
        "## Primary Question",
        "Are all Stage24 variants labeled consistently enough for autonomous decisions?",
        "",
        "## Findings",
        f"- Total labeled rows: {detail['total_rows']}",
        f"- Failed rows: {detail['failed_rows']}",
        f"- Missing failed labels: {detail['missing_failed_labels']}",
        f"- Unclassified failure rate: {detail['unclassified_failure_rate']:.4f}",
        f"- Complete cells: {detail['complete_cells']}",
        f"- Label integrity gate: {detail['label_integrity']}",
        f"- Failure label quality gate: {detail['failure_label_quality']}",
        "",
        "## Interpretation",
        "Round 1 is a protocol-repair round. It does not claim a new BARS method. It only determines whether later autonomous decisions can rely on all-variant failure labels.",
        "",
        "## Constraint Check",
        "- Planner evidence remains no-fallback only.",
        "- p_bridge and integrated BARS-v3 remain skipped because oracle headroom has not passed.",
        "- Boundary remains diagnostic-only.",
        "- D4RL remains audit/debug-only.",
    ]
    (round_dir / "reflection.md").write_text("\n".join(reflection) + "\n")
    Path(f"reports/{rn}_reflection.md").write_text("\n".join(reflection) + "\n")
    print(json.dumps({"round": rn, "gate": detail["failure_label_quality"], "label_integrity": detail["label_integrity"]}, sort_keys=True))


if __name__ == "__main__":
    main()
