#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from cage_gp0_common import iter_jsonl, pearson, summarize_numeric, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CAGE-CLP0 closed-loop edge probe JSONL files.")
    parser.add_argument("--probe_files", nargs="+", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_csv", required=True)
    return parser.parse_args()


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for pattern in paths:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in matches:
            if not path.exists():
                continue
            for row in iter_jsonl(path):
                if row.get("record_type") in ("closed_loop_probe", "probe_failure"):
                    row = dict(row)
                    row["probe_file"] = str(path)
                    rows.append(row)
    return rows


def dphi_bin(value: Any) -> str:
    if value is None:
        return "unknown"
    v = float(value)
    if v < 4:
        return "<4"
    if v < 8:
        return "4-8"
    if v < 16:
        return "8-16"
    if v < 32:
        return "16-32"
    return ">=32"


def support_bin(value: Any) -> str:
    if value is None:
        return "unknown"
    v = float(value)
    if v < 0.25:
        return "<0.25"
    if v < 0.5:
        return "0.25-0.5"
    if v < 0.75:
        return "0.5-0.75"
    return ">=0.75"


def summarize_group(rows: list[dict[str, Any]], group_name: str, group_value: str) -> dict[str, Any]:
    valid = [r for r in rows if not r.get("failure_reason")]
    return {
        "group": group_name,
        "value": group_value,
        "num_rows": len(rows),
        "num_valid": len(valid),
        "hit_rate": float(np.mean([bool(r.get("hit")) for r in valid])) if valid else None,
        "mean_delta_phi": summarize_numeric(r.get("delta_phi") for r in valid)["mean"],
        "mean_normalized_progress": summarize_numeric(r.get("normalized_progress") for r in valid)["mean"],
        "negative_progress_rate": float(np.mean([(r.get("normalized_progress") or 0) < 0 for r in valid])) if valid else None,
        "action_norm_max_mean": summarize_numeric(r.get("action_norm_max") for r in valid)["mean"],
    }


def group_by(rows: list[dict[str, Any]], key_fn) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(key_fn(row)), []).append(row)
    return groups


def main() -> None:
    args = parse_args()
    rows = load_rows(args.probe_files)
    valid = [r for r in rows if not r.get("failure_reason")]
    table = []
    for name, fn in [
        ("env", lambda r: r.get("env_name", "unknown")),
        ("pair_source", lambda r: r.get("pair_source", "unknown")),
        ("path_position", lambda r: r.get("path_position_bucket", r.get("path_position", "unknown"))),
        ("final_phase", lambda r: "final" if r.get("final_phase") else "non_final"),
        ("recovery_candidate", lambda r: "recovery" if r.get("recovery_candidate") else "non_recovery"),
        ("d_phi_bin", lambda r: dphi_bin(r.get("d_phi_start"))),
        ("q_train_support_bin", lambda r: support_bin(r.get("q_train_support"))),
    ]:
        for value, records in sorted(group_by(rows, fn).items()):
            table.append(summarize_group(records, name, value))
    output = {
        "num_rows": len(rows),
        "num_valid": len(valid),
        "overall": summarize_group(rows, "overall", "overall"),
        "correlations": {
            "hit_vs_d_phi": pearson([float(bool(r.get("hit"))) for r in valid], [r.get("d_phi_start") for r in valid]),
            "hit_vs_q_train_support": pearson([float(bool(r.get("hit"))) for r in valid], [r.get("q_train_support") for r in valid]),
            "hit_vs_normalized_progress": pearson([float(bool(r.get("hit"))) for r in valid], [r.get("normalized_progress") for r in valid]),
        },
        "table": table,
        "notes": [
            "R_pi is estimated as hit_rate.",
            "Delta_pi is mean_delta_phi.",
            "q_train support bins are populated only when probe rows carry q_train_support.",
        ],
    }
    write_json(args.out_json, output)
    write_csv(args.out_csv, table)
    write_md(args.out_md, output)
    print({"out_json": args.out_json, "out_md": args.out_md, "out_csv": args.out_csv, "rows": len(rows), "valid": len(valid)})


def write_md(path: str | Path, output: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["group", "value", "num_rows", "num_valid", "hit_rate", "mean_delta_phi", "mean_normalized_progress", "negative_progress_rate"]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# CAGE-CLP0 Closed-Loop Probe Summary\n\n")
        fh.write("## Overall\n\n")
        for key, value in output["overall"].items():
            fh.write(f"- `{key}`: {value}\n")
        fh.write("\n## Correlations\n\n")
        for key, value in output["correlations"].items():
            fh.write(f"- `{key}`: {value if value is not None else 'NA'}\n")
        fh.write("\n## Grouped Metrics\n\n")
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for row in output["table"]:
            vals = []
            for col in cols:
                val = row.get(col)
                vals.append("NA" if val is None else (f"{val:.4f}" if isinstance(val, float) else str(val)))
            fh.write("| " + " | ".join(vals) + " |\n")
        fh.write("\n## Notes\n\n")
        for note in output["notes"]:
            fh.write(f"- {note}\n")


if __name__ == "__main__":
    main()
