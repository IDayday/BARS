#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from cage_gp0_common import iter_jsonl, pearson, summarize_numeric, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CLP1 branchable contract probes and oracle headroom.")
    parser.add_argument("--probe_files", nargs="+", required=True)
    parser.add_argument("--segment_files", nargs="*", default=[])
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--progress_threshold", type=float, default=0.2)
    return parser.parse_args()


def expand(paths: list[str]) -> list[Path]:
    out = []
    for item in paths:
        if any(ch in item for ch in "*?[]"):
            out.extend(sorted(Path().glob(item)))
        else:
            out.append(Path(item))
    return [p for p in out if p.exists()]


def load_probe_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in expand(paths):
        for row in iter_jsonl(path):
            if row.get("record_type") in {"branchable_probe", "closed_loop_probe"}:
                row = dict(row)
                row["probe_file"] = str(path)
                rows.append(row)
    return rows


def hit_rate(rows):
    valid = [r for r in rows if not r.get("failure_reason")]
    if not valid:
        return None
    return float(np.mean([bool(r.get("hit")) for r in valid]))


def group(rows, keys):
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    return out


def summarize_group(records, keys, vals):
    valid = [r for r in records if not r.get("failure_reason")]
    return {
        **dict(zip(keys, vals)),
        "num_rows": len(records),
        "num_valid": len(valid),
        "hit_rate": hit_rate(records),
        "mean_delta_phi": summarize_numeric(r.get("delta_phi") for r in valid)["mean"],
        "mean_normalized_progress": summarize_numeric(r.get("normalized_progress") for r in valid)["mean"],
        "negative_progress_rate": float(np.mean([bool(r.get("negative_progress", (r.get("normalized_progress") or 0) < 0)) for r in valid])) if valid else None,
    }


def dphi_bin(row):
    v = row.get("d_phi_start", row.get("graph_d_phi"))
    if v is None:
        return "unknown"
    v = float(v)
    if v < 4:
        return "<4"
    if v < 8:
        return "4-8"
    if v < 16:
        return "8-16"
    if v < 32:
        return "16-32"
    return ">=32"


def support_bin(row):
    v = row.get("q_train_support")
    if v is None:
        return "unknown"
    v = float(v)
    if v < 0.25:
        return "<0.25"
    if v < 0.5:
        return "0.25-0.5"
    if v < 0.75:
        return "0.5-0.75"
    return ">=0.75"


def oracle_rows(rows, progress_threshold):
    by_seg_h = group([r for r in rows if not r.get("failure_reason")], ["env_name", "variant_source", "source_segment_id", "horizon"])
    out = []
    for (env_name, variant_source, segment_id, horizon), records in by_seg_h.items():
        originals = [r for r in records if r.get("target_mode") == "original_target"]
        if not originals:
            continue
        original = originals[0]
        best = sorted(records, key=lambda r: (bool(r.get("hit")), float(r.get("normalized_progress") or -1e9)), reverse=True)[0]
        out.append({
            "env_name": env_name,
            "variant_source": variant_source,
            "source_segment_id": segment_id,
            "horizon": horizon,
            "original_hit": bool(original.get("hit")),
            "oracle_hit": bool(best.get("hit")),
            "original_progress": original.get("normalized_progress"),
            "oracle_progress": best.get("normalized_progress"),
            "oracle_target_mode": best.get("target_mode"),
            "oracle_contract_positive": bool(best.get("hit")) or float(best.get("normalized_progress") or -1e9) >= progress_threshold,
        })
    return out


def main() -> None:
    args = parse_args()
    rows = load_probe_rows(args.probe_files)
    table = []
    for keys in (["env_name", "target_mode", "horizon"], ["env_name"], ["target_mode"], ["horizon"], ["env_name", "path_position"], ["env_name", "final_phase"]):
        for vals, records in sorted(group(rows, keys).items()):
            table.append(summarize_group(records, keys, vals))
    for bin_name, fn in [("d_phi_bin", dphi_bin), ("q_train_support_bin", support_bin)]:
        for val, records in sorted(group([{**r, bin_name: fn(r)} for r in rows], [bin_name]).items()):
            table.append(summarize_group(records, [bin_name], val))
    oracles = oracle_rows(rows, args.progress_threshold)
    output = {
        "num_probe_rows": len(rows),
        "num_valid_probe_rows": len([r for r in rows if not r.get("failure_reason")]),
        "table": table,
        "oracle": {
            "num_segments": len(oracles),
            "original_hit_rate": float(np.mean([r["original_hit"] for r in oracles])) if oracles else None,
            "oracle_hit_rate": float(np.mean([r["oracle_hit"] for r in oracles])) if oracles else None,
            "original_progress_mean": summarize_numeric(r.get("original_progress") for r in oracles)["mean"],
            "oracle_progress_mean": summarize_numeric(r.get("oracle_progress") for r in oracles)["mean"],
        },
        "correlations": {
            "hit_vs_d_phi": pearson([float(bool(r.get("hit"))) for r in rows if not r.get("failure_reason")], [r.get("d_phi_start", r.get("graph_d_phi")) for r in rows if not r.get("failure_reason")]),
            "hit_vs_q_train_support": pearson([float(bool(r.get("hit"))) for r in rows if not r.get("failure_reason")], [r.get("q_train_support") for r in rows if not r.get("failure_reason")]),
            "hit_vs_progress": pearson([float(bool(r.get("hit"))) for r in rows if not r.get("failure_reason")], [r.get("normalized_progress") for r in rows if not r.get("failure_reason")]),
        },
        "failure_modes": classify_failure_modes(rows),
    }
    write_json(args.out_json, output)
    write_csv(args.out_csv, table)
    write_md(args.out_md, output)
    print({"out_json": args.out_json, "out_md": args.out_md, "rows": len(rows), "oracle_segments": output["oracle"]["num_segments"]})


def classify_failure_modes(rows):
    counts = {
        "no_contract_target_available": 0,
        "original_target_non_contractive": 0,
        "recovery_target_non_contractive": 0,
        "all_candidates_non_contractive": 0,
    }
    for row in rows:
        if row.get("failure_reason"):
            counts["no_contract_target_available"] += 1
        if row.get("target_mode") == "original_target" and not row.get("hit") and (row.get("normalized_progress") or 0) <= 0:
            counts["original_target_non_contractive"] += 1
        if row.get("target_mode") == "recovery_candidate" and not row.get("hit") and (row.get("normalized_progress") or 0) <= 0:
            counts["recovery_target_non_contractive"] += 1
    by_segment = group([r for r in rows if not r.get("failure_reason")], ["env_name", "variant_source", "source_segment_id", "horizon"])
    for records in by_segment.values():
        if records and all((not r.get("hit") and (r.get("normalized_progress") or 0) <= 0) for r in records):
            counts["all_candidates_non_contractive"] += 1
    return counts


def write_md(path, output):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for row in output["table"] for k in row.keys()})
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# CAGE-CLP1 Contract Oracle Summary\n\n")
        fh.write("## Oracle\n\n")
        for k, v in output["oracle"].items():
            fh.write(f"- `{k}`: {v}\n")
        fh.write("\n## Correlations\n\n")
        for k, v in output["correlations"].items():
            fh.write(f"- `{k}`: {v if v is not None else 'NA'}\n")
        fh.write("\n## Grouped Table\n\n")
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for row in output["table"]:
            vals = []
            for col in cols:
                val = row.get(col)
                vals.append("NA" if val is None else (f"{val:.4f}" if isinstance(val, float) else str(val)))
            fh.write("| " + " | ".join(vals) + " |\n")
        fh.write("\n## Failure Modes\n\n")
        for k, v in output["failure_modes"].items():
            fh.write(f"- `{k}`: {v}\n")


if __name__ == "__main__":
    main()
