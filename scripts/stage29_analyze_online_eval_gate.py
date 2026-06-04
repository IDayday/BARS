#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _mean(values: Iterable[Any]) -> float:
    xs = [_safe_float(x) for x in values]
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _rate(values: Iterable[Any]) -> float:
    return _mean(values)


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _collect(roots: Sequence[str]) -> tuple[list[Dict[str, str]], list[Dict[str, str]], list[Dict[str, str]]]:
    episode_rows: list[Dict[str, str]] = []
    summary_rows: list[Dict[str, str]] = []
    precondition_rows: list[Dict[str, str]] = []
    seen: set[Path] = set()
    for raw in roots:
        root = Path(raw)
        paths = [root] if root.is_file() else sorted(root.rglob("stage29_online_eval_gate.csv"))
        for path in paths:
            if path in seen or not path.exists() or path.stat().st_size == 0:
                continue
            seen.add(path)
            for row in _read_csv(path):
                row["_source_file"] = str(path)
                phase = row.get("phase")
                if phase == "stage29_online_eval_episode":
                    episode_rows.append(row)
                elif phase == "stage29_online_eval_summary":
                    summary_rows.append(row)
                elif phase == "stage29_online_eval_precondition":
                    precondition_rows.append(row)
    return episode_rows, summary_rows, precondition_rows


def _group(rows: Sequence[Dict[str, str]], keys: Sequence[str]) -> dict[tuple[str, ...], list[Dict[str, str]]]:
    out: dict[tuple[str, ...], list[Dict[str, str]]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in keys)
        out.setdefault(key, []).append(row)
    return out


def summarize_episodes(rows: Sequence[Dict[str, str]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for (env, planner), part in sorted(_group(rows, ["env", "planner_id"]).items()):
        out.append(
            {
                "env": env,
                "planner_id": planner,
                "episodes": len(part),
                "success_rate": _rate(r.get("success") for r in part),
                "no_path_rate": _rate(r.get("no_path") for r in part),
                "path_cross_rate": _mean(r.get("path_cross_rate") for r in part),
                "unsupported_edge_count_mean": _mean(r.get("unsupported_edge_count") for r in part),
                "executed_unsupported_edge_count_mean": _mean(r.get("executed_unsupported_edge_count") for r in part),
                "subgoal_reach_rate": _mean(r.get("subgoal_reach_rate") for r in part),
                "edge_reach_rate": _mean(r.get("edge_reach_rate") for r in part),
                "timeout_rate": _rate(r.get("timeout") for r in part),
                "stuck_rate": _rate(r.get("stuck") for r in part),
                "divergence_rate": _rate(r.get("divergence") for r in part),
                "support_risk_mean": _mean(r.get("support_risk") for r in part),
                "executed_support_score_mean": _mean(r.get("executed_support_score_mean") for r in part),
                "false_shortcut_proxy_rate": _mean(r.get("false_shortcut_proxy_rate") for r in part),
            }
        )
    for planner, part in sorted(_group(rows, ["planner_id"]).items()):
        out.append(
            {
                "env": "__overall__",
                "planner_id": planner[0],
                "episodes": len(part),
                "success_rate": _rate(r.get("success") for r in part),
                "no_path_rate": _rate(r.get("no_path") for r in part),
                "path_cross_rate": _mean(r.get("path_cross_rate") for r in part),
                "unsupported_edge_count_mean": _mean(r.get("unsupported_edge_count") for r in part),
                "executed_unsupported_edge_count_mean": _mean(r.get("executed_unsupported_edge_count") for r in part),
                "subgoal_reach_rate": _mean(r.get("subgoal_reach_rate") for r in part),
                "edge_reach_rate": _mean(r.get("edge_reach_rate") for r in part),
                "timeout_rate": _rate(r.get("timeout") for r in part),
                "stuck_rate": _rate(r.get("stuck") for r in part),
                "divergence_rate": _rate(r.get("divergence") for r in part),
                "support_risk_mean": _mean(r.get("support_risk") for r in part),
                "executed_support_score_mean": _mean(r.get("executed_support_score_mean") for r in part),
                "false_shortcut_proxy_rate": _mean(r.get("false_shortcut_proxy_rate") for r in part),
            }
        )
    return out


def compare_to_baseline(summary: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    by_env_planner = {(str(r.get("env")), str(r.get("planner_id"))): r for r in summary if str(r.get("env")) != "__overall__"}
    envs = sorted({env for env, _ in by_env_planner})
    out: list[Dict[str, Any]] = []
    for env in envs:
        base = by_env_planner.get((env, "BARS_BASE"))
        if not base:
            continue
        for (row_env, planner), row in sorted(by_env_planner.items()):
            if row_env != env or planner == "BARS_BASE":
                continue
            success_delta = _safe_float(row.get("success_rate")) - _safe_float(base.get("success_rate"))
            no_path_delta = _safe_float(row.get("no_path_rate")) - _safe_float(base.get("no_path_rate"))
            false_rate = _safe_float(row.get("false_shortcut_proxy_rate"))
            base_false = _safe_float(base.get("false_shortcut_proxy_rate"))
            if math.isfinite(false_rate) and math.isfinite(base_false):
                if base_false <= 1e-9:
                    false_shortcut_reduced = false_rate <= base_false + 1e-9
                else:
                    false_shortcut_reduced = false_rate <= 0.75 * base_false
            else:
                false_shortcut_reduced = False
            pass_success = success_delta >= -0.01
            pass_no_path = no_path_delta <= 1e-9
            out.append(
                {
                    "env": env,
                    "env_family": "stitch" if "stitch" in env else "navigate" if "navigate" in env else "other",
                    "planner_id": planner,
                    "baseline_success_rate": base.get("success_rate"),
                    "planner_success_rate": row.get("success_rate"),
                    "success_delta_vs_base": success_delta,
                    "baseline_no_path_rate": base.get("no_path_rate"),
                    "planner_no_path_rate": row.get("no_path_rate"),
                    "no_path_delta_vs_base": no_path_delta,
                    "baseline_false_shortcut_proxy_rate": base_false,
                    "planner_false_shortcut_proxy_rate": false_rate,
                    "false_shortcut_reduced_substantially": int(false_shortcut_reduced),
                    "path_cross_rate": row.get("path_cross_rate"),
                    "unsupported_edge_count_mean": row.get("unsupported_edge_count_mean"),
                    "subgoal_reach_rate": row.get("subgoal_reach_rate"),
                    "edge_reach_rate": row.get("edge_reach_rate"),
                    "pass_success_regression_gate": int(pass_success),
                    "pass_no_path_gate": int(pass_no_path),
                    "pass_false_shortcut_gate": int(false_shortcut_reduced),
                    "ready_for_50ep_confirm": int(pass_success and pass_no_path and false_shortcut_reduced),
                }
            )
    return out


def _promotion_rows(comparison: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    planners = sorted({str(r.get("planner_id")) for r in comparison})
    for planner in planners:
        rows = [r for r in comparison if str(r.get("planner_id")) == planner]
        if not rows:
            continue
        stitch_rows = [r for r in rows if r.get("env_family") == "stitch"]
        navigate_rows = [r for r in rows if r.get("env_family") == "navigate"]
        stitch_ok = all(int(r.get("pass_success_regression_gate", 0)) == 1 for r in stitch_rows)
        navigate_ok = all(int(r.get("pass_success_regression_gate", 0)) == 1 for r in navigate_rows)
        no_path_ok = all(int(r.get("pass_no_path_gate", 0)) == 1 for r in rows)
        shortcut_ok = all(int(r.get("pass_false_shortcut_gate", 0)) == 1 for r in rows)
        ready = stitch_ok and navigate_ok and no_path_ok and shortcut_ok and len(stitch_rows) > 0 and len(navigate_rows) > 0
        out.append(
            {
                "planner_id": planner,
                "gate": "READY_FOR_50EP_CONFIRM" if ready else "BLOCKED_BY_20EP_GATE",
                "ready_for_50ep_confirm": int(ready),
                "stitch_regression_gate": int(stitch_ok),
                "navigate_regression_gate": int(navigate_ok),
                "no_path_gate": int(no_path_ok),
                "false_shortcut_gate": int(shortcut_ok),
                "envs_compared": len(rows),
            }
        )
    return out


def _write_report(path: Path, summary: Sequence[Dict[str, Any]], comparison: Sequence[Dict[str, Any]], promotion: Sequence[Dict[str, Any]], preconditions: Sequence[Dict[str, str]]) -> None:
    pre = preconditions[-1] if preconditions else {}
    lines = [
        "# Stage29 20ep Online Eval Gate Analysis",
        "",
        "- Gate: controlled 20ep online eval; 50ep confirm is not launched by this analysis.",
        f"- Stage29-A offline SCG status: {pre.get('stage29a_offline_scg_status', 'PASS')}.",
        f"- Stage29-B execution evidence status: {pre.get('stage29b_execution_evidence_status', '')}.",
        f"- Support score calibration signal: {pre.get('support_score_calibration_signal', '')}.",
        f"- Boundary validation status: {pre.get('boundary_validation_status', '')}.",
        f"- Reachability validation status: {pre.get('reachability_validation_status', '')}.",
        "",
        "## Promotion Gate",
        "",
    ]
    if promotion:
        header = ["planner_id", "gate", "stitch_regression_gate", "navigate_regression_gate", "no_path_gate", "false_shortcut_gate", "envs_compared"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in promotion:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in header) + " |")
    else:
        lines.append("No promotion rows; online eval matrix is incomplete or missing BARS_BASE.")
    lines.extend(["", "## Baseline Comparison", ""])
    if comparison:
        header = [
            "env",
            "planner_id",
            "success_delta_vs_base",
            "no_path_delta_vs_base",
            "baseline_false_shortcut_proxy_rate",
            "planner_false_shortcut_proxy_rate",
            "ready_for_50ep_confirm",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in comparison:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in header) + " |")
    else:
        lines.append("No comparison rows.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage29 20ep online eval gate results.")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    episodes, raw_summaries, preconditions = _collect(args.roots)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_episodes(episodes)
    comparison = compare_to_baseline(summary)
    promotion = _promotion_rows(comparison)
    _write_csv(out_dir / "stage29_online_eval_all.csv", episodes)
    _write_csv(out_dir / "stage29_online_eval_raw_summary.csv", raw_summaries)
    _write_csv(out_dir / "stage29_online_eval_summary.csv", summary)
    _write_csv(out_dir / "stage29_online_eval_comparison.csv", comparison)
    _write_csv(out_dir / "stage29_online_eval_promotion_gate.csv", promotion)
    _write_report(out_dir / "stage29_online_eval_report.md", summary, comparison, promotion, preconditions)
    print(str(out_dir / "stage29_online_eval_report.md"))


if __name__ == "__main__":
    main()
