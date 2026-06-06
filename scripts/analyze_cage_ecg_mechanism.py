#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Stage32-34 CAGE mechanism laws for ECG design.")
    parser.add_argument("--stage33_analysis", required=True)
    parser.add_argument("--stage34_analysis", required=True)
    parser.add_argument("--stage34_summary", required=True)
    parser.add_argument("--contract_metrics", required=True)
    parser.add_argument("--split_summary", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
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


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def corr(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[float | None, int]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = numeric(row.get(x_key))
        y = numeric(row.get(y_key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if len(xs) < 3:
        return None, len(xs)
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None, len(xs)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy)), len(xs)


def add_law(rows: list[dict[str, Any]], name: str, status: str, evidence: str, value: Any = None, n: int | None = None) -> None:
    rows.append({"law": name, "status": status, "value": value, "n": n, "evidence": evidence})


def main() -> int:
    args = build_parser().parse_args()
    stage33 = load_csv(args.stage33_analysis)
    stage34 = load_csv(args.stage34_analysis)
    stage34_summary = load_csv(args.stage34_summary)
    metrics = load_json(args.contract_metrics)
    split_summary = load_json(args.split_summary)
    combined = [*stage33, *stage34]
    laws: list[dict[str, Any]] = []

    for key, label in [
        ("segment_target_reach_rate_mean", "segment_target_reach_rate vs success"),
        ("mean_segment_progress_mean", "mean_segment_progress vs success"),
        ("final_goal_on_rate_mean", "final_goal_on_rate vs success"),
        ("stall_count_mean", "stall_count vs success"),
        ("intervention_rate_mean", "intervention_rate vs success"),
        ("source_gas_rate_mean", "source_gas_rate vs success"),
        ("source_cage_rate_mean", "source_cage_rate vs success"),
        ("source_committed_rate_mean", "source_committed_rate vs success"),
        ("committed_usage_rate_mean", "committed_usage_rate vs success"),
    ]:
        value, n = corr(combined, key, "success_rate_mean")
        status = "INCONCLUSIVE" if n < 10 or value is None else "OBSERVED"
        add_law(laws, label, status, f"Pearson correlation over compact deployment rows; n={n}.", value, n)

    safe_loop_rows = []
    for row in combined:
        reach = numeric(row.get("segment_target_reach_rate_mean"))
        stall = numeric(row.get("stall_count_mean"))
        success = numeric(row.get("success_rate_mean"))
        final_on = numeric(row.get("final_goal_on_rate_mean"))
        if reach is not None and stall is not None and success is not None:
            is_loop = bool(reach >= 0.30 and stall >= 15.0 and success <= 0.50 and (final_on or 0.0) <= 0.70)
            if is_loop:
                safe_loop_rows.append(row)
    add_law(
        laws,
        "local_safe_loop proxy",
        "OBSERVED" if safe_loop_rows else "INCONCLUSIVE",
        "Proxy: segment reach >=0.30, stall >=15, success <=0.50, final_goal_on_rate <=0.70.",
        len(safe_loop_rows),
        len(combined),
    )

    add_law(
        laws,
        "local executability is insufficient",
        "OBSERVED",
        "Stage33 rank had high segment reach relative to trace-only but lower success; Stage34 intervention reduced committed use yet still failed success safety.",
        None,
        len(combined),
    )
    add_law(
        laws,
        "intervention is insufficient",
        "OBSERVED",
        "Stage34 contract_intervene improved over contract_rank but remained below GAS on both AntMaze envs.",
        None,
        len(stage34),
    )
    add_law(
        laws,
        "need contract graph and policy alignment",
        "OBSERVED",
        "GP0/CLP1 show q_train support alone is insufficient; closed-loop contractibility and final-goal/farther path target quality are bottlenecks.",
        None,
        len(stage34),
    )

    test_examples = metrics.get("num_examples") or metrics.get("num_eval_examples")
    if test_examples is None:
        for key, value in metrics.items():
            if key.endswith("num_examples"):
                test_examples = value
                break
    add_law(
        laws,
        "contract model evidence strength",
        "INCONCLUSIVE" if (numeric(test_examples) or 0) < 500 else "OBSERVED",
        "Held-out contract model was useful for smoke but test feature count is small; no SOTA claim.",
        test_examples,
        None,
    )
    add_law(
        laws,
        "split coverage",
        split_summary.get("status", "NA"),
        "Contract split summary is included to preserve dataset audit context.",
        split_summary.get("total_examples"),
        None,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = ["law", "status", "value", "n", "evidence"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(laws)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"laws": laws, "num_rows": len(combined)}, indent=2, sort_keys=True), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown(laws), encoding="utf-8")
    print(json.dumps({"out_csv": str(out_csv), "out_json": str(out_json), "out_md": str(out_md), "num_laws": len(laws)}, sort_keys=True))
    return 0


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage35 CAGE-ECG Mechanism Laws",
        "",
        "本报告只给出机制规律和离线证据，不声明统计显著性。样本数过少处标记 INCONCLUSIVE。",
        "",
        "| law | status | value | n | evidence |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['law']} | {row['status']} | {_fmt(row.get('value'))} | {_fmt(row.get('n'))} | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "Stage32-34 的失败演化说明：只做在线 gate/rank/intervention 不足以解决图路径、执行合同和任务推进之间的不一致。下一步需要显式构建执行 funnel node、合同 edge、边界兼容 contract，并用这些对象驱动离线合同路径规划和图诱导策略对齐数据集。",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
