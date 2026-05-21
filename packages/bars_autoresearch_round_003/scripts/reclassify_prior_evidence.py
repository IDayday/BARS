#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "stage",
    "run_root",
    "env",
    "algorithm_or_variant",
    "condition",
    "fallback_mode",
    "train_steps",
    "official_artifact_used",
    "public_target_available",
    "baseline_certification_status",
    "adapter_certification_status",
    "evidence_class",
    "allowed_claim_level",
    "downgrade_reason",
    "report_file",
    "row_count",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def stage_from_name(path: Path) -> str:
    name = path.name
    if name.startswith("round_001"):
        return "Round001"
    for prefix in ["stage25", "stage24", "stage23", "stage22r", "stage22", "stage21", "stage20", "stage19"]:
        if name.startswith(prefix):
            return prefix.replace("stage", "Stage")
    return "unknown"


def infer_env(row: dict[str, str], path: Path) -> str:
    for key in ["env", "env_name", "environment"]:
        if row.get(key):
            return row[key]
    return "mixed_or_not_recorded"


def infer_variant(row: dict[str, str], path: Path) -> str:
    for key in ["variant", "algorithm", "method", "planner", "route", "algorithm_or_variant"]:
        if row.get(key):
            return row[key]
    return path.stem


def infer_fallback(row: dict[str, str], path: Path) -> str:
    for key in ["fallback_mode", "fallback", "fallback_used"]:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    name = path.name.lower()
    if "fallback" in name:
        return "fallback_protocol"
    return "none"


def infer_run_root(row: dict[str, str], path: Path) -> str:
    for key in ["eval_csv", "eval_path", "path", "run_root", "log"]:
        value = row.get(key)
        if value:
            return value
    return str(path)


def infer_train_steps(env: str, variant: str, path: Path) -> str:
    text = f"{path.name} {variant}".lower()
    if "smoke" in text or "quick" in text or "medium50" in text or "medium100" in text:
        return "reduced_or_smoke"
    if env in {"antmaze-medium-stitch-v0", "antmaze-medium-navigate-v0"}:
        return "100000"
    return "unknown"


def load_card_status(cards_path: Path) -> dict[str, dict[str, Any]]:
    cards = read_jsonl(cards_path)
    out: dict[str, dict[str, Any]] = {}
    for card in cards:
        if card.get("algorithm") == "GAS":
            out[str(card.get("env"))] = card
    return out


def public_target_available(env: str, card_status: dict[str, dict[str, Any]]) -> bool:
    if env in card_status:
        return True
    if env.startswith("antmaze-") and env.endswith("-v0"):
        return True
    return False


def classify(
    stage: str,
    env: str,
    variant: str,
    fallback: str,
    train_steps: str,
    card_status: dict[str, dict[str, Any]],
    adapter_status: str,
    report_file: str,
) -> tuple[str, str, str, str, str]:
    card = card_status.get(env, {})
    baseline_status = str(card.get("certification_status") or "UNCERTIFIED_OR_NOT_ASSESSED")
    official_used = str((card.get("we_used") or {}).get("source") == "official_checkpoint")
    reduced = train_steps in {"reduced_or_smoke", "100000"}
    fallback_lower = str(fallback).lower()
    fallback_protocol = fallback_lower not in {"", "none", "0", "false", "nan"}
    reasons: list[str] = []
    if reduced:
        reasons.append("reduced training or smoke/protocol budget")
    if fallback_protocol:
        reasons.append("direct/progress fallback is fallback-protocol evidence, not planner evidence")
    if baseline_status != "PASS_BASELINE_CERTIFICATION":
        reasons.append(f"baseline gate is {baseline_status}")
    if adapter_status != "PASS_ADAPTER_CERTIFICATION":
        reasons.append(f"adapter gate is {adapter_status}")
    if "adapter_protocol_repair" in report_file and "original" in variant.lower():
        reasons.append("original BARS adapter gap exceeded 2pp in Stage23 repair report")
    if reduced:
        evidence = "E0_SMOKE_ONLY"
        allowed = "SMOKE_ONLY"
    elif fallback_protocol:
        evidence = "FALLBACK_PROTOCOL_ONLY"
        allowed = "PROTOCOL_DEBUG_ONLY"
    elif baseline_status != "PASS_BASELINE_CERTIFICATION" or adapter_status != "PASS_ADAPTER_CERTIFICATION":
        evidence = "PROTOCOL_DEBUG_ONLY"
        allowed = "PROTOCOL_DEBUG_ONLY"
    else:
        evidence = "E3_SAME_BACKBONE_METHOD_COMPARISON"
        allowed = "SAME_BACKBONE_MECHANISM"
    return baseline_status, official_used, evidence, allowed, "; ".join(dict.fromkeys(reasons))


def grouped_rows(path: Path, card_status: dict[str, dict[str, Any]], adapter_status: str) -> list[dict[str, Any]]:
    raw = read_csv(path)
    if not raw:
        return []
    groups: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    run_roots: dict[tuple[str, str, str, str, str], str] = {}
    for row in raw:
        env = infer_env(row, path)
        variant = infer_variant(row, path)
        fallback = infer_fallback(row, path)
        train_steps = infer_train_steps(env, variant, path)
        condition = path.stem
        key = (env, variant, fallback, train_steps, condition)
        groups[key] += 1
        run_roots.setdefault(key, infer_run_root(row, path))
    rows: list[dict[str, Any]] = []
    stage = stage_from_name(path)
    for (env, variant, fallback, train_steps, condition), count in sorted(groups.items()):
        baseline_status, official_used, evidence, allowed, reason = classify(
            stage, env, variant, fallback, train_steps, card_status, adapter_status, str(path)
        )
        rows.append(
            {
                "stage": stage,
                "run_root": run_roots[(env, variant, fallback, train_steps, condition)],
                "env": env,
                "algorithm_or_variant": variant,
                "condition": condition,
                "fallback_mode": fallback,
                "train_steps": train_steps,
                "official_artifact_used": official_used,
                "public_target_available": public_target_available(env, card_status),
                "baseline_certification_status": baseline_status,
                "adapter_certification_status": adapter_status,
                "evidence_class": evidence,
                "allowed_claim_level": allowed,
                "downgrade_reason": reason,
                "report_file": str(path),
                "row_count": count,
            }
        )
    return rows


def adapter_repair_rows(path: Path, card_status: dict[str, dict[str, Any]], adapter_status: str) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        env = raw.get("env", "")
        for label, col in [("original_bars_adapter", "original_minus_official_pp"), ("official_control_adapter", "official_control_minus_official_pp")]:
            gap = raw.get(col, "")
            baseline_status, official_used, evidence, allowed, reason = classify(
                "Stage23", env, label, "none", "100000", card_status, adapter_status, str(path)
            )
            reason = f"{reason}; adapter_gap_pp={gap}"
            rows.append(
                {
                    "stage": "Stage23",
                    "run_root": raw.get("eval_csv", str(path)),
                    "env": env,
                    "algorithm_or_variant": label,
                    "condition": "adapter_protocol_repair",
                    "fallback_mode": "none",
                    "train_steps": "100000",
                    "official_artifact_used": official_used,
                    "public_target_available": public_target_available(env, card_status),
                    "baseline_certification_status": baseline_status,
                    "adapter_certification_status": adapter_status,
                    "evidence_class": evidence,
                    "allowed_claim_level": allowed,
                    "downgrade_reason": reason,
                    "report_file": str(path),
                    "row_count": 1,
                }
            )
    return rows


def round001_row(stage_reports: Path, adapter_status: str) -> dict[str, Any]:
    return {
        "stage": "Round001",
        "run_root": "rounds/round_001",
        "env": "antmaze-medium-stitch-v0|antmaze-medium-navigate-v0",
        "algorithm_or_variant": "failure_atlas_label_quality",
        "condition": "round_001_autoresearch",
        "fallback_mode": "none_only",
        "train_steps": "100000",
        "official_artifact_used": False,
        "public_target_available": True,
        "baseline_certification_status": "FAIL_UNDERTRAINED_BASELINE",
        "adapter_certification_status": adapter_status,
        "evidence_class": "PROTOCOL_DEBUG_ONLY",
        "allowed_claim_level": "PROTOCOL_DEBUG_ONLY",
        "downgrade_reason": "Round001 label integrity may remain useful as protocol bookkeeping, but baseline and adapter certification were missing",
        "report_file": str(stage_reports / "round_001_gate_status.json"),
        "row_count": 1,
    }


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(str(row["evidence_class"]), str(row["allowed_claim_level"]))] += int(row.get("row_count") or 1)
    lines = [
        "# Round 002 Prior Evidence Reclassification",
        "",
        "Baseline certification failed for the medium GAS backbone, so no Stage19-Round001 result is promoted to a causal failure-mode or same-backbone mechanism claim.",
        "",
        "## Summary",
        "",
        "| evidence class | allowed claim level | grouped rows |",
        "| --- | --- | ---: |",
    ]
    for (evidence, allowed), count in sorted(counts.items()):
        lines.append(f"| {evidence} | {allowed} | {count} |")
    lines.extend(
        [
            "",
            "## Downgrade Rules Applied",
            "",
            "- Reduced training or smoke/protocol budgets are E0_SMOKE_ONLY.",
            "- Uncertified baseline rows are PROTOCOL_DEBUG_ONLY.",
            "- Direct-goal/progress fallback rows are fallback-protocol evidence, not planner evidence.",
            "- Same-backbone mechanism claims are blocked until both baseline and adapter certification pass.",
            "",
            "## Explicit Decision",
            "",
            "All Stage19-Round001 BARS results are downgraded to smoke/protocol evidence for scientific interpretation purposes.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cards", required=True)
    parser.add_argument("--stage-reports", default="reports")
    parser.add_argument("--official-vs-adapter", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    card_status = load_card_status(Path(args.baseline_cards))
    adapter_status = "SKIP_NO_OFFICIAL_EVAL"
    if args.official_vs_adapter:
        comp = read_csv(Path(args.official_vs_adapter))
        statuses = {r.get("adapter_certification_status") for r in comp}
        if statuses == {"PASS_ADAPTER_CERTIFICATION"}:
            adapter_status = "PASS_ADAPTER_CERTIFICATION"
        elif "FAIL_ADAPTER_CERTIFICATION" in statuses:
            adapter_status = "FAIL_ADAPTER_CERTIFICATION"
    stage_reports = Path(args.stage_reports)
    patterns = [
        "stage19*.csv",
        "stage20*.csv",
        "stage21*.csv",
        "stage22*.csv",
        "stage22r*.csv",
        "stage23*.csv",
        "stage24*.csv",
        "stage25*.csv",
    ]
    rows: list[dict[str, Any]] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(stage_reports.glob(pattern)):
            if path.name.startswith("round_002"):
                continue
            if path in seen:
                continue
            seen.add(path)
            if path.name == "stage23_adapter_protocol_repair.csv":
                rows.extend(adapter_repair_rows(path, card_status, adapter_status))
            else:
                rows.extend(grouped_rows(path, card_status, adapter_status))
    if (stage_reports / "round_001_gate_status.json").exists():
        rows.append(round001_row(stage_reports, adapter_status))
    rows = [row for row in rows if row["stage"] != "unknown"]
    write_csv(Path(args.out), rows)
    write_md(Path(args.md_out), rows)
    print(json.dumps({"rows": len(rows), "out": args.out}, sort_keys=True))


if __name__ == "__main__":
    main()
