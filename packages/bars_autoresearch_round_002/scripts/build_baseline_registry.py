#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bars.external.gas_artifacts import resolve_gas_artifacts
from fetch_public_baseline_targets import (
    ALGORITHMS,
    OFFICIAL_ARTIFACT_URL,
    OFFICIAL_CODE_URL,
    PUBLIC_SOURCE_URL,
    TARGETS_PP,
    env_to_slug,
    gas_required_hyperparameters,
    gas_required_train_steps,
    public_eval_protocol,
    target_rows,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def checkpoint_step(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"params_(\d+)\.pkl", Path(path).name)
    return int(match.group(1)) if match else None


def used_artifacts(env: str, seed: int, artifact_root: str) -> dict[str, Any]:
    artifacts = resolve_gas_artifacts(env, seed, artifact_root)
    manifest = read_json(artifacts.root / "manifest.json", {})
    source = str(manifest.get("source") or "unknown")
    policy = str(artifacts.policy_checkpoint) if artifacts.policy_checkpoint else None
    tdr = str(artifacts.tdr_checkpoint) if artifacts.tdr_checkpoint else None
    graph = str(artifacts.keygraph) if artifacts.keygraph else None
    step = checkpoint_step(policy)
    if artifacts.complete and source == "huggingface":
        used_source = "official_checkpoint"
    elif artifacts.complete and step is not None and step >= gas_required_train_steps(env):
        used_source = "full_budget_train"
    elif artifacts.complete:
        used_source = "reduced_train"
    else:
        used_source = "unknown"
    return {
        "source": used_source,
        "manifest_source": source,
        "train_steps": step,
        "checkpoint_path": policy,
        "tdr_path": tdr,
        "graph_path": graph,
        "policy_path": policy,
        "complete": artifacts.complete,
        "manifest": str(artifacts.root / "manifest.json") if (artifacts.root / "manifest.json").exists() else None,
    }


def gas_initial_status(env: str, exact_public: bool, official_available: bool, used: dict[str, Any]) -> str:
    if not exact_public:
        return "HOLD_NO_EXACT_PUBLIC_TARGET"
    if used["source"] == "official_checkpoint":
        return "READY_FOR_OFFICIAL_EVAL"
    if used["source"] == "full_budget_train":
        return "READY_FOR_FULL_BUDGET_EVAL"
    if used["source"] == "reduced_train":
        return "FAIL_UNDERTRAINED_BASELINE"
    if not official_available:
        return "SKIP_ARTIFACT_UNAVAILABLE"
    return "SKIP_ARTIFACT_UNAVAILABLE"


def card_for(row: dict[str, Any], artifact_root: str, seed: int) -> dict[str, Any]:
    env = str(row["env"])
    algorithm = str(row["algorithm"])
    exact = bool(row["exact_public_target_available"])
    official_available = bool(row["official_checkpoint_available"]) if algorithm == "GAS" else None
    used = used_artifacts(env, seed, artifact_root) if algorithm == "GAS" else {
        "source": "unknown",
        "train_steps": None,
        "checkpoint_path": None,
        "tdr_path": None,
        "graph_path": None,
        "policy_path": None,
    }
    status = gas_initial_status(env, exact, bool(official_available), used) if algorithm == "GAS" else "UNASSESSED_REFERENCE_METHOD"
    return {
        "env": env,
        "suite": "ogbench",
        "algorithm": algorithm,
        "baseline_role": "primary_strong_backbone" if algorithm == "GAS" else "public_reference_method",
        "exact_public_target_available": exact,
        "public_source": row["public_source"],
        "public_source_url": row["public_source_url"],
        "public_metric": row["public_metric"],
        "public_mean": float(row["public_mean_pp"]) / 100.0,
        "public_std": float(row["public_std_pp"]) / 100.0,
        "public_mean_pp": float(row["public_mean_pp"]),
        "public_std_pp": float(row["public_std_pp"]),
        "public_eval_protocol": public_eval_protocol(env),
        "required_train_steps": gas_required_train_steps(env) if algorithm == "GAS" else None,
        "required_batch_size": 1024 if algorithm == "GAS" else None,
        "required_hyperparameters": gas_required_hyperparameters(env) if algorithm == "GAS" else {},
        "official_checkpoint_available": official_available,
        "official_tdr_available": official_available,
        "official_graph_available": official_available,
        "official_artifact_source_url": OFFICIAL_ARTIFACT_URL if algorithm == "GAS" else "",
        "we_used": used,
        "official_eval_score": None,
        "bars_adapter_score": None,
        "adapter_gap_pp": None,
        "lower_bound": float(row["lower_bound_pp"]) / 100.0,
        "lower_bound_pp": float(row["lower_bound_pp"]),
        "certification_status": status,
        "evidence_class": "E1_BASELINE_REGISTRY",
    }


def flatten_card(card: dict[str, Any]) -> dict[str, Any]:
    used = card.get("we_used", {}) or {}
    return {
        "env": card["env"],
        "algorithm": card["algorithm"],
        "baseline_role": card["baseline_role"],
        "exact_public_target_available": card["exact_public_target_available"],
        "public_metric": card["public_metric"],
        "public_mean_pp": card["public_mean_pp"],
        "public_std_pp": card["public_std_pp"],
        "lower_bound_pp": card["lower_bound_pp"],
        "required_train_steps": card["required_train_steps"],
        "required_batch_size": card["required_batch_size"],
        "official_checkpoint_available": card["official_checkpoint_available"],
        "official_graph_available": card["official_graph_available"],
        "we_used_source": used.get("source"),
        "we_used_manifest_source": used.get("manifest_source"),
        "we_used_train_steps": used.get("train_steps"),
        "policy_path": used.get("policy_path"),
        "tdr_path": used.get("tdr_path"),
        "graph_path": used.get("graph_path"),
        "certification_status": card["certification_status"],
        "evidence_class": card["evidence_class"],
        "public_source_url": card["public_source_url"],
        "official_artifact_source_url": card.get("official_artifact_source_url", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def safe_name(env: str, algorithm: str) -> str:
    return f"{env}__{algorithm}".replace("/", "_")


def write_lookup_md(path: Path, cards: list[dict[str, Any]]) -> None:
    gas_cards = [c for c in cards if c["algorithm"] == "GAS"]
    exact_missing = [c for c in cards if not c["exact_public_target_available"]]
    unavailable = [c for c in gas_cards if c["certification_status"] == "SKIP_ARTIFACT_UNAVAILABLE"]
    undertrained = [c for c in gas_cards if c["certification_status"] == "FAIL_UNDERTRAINED_BASELINE"]
    lines = [
        "# Round 002 Public Target Lookup",
        "",
        "## Sources",
        f"- Public target table: {PUBLIC_SOURCE_URL}",
        f"- Official GAS code and command templates: {OFFICIAL_CODE_URL}",
        f"- Official GAS checkpoint listing: {OFFICIAL_ARTIFACT_URL}",
        "",
        "## Protocol Extracted",
        "- Metric: normalized return in percentage points.",
        "- Evaluation: five test-time goals, 50 rollouts per goal, averaged over 4 seeds.",
        "- GAS state-task training command: 1,000,000 TDR steps and 1,000,000 policy steps with batch size 1024.",
        "- Certification lower bound: public_mean_pp - max(2 * public_std_pp, 5pp).",
        "",
        "## Exact Target Status",
        f"- exact_public_target_missing_rows: {len(exact_missing)}",
        f"- gas_official_artifact_unavailable_rows: {len(unavailable)}",
        f"- gas_undertrained_local_rows: {len(undertrained)}",
        "",
        "## GAS Rows",
        "",
        "| env | public mean +/- std | lower bound | official artifact | local source | local steps | initial status |",
        "| --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for card in gas_cards:
        used = card["we_used"]
        lines.append(
            "| {env} | {mean:.1f} +/- {std:.1f} | {lb:.1f} | {official} | {source} | {steps} | {status} |".format(
                env=card["env"],
                mean=card["public_mean_pp"],
                std=card["public_std_pp"],
                lb=card["lower_bound_pp"],
                official=card["official_checkpoint_available"],
                source=used.get("source"),
                steps=used.get("train_steps"),
                status=card["certification_status"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="002")
    parser.add_argument("--envs", default=",".join(TARGETS_PP))
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--artifact-root", default="artifacts/gas")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--state-root", default="research_state")
    args = parser.parse_args()

    round_id = f"{int(args.round):03d}"
    envs = [x.strip() for x in args.envs.split(",") if x.strip()]
    algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()]
    cards = [card_for(row, args.artifact_root, args.seed) for row in target_rows(envs, algorithms)]
    flat = [flatten_card(card) for card in cards]

    reports = Path(args.reports_root)
    state = Path(args.state_root)
    cards_dir = state / "baseline_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for card in cards:
        (cards_dir / f"{safe_name(card['env'], card['algorithm'])}.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")

    write_csv(reports / f"round_{round_id}_baseline_registry.csv", flat)
    write_jsonl(reports / f"round_{round_id}_baseline_cards.jsonl", cards)
    write_jsonl(state / "baseline_registry.jsonl", cards)
    write_lookup_md(reports / f"round_{round_id}_public_target_lookup.md", cards)
    print(json.dumps({"round": round_id, "cards": len(cards), "generated_at": now()}, sort_keys=True))


if __name__ == "__main__":
    main()
