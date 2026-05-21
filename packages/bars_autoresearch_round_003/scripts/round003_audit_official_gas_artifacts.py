#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from round003_lib import (
    AUDIT_ENVS,
    certification_recommended_action,
    env_state_or_visual,
    graph_file_url,
    is_official_slug,
    local_artifact_record,
    official_artifact_url,
    policy_file_url,
    public_target,
    required_train_steps,
    write_csv,
)


FIELDNAMES = [
    "env",
    "state_or_visual",
    "public_mean_pp",
    "public_std_pp",
    "lower_bound_pp",
    "required_train_steps",
    "official_graph_available",
    "official_policy_available",
    "official_tdr_available",
    "official_artifact_path_or_url",
    "local_artifact_path",
    "local_train_steps",
    "artifact_status",
    "recommended_action",
]


def row_for_env(env: str, seed: int, artifact_root: str) -> dict[str, Any]:
    mean, std, lower = public_target(env)
    rec = local_artifact_record(env, seed, artifact_root)
    official_available = is_official_slug(env)
    return {
        "env": env,
        "state_or_visual": env_state_or_visual(env),
        "public_mean_pp": "" if mean is None else mean,
        "public_std_pp": "" if std is None else std,
        "lower_bound_pp": "" if lower is None else lower,
        "required_train_steps": required_train_steps(env),
        "official_graph_available": official_available,
        "official_policy_available": official_available,
        "official_tdr_available": official_available,
        "official_artifact_path_or_url": official_artifact_url(env) if official_available else "",
        "local_artifact_path": rec["root"],
        "local_train_steps": "" if rec["local_train_steps"] is None else rec["local_train_steps"],
        "artifact_status": rec["artifact_status"],
        "recommended_action": certification_recommended_action(env, rec),
        "_graph_file_url": graph_file_url(env) if official_available else "",
        "_policy_file_url": policy_file_url(env) if official_available else "",
    }


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["artifact_status"])] = counts.get(str(row["artifact_status"]), 0) + 1
    lines = [
        "# Round 003 Official GAS Artifact Audit",
        "",
        "Evidence class: E1_BASELINE_REGISTRY. This audit is availability and lineage bookkeeping only.",
        "",
        "## Summary",
        "",
        f"- Rows: {len(rows)}",
        f"- Artifact status counts: `{json.dumps(counts, sort_keys=True)}`",
        "- Medium stitch/navigate remain local 100000-step artifacts, below the public 1000000-step GAS budget.",
        "- Official full-budget local artifacts are present for antmaze-giant-stitch-v0, antmaze-large-explore-v0, and scene-play-v0.",
        "- antmaze-giant-navigate-v0 has a local keygraph and partial policy download only, so it is not certification-ready.",
        "",
        "## Audit Rows",
        "",
        "| env | public mean pp | lower bound pp | official artifact | local steps | status | action |",
        "| --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {env} | {mean} | {lower} | {official} | {steps} | {status} | {action} |".format(
                env=row["env"],
                mean=row["public_mean_pp"],
                lower=row["lower_bound_pp"],
                official=row["official_artifact_path_or_url"] or "not listed",
                steps=row["local_train_steps"],
                status=row["artifact_status"],
                action=row["recommended_action"],
            )
        )
    lines.extend(
        [
            "",
            "## Direct Artifact URLs",
            "",
            "| env | keygraph | params |",
            "| --- | --- | --- |",
        ]
    )
    for row in rows:
        if row.get("_graph_file_url"):
            lines.append(f"| {row['env']} | {row['_graph_file_url']} | {row['_policy_file_url']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", default=",".join(AUDIT_ENVS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-root", default="artifacts/gas")
    parser.add_argument("--out", default="reports/round_003_official_artifact_audit.csv")
    parser.add_argument("--md-out", default="reports/round_003_official_artifact_audit.md")
    args = parser.parse_args()

    envs = [x.strip() for x in args.envs.split(",") if x.strip()]
    rows = [row_for_env(env, args.seed, args.artifact_root) for env in envs]
    write_csv(Path(args.out), rows, FIELDNAMES)
    write_md(Path(args.md_out), rows)
    print(json.dumps({"rows": len(rows), "out": args.out, "md_out": args.md_out}, sort_keys=True))


if __name__ == "__main__":
    main()

