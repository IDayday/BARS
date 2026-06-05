#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from build_cage_eval_command import artifact_env_name, infer_checkpoint_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit GAS artifacts for CAGE pilot runs.")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    return parser


def parse_seed(path: Path) -> int | None:
    match = re.fullmatch(r"seed_?(\d+)", path.name)
    if match:
        return int(match.group(1))
    if path.name.isdigit():
        return int(path.name)
    return None


def discover_seeds(checkpoint_root: Path, env_name: str) -> list[int]:
    seeds: set[int] = set()
    for env_dir_name in [env_name, artifact_env_name(env_name)]:
        env_dir = checkpoint_root / env_dir_name
        if not env_dir.exists():
            continue
        for child in env_dir.iterdir():
            if not child.is_dir():
                continue
            seed = parse_seed(child)
            if seed is not None:
                seeds.add(seed)
        if (env_dir / "graph" / "keygraph.pkl").exists() or (env_dir / "keygraph.pkl").exists():
            seeds.add(0)
    return sorted(seeds)


def audit_env(checkpoint_root: Path, env_name: str) -> dict[str, Any]:
    discovered = discover_seeds(checkpoint_root, env_name)
    rows: list[dict[str, Any]] = []
    for seed in discovered:
        paths = infer_checkpoint_paths(checkpoint_root, env_name, seed)
        missing = [
            key
            for key in ["keygraph_path", "policy_path"]
            if not paths.get(key) or not Path(str(paths[key])).exists()
        ]
        if paths.get("tdr_path") and not Path(str(paths["tdr_path"])).exists():
            missing.append("tdr_path")
        rows.append(
            {
                "seed": seed,
                "keygraph_path": paths.get("keygraph_path"),
                "policy_path": paths.get("policy_path"),
                "tdr_path": paths.get("tdr_path"),
                "missing_files": missing,
                "runnable": not missing,
                "seed_note": "official seed0 pilot" if seed == 0 else "artifact seed, not official seed0 pilot",
            }
        )
    selected = None
    runnable = [row for row in rows if row["runnable"]]
    if runnable:
        selected = next((row for row in runnable if row["seed"] == 0), runnable[0])
    return {
        "env_name": env_name,
        "discovered_seeds": discovered,
        "selected_seed": selected["seed"] if selected else None,
        "selected_seed_note": selected["seed_note"] if selected else None,
        "artifacts": rows,
        "runnable": bool(runnable),
    }


def write_md(path: Path, records: list[dict[str, Any]], checkpoint_root: Path) -> None:
    lines = [
        "# CAGE Checkpoint Audit",
        "",
        f"- checkpoint_root: `{checkpoint_root}`",
        "",
        "| env_name | discovered_seeds | selected_seed | runnable | missing |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        missing: list[str] = []
        for artifact in record["artifacts"]:
            if artifact["missing_files"]:
                missing.append(f"seed{artifact['seed']}:{','.join(artifact['missing_files'])}")
        lines.append(
            "| {env} | {seeds} | {seed} | {runnable} | {missing} |".format(
                env=record["env_name"],
                seeds=" ".join(str(seed) for seed in record["discovered_seeds"]) or "none",
                seed=record["selected_seed"] if record["selected_seed"] is not None else "none",
                runnable=record["runnable"],
                missing="<br>".join(missing) if missing else "none",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    records = [audit_env(checkpoint_root, env_name) for env_name in args.envs]
    out = {"checkpoint_root": str(checkpoint_root), "envs": records}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    write_md(Path(args.out_md), records, checkpoint_root)
    print(json.dumps({"out_json": args.out_json, "runnable_envs": [r["env_name"] for r in records if r["runnable"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
