#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.external.gas_artifacts import env_to_hf_slug, resolve_gas_artifacts
from bars.external.gas_backbone import GASBackbone
from bars.gas_bars.graph_table import export_edges, export_nodes, load_gas_keygraph


def _git(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git"] + cmd, cwd=str(cwd), text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception as exc:
        return f"ERROR:{exc!r}"


def _checkpoint_step(path: str | None) -> int | None:
    if not path:
        return None
    try:
        return int(Path(path).name.split("params_")[-1].split(".")[0])
    except Exception:
        return None


def _audit_env(args: argparse.Namespace, env_name: str, seed: int, gpu: str) -> dict[str, Any]:
    row: dict[str, Any] = {"env": env_name, "seed": seed, "gpu": gpu}
    artifacts = resolve_gas_artifacts(env_name, seed, args.gas_artifact_root)
    row.update(artifacts.to_dict())
    row["hf_slug"] = env_to_hf_slug(env_name)
    row["policy_checkpoint_step"] = _checkpoint_step(str(artifacts.policy_checkpoint) if artifacts.policy_checkpoint else None)
    row["tdr_checkpoint_step"] = _checkpoint_step(str(artifacts.tdr_checkpoint) if artifacts.tdr_checkpoint else None)
    row["artifact_complete"] = bool(artifacts.complete)
    if artifacts.keygraph is not None:
        try:
            kg = load_gas_keygraph(artifacts.keygraph)
            nodes = export_nodes(kg)
            edges = export_edges(kg)
            row["keygraph_node_count"] = int(len(nodes))
            row["keygraph_edge_count"] = int(len(edges))
            row["keygraph_way_steps"] = float(getattr(kg, "way_steps", 0) or 0)
            row["keygraph_task_count"] = int(len(getattr(kg, "task_node_idx_dict", {}) or {}))
            row["gas_cross_edge_count"] = int((edges["edge_source"].astype(str) == "gas_scc_connector").sum()) if "edge_source" in edges else 0
        except Exception as exc:
            row["keygraph_error"] = repr(exc)
    if not artifacts.complete:
        row["protocol_status"] = "missing_artifact"
        return row
    try:
        bb = GASBackbone(env_name=env_name, seed=seed, artifact_root=Path(args.gas_artifact_root), gas_repo_path=Path(args.gas_repo_path), gpu=gpu, artifacts=artifacts)
        bb.load_keygraph(artifacts.keygraph)
        bb.load_policy(artifacts.policy_checkpoint)
        env, _, _ = bb.load_env_and_dataset()
        task_ids = bb.get_task_ids(env)
        row["task_ids"] = "|".join(map(str, task_ids))
        row["task_id_count"] = int(len(task_ids))
        spec = getattr(env, "spec", None)
        row["max_episode_steps"] = getattr(spec, "max_episode_steps", None)
        checks = []
        for task_id in task_ids[: min(args.max_tasks, len(task_ids))]:
            env, observation, goal, _, done, _ = bb.setup_task_env(env, env_name, int(task_id), seed + int(task_id), render_goal=False)
            phi_obs = bb.get_phi(observation)
            phi_goal = bb.get_phi(goal)
            checks.append(
                {
                    "task_id": int(task_id),
                    "goal_present": goal is not None,
                    "obs_dim": int(len(observation)),
                    "goal_dim": int(len(goal)),
                    "phi_dim": int(len(phi_goal)),
                    "initial_goal_dist_phi": float(((phi_goal - phi_obs) ** 2).sum() ** 0.5),
                    "done_after_reset": bool(done),
                }
            )
        row["reset_checks"] = json.dumps(checks, sort_keys=True)
        row["goal_present_all_checked"] = all(x["goal_present"] for x in checks)
        row["protocol_status"] = "ok" if row["goal_present_all_checked"] else "goal_missing"
    except Exception as exc:
        row["protocol_status"] = "error"
        row["protocol_error"] = repr(exc)
    return row


def write_md(rows: list[dict[str, Any]], path: Path, gas_repo: Path) -> None:
    dirty = _git(["status", "--short"], gas_repo)
    commit = _git(["rev-parse", "HEAD"], gas_repo)
    remote = _git(["remote", "-v"], gas_repo)
    df = pd.DataFrame(rows)
    lines = [
        "# Stage23 Protocol Audit",
        "",
        "## Official GAS Repo",
        f"- Path: `{gas_repo}`",
        f"- Commit: `{commit}`",
        f"- Dirty: `{bool(dirty.strip())}`",
        "```",
        dirty or "clean",
        "```",
        "```",
        remote,
        "```",
        "",
        "## Environment Checks",
    ]
    if len(df):
        cols = [c for c in ["env", "seed", "artifact_complete", "policy_checkpoint_step", "tdr_checkpoint_step", "keygraph_node_count", "keygraph_edge_count", "keygraph_way_steps", "task_id_count", "max_episode_steps", "goal_present_all_checked", "protocol_status"] if c in df.columns]
        try:
            lines.append(df[cols].to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + df[cols].to_csv(index=False).strip() + "\n```")
    lines.extend(
        [
            "",
            "## Red Flags",
        ]
    )
    if dirty.strip():
        lines.append("- Local `external_src/GAS` is dirty; A-route reproduction must distinguish pristine official from locally patched official scripts.")
    for row in rows:
        if row.get("policy_checkpoint_step") not in (500000, 1000000):
            lines.append(f"- `{row.get('env')}` uses policy checkpoint step `{row.get('policy_checkpoint_step')}`, below full official 1M/500k reproduction.")
        if row.get("protocol_status") != "ok":
            lines.append(f"- `{row.get('env')}` protocol status is `{row.get('protocol_status')}`: {row.get('protocol_error', '')}")
    if not any(line.startswith("-") for line in lines[lines.index("## Red Flags") + 1 :]):
        lines.append("- None detected in checked tasks.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", default="antmaze-medium-navigate-v0,antmaze-medium-stitch-v0")
    p.add_argument("--seeds", default="0")
    p.add_argument("--gpus", default="cpu")
    p.add_argument("--gas-artifact-root", default="artifacts/gas")
    p.add_argument("--gas-repo-path", default="external_src/GAS")
    p.add_argument("--out-csv", default="reports/stage23_protocol_audit.csv")
    p.add_argument("--out-md", default="reports/stage23_protocol_audit.md")
    p.add_argument("--max-tasks", type=int, default=2)
    args = p.parse_args()
    envs = [x for x in args.envs.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    gpus = [x for x in args.gpus.split(",") if x] or ["cpu"]
    rows = []
    i = 0
    for env in envs:
        for seed in seeds:
            gpu = gpus[i % len(gpus)]
            i += 1
            rows.append(_audit_env(args, env, seed, gpu))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    write_md(rows, Path(args.out_md), Path(args.gas_repo_path))


if __name__ == "__main__":
    main()
