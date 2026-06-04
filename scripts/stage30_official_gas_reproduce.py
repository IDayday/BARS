#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from stage30_official_gas_common import (
    ARCHIVED_PRE_STAGE30_STATUS,
    configure_official_env,
    ensure_ogbench_default_symlinks,
    file_sha256,
    final_goal_threshold,
    gas_agent_flag_args,
    gas_source_identity,
    mean,
    ci95,
    parse_csv_list,
    parse_seed_list,
    read_official_eval_csv,
    scan_official_artifacts,
    write_csv,
)


def _identity_row(art, source_identity: dict[str, str], eval_episodes: int, eval_video_episodes: int, mode: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    if art.manifest_path and art.manifest_path.exists():
        try:
            manifest = json.loads(art.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    return {
        "stage": "stage30_official_gas_reproduction",
        "evidence_class": "OFFICIAL_GAS_BASELINE_CERTIFICATION",
        "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
        **source_identity,
        "artifact_root": str(art.root),
        "env_name": art.env_name,
        "seed": art.seed,
        "task_id": "all_official_tasks",
        "eval_episodes": eval_episodes,
        "eval_video_episodes": eval_video_episodes,
        "eval_final_goal_threshold": final_goal_threshold(art.env_name),
        "keygraph_path": str(art.keygraph_path),
        "keygraph_sha256": manifest.get("keygraph_sha256") or file_sha256(art.keygraph_path),
        "policy_path": str(art.policy_path),
        "policy_sha256": manifest.get("policy_checkpoint_sha256") or file_sha256(art.policy_path),
        "tdr_path": str(art.tdr_path) if art.tdr_path else "",
        "tdr_sha256": manifest.get("tdr_checkpoint_sha256") or (file_sha256(art.tdr_path) if art.tdr_path else ""),
        "official_command_mode": mode,
        "official_eval_csv": str(art.eval_csv) if art.eval_csv else "",
        "artifact_manifest": str(art.manifest_path) if art.manifest_path else "",
        "command_flags": " ".join(gas_agent_flag_args(art.env_name)),
    }


def _copy_for_official_run(art, run_dir: Path) -> tuple[Path, Path]:
    graph_dir = run_dir / "official_artifact_copy" / "graph"
    policy_dir = run_dir / "official_artifact_copy" / "policy"
    graph_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)
    keygraph = graph_dir / "keygraph.pkl"
    policy = policy_dir / art.policy_path.name
    shutil.copy2(art.keygraph_path, keygraph)
    shutil.copy2(art.policy_path, policy)
    return keygraph, policy


def _run_official_eval(args, art, out_root: Path) -> tuple[str, str, int]:
    run_dir = out_root / "official_runs" / art.env_name / f"seed{art.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    keygraph, policy = _copy_for_official_run(art, run_dir)
    log_path = run_dir / "evaluate_gas.log"
    save_eval_dir = run_dir / "evaluate_gas_output"
    cmd = [
        sys.executable,
        "evaluate_gas.py",
        "--run_eval_project",
        "Stage30_Official_GAS_Reproduction",
        "--run_group",
        f"stage30_official_{art.env_name}_seed{art.seed}",
        "--env_name",
        art.env_name,
        "--seed",
        str(art.seed),
        "--gpu",
        str(args.gpu),
        "--save_eval_dir",
        str(save_eval_dir.resolve()),
        "--eval_on_cpu",
        str(args.eval_on_cpu),
        "--eval_episodes",
        str(args.eval_episodes),
        "--eval_video_episodes",
        str(args.eval_video_episodes),
        "--eval_final_goal_threshold",
        str(final_goal_threshold(art.env_name)),
        "--keygraph_path",
        str(keygraph.resolve()),
        "--policy_path",
        str(policy.resolve()),
    ] + gas_agent_flag_args(art.env_name)
    env = configure_official_env(args.gpu)
    env["BARS_USE_TENSORBOARD"] = "1"
    env["TENSORBOARD_LOGDIR"] = str((run_dir / "tensorboard").resolve())
    symlink_rows = ensure_ogbench_default_symlinks(art.env_name, Path(env["OGBENCH_DATASET_DIR"]))
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        for row in symlink_rows:
            log.write("[stage30 dataset symlink] " + json.dumps(row, sort_keys=True) + "\n")
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=str(Path(args.gas_repo_path)), env=env, stdout=log, stderr=subprocess.STDOUT)
    eval_csv = policy.parent / "eval.csv"
    return str(eval_csv) if eval_csv.exists() else "", str(log_path), proc.returncode


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |")
    return lines


def _write_report(out_dir: Path, eval_rows: list[dict[str, Any]], identity_rows: list[dict[str, Any]], mode: str) -> None:
    by_env: dict[str, list[dict[str, Any]]] = {}
    for row in eval_rows:
        by_env.setdefault(str(row["env_name"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for env_name, rows in sorted(by_env.items()):
        vals = [r.get("overall_episode_success") for r in rows]
        lo, hi = ci95(vals)
        summary_rows.append(
            {
                "env_name": env_name,
                "seeds": len(rows),
                "success_mean": f"{mean(vals):.4f}",
                "success_ci95_low": f"{lo:.4f}",
                "success_ci95_high": f"{hi:.4f}",
            }
        )
    lines = [
        "# Stage30 Official GAS Reproduction Report",
        "",
        "Status: OFFICIAL_GAS_BASELINE_CERTIFICATION.",
        f"Pre-Stage30 BARS/Stage28/Stage29 evidence: {ARCHIVED_PRE_STAGE30_STATUS}.",
        "Interpretation rule: these rows are official GAS reproduction evidence; BARS_BASE, Stage28, and Stage29 are not used as GAS evidence.",
        "",
        "## Source Identity",
        "",
    ]
    source = identity_rows[0] if identity_rows else {}
    for key in ["official_repo_url", "official_hf_repo", "gas_repo_path", "official_repo_sha", "gas_vendor_tree_sha256"]:
        lines.append(f"- {key}: `{source.get(key, '')}`")
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- evaluate_gas mode: `{mode}`.",
            "- Official planner, policy, graph, and action outputs are not modified.",
            "- If `mode=cached`, the script records existing local official `policy/eval.csv` outputs and artifact hashes.",
            "- If `mode=run`, official `evaluate_gas.py` is executed on copied keygraph/policy files so original artifacts are not overwritten.",
            "",
            "## Success Summary",
            "",
        ]
    )
    lines.extend(_markdown_table(summary_rows, ["env_name", "seeds", "success_mean", "success_ci95_low", "success_ci95_high"]))
    lines.extend(["", "## Files", ""])
    lines.append(f"- official_gas_eval.csv: `{out_dir / 'official_gas_eval.csv'}`")
    lines.append(f"- artifact_identity.csv: `{out_dir / 'artifact_identity.csv'}`")
    lines.append(f"- command_manifest.jsonl: `{out_dir / 'command_manifest.jsonl'}`")
    (out_dir / "reproduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage30 official-GAS-only reproduction and artifact identity checks.")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--out-root", default="runs_stage30_official_gas/reproduction")
    parser.add_argument("--envs", default="auto")
    parser.add_argument("--seeds", default="auto")
    parser.add_argument("--mode", choices=["cached", "run"], default="cached")
    parser.add_argument("--eval-episodes", type=int, default=49)
    parser.add_argument("--eval-video-episodes", type=int, default=0)
    parser.add_argument("--eval-on-cpu", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = scan_official_artifacts(Path(args.artifact_root), parse_csv_list(args.envs), parse_seed_list(args.seeds))
    source_identity = gas_source_identity(Path(args.gas_repo_path))
    identity_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for art in artifacts:
        identity = _identity_row(art, source_identity, args.eval_episodes, args.eval_video_episodes, args.mode)
        identity_rows.append(identity)
        eval_csv = str(art.eval_csv) if art.eval_csv else ""
        log_path = ""
        returncode: int | str = 0
        if args.mode == "run":
            eval_csv, log_path, returncode = _run_official_eval(args, art, out_dir)
        metrics = read_official_eval_csv(Path(eval_csv)) if eval_csv and Path(eval_csv).exists() else {}
        eval_row = {
            **identity,
            "returncode": returncode,
            "run_log": log_path,
            "eval_csv_used": eval_csv,
            "status": "PASS_OFFICIAL_GAS_EVAL_RECORDED" if metrics else "MISSING_OFFICIAL_EVAL_CSV",
            **metrics,
        }
        eval_rows.append(eval_row)
        manifest_rows.append({"env_name": art.env_name, "seed": art.seed, "command": identity["command_flags"], "eval_csv_used": eval_csv, "run_log": log_path, "returncode": returncode})

    write_csv(out_dir / "artifact_identity.csv", identity_rows)
    write_csv(out_dir / "official_gas_eval.csv", eval_rows)
    with (out_dir / "command_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    _write_report(out_dir, eval_rows, identity_rows, args.mode)
    print(out_dir / "reproduction_report.md")


if __name__ == "__main__":
    main()
