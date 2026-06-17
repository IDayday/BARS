#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stage30_official_gas_common import (
    configure_official_env,
    ensure_ogbench_default_symlinks,
    final_goal_threshold,
    gas_agent_flag_args,
    official_gas_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/root/miniconda3/envs/gcrlo/bin/python")
DEFAULT_GAS_REPO = REPO_ROOT / "external_src" / "GAS"
DEFAULT_DATASET_DIR = Path("/mnt/project/offlinerl_datasets/ogbench")


@dataclass(frozen=True)
class Variant:
    name: str
    source_kind: str = "original"
    source_path: str = ""


STATUS_FIELDS = [
    "time",
    "variant",
    "env_name",
    "seed",
    "gpu",
    "eval_on_cpu",
    "episodes",
    "risk_weight",
    "status",
    "pid",
    "returncode",
    "duration_sec",
    "keygraph_path",
    "policy_path",
    "eval_csv",
    "log_path",
    "command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch generic official-GAS graph-patch evaluations on one GPU. "
            "Graph patch generation stays offline-only; environment rollouts are used only for reporting."
        )
    )
    parser.add_argument("--gpu", default="3", help="Single visible GPU id. Default: 3.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-name", required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Seed root containing graph/keygraph.pkl and policy/params_*.pkl.",
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-video-episodes", type=int, default=0)
    parser.add_argument(
        "--eval-on-cpu",
        type=int,
        default=-1,
        help="Override eval_on_cpu. -1 uses the official protocol registry.",
    )
    parser.add_argument("--risk-weight", type=float, default=0.10)
    parser.add_argument("--contract-threshold", type=float, default=0.5)
    parser.add_argument("--score-column", default="contract_prob_mean")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gas-repo", type=Path, default=DEFAULT_GAS_REPO)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--variant-spec",
        action="append",
        default=[],
        help=(
            "Variant spec. Use 'original' or 'NAME=rows:PATH' or 'NAME=edge:PATH'. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root. Defaults to runs_stage62_official_contract_eval_gpu3/<utc stamp>.",
    )
    parser.add_argument("--run-eval-project", default="stage62_official_contract_eval")
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--dry-run", type=int, default=0)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_variant_specs(values: list[str]) -> list[Variant]:
    if not values:
        return [Variant(name="original")]
    out: list[Variant] = []
    for raw in values:
        spec = raw.strip()
        if not spec:
            continue
        if spec == "original":
            out.append(Variant(name="original"))
            continue
        if "=" not in spec:
            raise SystemExit(f"Invalid --variant-spec {spec!r}: expected original or NAME=rows:PATH or NAME=edge:PATH")
        name, rhs = spec.split("=", 1)
        if ":" not in rhs:
            raise SystemExit(f"Invalid --variant-spec {spec!r}: missing rows:/edge: prefix")
        source_kind, source_path = rhs.split(":", 1)
        if source_kind not in {"rows", "edge"}:
            raise SystemExit(f"Invalid --variant-spec {spec!r}: source kind must be rows or edge")
        out.append(Variant(name=name.strip(), source_kind=source_kind, source_path=source_path.strip()))
    if not out:
        raise SystemExit("No valid --variant-spec entries")
    return out


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = configure_official_env(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["OGBENCH_DATASET_DIR"] = str(args.dataset_dir)
    env["BARS_OGBENCH_DATASET_DIR"] = str(args.dataset_dir)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MUJOCO_GL", "egl")
    pythonpath = [
        str(REPO_ROOT / "external_src" / "tmd-release"),
        str(args.gas_repo),
        str(REPO_ROOT),
    ]
    existing = [part for part in env.get("PYTHONPATH", "").split(":") if part]
    for part in existing:
        if part not in pythonpath:
            pythonpath.append(part)
    env["PYTHONPATH"] = ":".join(pythonpath)
    return env


def append_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STATUS_FIELDS})


def append_command(path: Path, cwd: Path, env: dict[str, str], cmd: list[str]) -> None:
    visible_env = {
        "OGBENCH_DATASET_DIR": env.get("OGBENCH_DATASET_DIR", ""),
        "PYTHONPATH": env.get("PYTHONPATH", ""),
        "WANDB_MODE": env.get("WANDB_MODE", ""),
        "WANDB_DISABLED": env.get("WANDB_DISABLED", ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": env.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "MUJOCO_GL": env.get("MUJOCO_GL", ""),
        "MUJOCO_EGL_DEVICE_ID": env.get("MUJOCO_EGL_DEVICE_ID", ""),
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] ")
        fh.write(f"cwd={shlex.quote(str(cwd))} ")
        fh.write("env ")
        for key, value in visible_env.items():
            if value:
                fh.write(f"{key}={shlex.quote(value)} ")
        fh.write("setsid ")
        fh.write(" ".join(shlex.quote(part) for part in cmd))
        fh.write("\n")


def weight_tag(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def run_step(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as fh:
        fh.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] ".encode("utf-8"))
        fh.write((" ".join(shlex.quote(part) for part in cmd) + "\n").encode("utf-8"))
        if dry_run:
            return
        subprocess.run(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, check=True)


def prepare(args: argparse.Namespace) -> None:
    if "," in str(args.gpu):
        raise SystemExit("This eval launcher is constrained to one GPU; pass one --gpu id.")
    if not args.python_bin.exists():
        raise SystemExit(f"Missing python binary: {args.python_bin}")
    if not args.gas_repo.exists():
        raise SystemExit(f"Missing GAS repo: {args.gas_repo}")
    if not args.artifact_root.exists():
        raise SystemExit(f"Missing artifact root: {args.artifact_root}")
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", "-val.npz"):
        dataset = args.dataset_dir / f"{args.env_name}{suffix}"
        if not dataset.exists():
            raise SystemExit(f"Missing local dataset: {dataset}")
    ensure_ogbench_default_symlinks(args.env_name, dataset_dir=args.dataset_dir)


def original_keygraph_path(args: argparse.Namespace) -> Path:
    path = args.artifact_root / "graph" / "keygraph.pkl"
    if not path.exists():
        raise SystemExit(f"Missing original keygraph: {path}")
    return path


def policy_path(args: argparse.Namespace) -> Path:
    for name in ("params_1000000.pkl", "params_500000.pkl"):
        path = args.artifact_root / "policy" / name
        if path.exists():
            return path
    raise SystemExit(f"Missing policy checkpoint under {args.artifact_root / 'policy'}")


def variant_keygraph(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    variant: Variant,
    prep_log_dir: Path,
) -> Path:
    if variant.source_kind == "original":
        return original_keygraph_path(args)

    variant_root = args.out_root / variant.name
    patch_dir = variant_root / f"patched_graph_contract_w{weight_tag(args.risk_weight)}"
    patched_keygraph = patch_dir / "keygraph.pkl"
    if patched_keygraph.exists():
        return patched_keygraph

    if variant.source_kind == "rows":
        scored_rows = resolve_repo_path(Path(variant.source_path))
        if not scored_rows.exists():
            raise SystemExit(f"Missing scored rows for {variant.name}: {scored_rows}")
        edge_scores_dir = variant_root / "edge_scores"
        edge_score_cmd = [
            str(args.python_bin),
            "scripts/stage45_make_caplite_edge_scores.py",
            "--scored-rows",
            str(scored_rows),
            "--env",
            args.env_name,
            "--out",
            str(edge_scores_dir),
            "--score-column",
            args.score_column,
            "--contract-threshold",
            str(args.contract_threshold),
        ]
        run_step(
            cmd=edge_score_cmd,
            cwd=REPO_ROOT,
            env=env,
            log_path=prep_log_dir / f"{variant.name}_edge_scores.log",
            dry_run=bool(args.dry_run),
        )
        edge_scores_csv = edge_scores_dir / "caplite_edge_scores.csv"
    elif variant.source_kind == "edge":
        edge_scores_csv = resolve_repo_path(Path(variant.source_path))
        if not edge_scores_csv.exists():
            raise SystemExit(f"Missing edge scores for {variant.name}: {edge_scores_csv}")
    else:
        raise SystemExit(f"Unsupported source kind for {variant.name}: {variant.source_kind}")

    patch_cmd = [
        str(args.python_bin),
        "scripts/stage36_patch_official_gas_keygraph_support.py",
        "--keygraph-path",
        str(original_keygraph_path(args)),
        "--edge-scores-csv",
        str(edge_scores_csv),
        "--out-keygraph-path",
        str(patched_keygraph),
        "--mode",
        "penalize",
        "--support-column",
        "contract_available",
        "--min-support",
        "1",
        "--risk-column",
        "r_exec",
        "--risk-weight",
        str(args.risk_weight),
        "--missing-score-policy",
        "protect",
        "--protect-goal-edges",
        "1",
    ]
    run_step(
        cmd=patch_cmd,
        cwd=REPO_ROOT,
        env=env,
        log_path=prep_log_dir / f"{variant.name}_patch_graph.log",
        dry_run=bool(args.dry_run),
    )
    return patched_keygraph


def job_command(
    *,
    args: argparse.Namespace,
    variant: Variant,
    keygraph_path: Path,
    policy_ckpt: Path,
    save_eval_dir: Path,
    eval_csv: Path,
) -> tuple[list[str], int]:
    protocol = official_gas_protocol(args.env_name) or {}
    eval_on_cpu = int(protocol.get("eval_on_cpu", 1)) if args.eval_on_cpu < 0 else int(args.eval_on_cpu)
    cmd = [
        str(args.python_bin),
        "evaluate_gas.py",
        "--run_eval_project",
        args.run_eval_project,
        "--run_group",
        variant.name,
        "--env_name",
        args.env_name,
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--save_eval_dir",
        str(save_eval_dir),
        "--eval_result_path",
        str(eval_csv),
        "--eval_on_cpu",
        str(eval_on_cpu),
        "--eval_episodes",
        str(args.eval_episodes),
        "--eval_video_episodes",
        str(args.eval_video_episodes),
        "--eval_final_goal_threshold",
        str(final_goal_threshold(args.env_name)),
        "--keygraph_path",
        str(keygraph_path),
        "--policy_path",
        str(policy_ckpt),
        *gas_agent_flag_args(args.env_name),
    ]
    return cmd, eval_on_cpu


def load_eval_metric(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh), None)
    if not row or key not in row or row[key] == "":
        return None
    return float(row[key])


def write_summary(out_root: Path, rows: list[dict[str, Any]]) -> Path:
    summary_path = out_root / "smoke_summary.csv"
    fields = [
        "variant",
        "status",
        "episodes",
        "risk_weight",
        "eval_csv",
        "overall_success",
        "overall_normalized_return",
        "overall_return",
        "overall_length",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            eval_csv = Path(str(row["eval_csv"]))
            writer.writerow(
                {
                    "variant": row["variant"],
                    "status": row["status"],
                    "episodes": row["episodes"],
                    "risk_weight": row["risk_weight"],
                    "eval_csv": row["eval_csv"],
                    "overall_success": load_eval_metric(eval_csv, "eval/overall_episode.success"),
                    "overall_normalized_return": load_eval_metric(eval_csv, "eval/overall_episode.normalized_return"),
                    "overall_return": load_eval_metric(eval_csv, "eval/overall_episode.return"),
                    "overall_length": load_eval_metric(eval_csv, "eval/overall_episode.length"),
                }
            )
    return summary_path


def main() -> None:
    args = parse_args()
    args.python_bin = resolve_repo_path(args.python_bin)
    args.gas_repo = resolve_repo_path(args.gas_repo)
    args.dataset_dir = resolve_repo_path(args.dataset_dir)
    args.artifact_root = resolve_repo_path(args.artifact_root)
    if args.out_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        args.out_root = REPO_ROOT / "runs_stage62_official_contract_eval_gpu3" / stamp
    else:
        args.out_root = resolve_repo_path(args.out_root)
    variants = parse_variant_specs(args.variant_spec)

    prepare(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(exist_ok=True)
    (args.out_root / "prep_logs").mkdir(exist_ok=True)
    (args.out_root / "raw_eval").mkdir(exist_ok=True)
    (args.out_root / "eval_csv").mkdir(exist_ok=True)
    latest = REPO_ROOT / "runs_stage62_official_contract_eval_gpu3" / "latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    if latest.exists() or latest.is_symlink():
        latest.unlink(missing_ok=True)
    try:
        latest.symlink_to(args.out_root)
    except FileExistsError:
        latest.unlink(missing_ok=True)
        latest.symlink_to(args.out_root)

    env = build_env(args)
    status_path = args.out_root / "stage62_official_contract_eval_status.csv"
    command_log = args.out_root / "commands.log"
    policy_ckpt = policy_path(args)
    rows: list[dict[str, Any]] = []
    active: list[tuple[subprocess.Popen[Any], dict[str, Any], Any]] = []

    keygraph_by_variant: dict[str, Path] = {}
    for variant in variants:
        keygraph_by_variant[variant.name] = variant_keygraph(
            args=args,
            env=env,
            variant=variant,
            prep_log_dir=args.out_root / "prep_logs",
        )

    for variant in variants:
        keygraph_path = keygraph_by_variant[variant.name]
        save_eval_dir = args.out_root / "raw_eval" / variant.name
        eval_csv = args.out_root / "eval_csv" / f"{variant.name}.csv"
        log_path = args.out_root / "logs" / f"{variant.name}.log"
        cmd, eval_on_cpu = job_command(
            args=args,
            variant=variant,
            keygraph_path=keygraph_path,
            policy_ckpt=policy_ckpt,
            save_eval_dir=save_eval_dir,
            eval_csv=eval_csv,
        )
        append_command(command_log, args.gas_repo, env, cmd)
        row: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "variant": variant.name,
            "env_name": args.env_name,
            "seed": args.seed,
            "gpu": args.gpu,
            "eval_on_cpu": eval_on_cpu,
            "episodes": args.eval_episodes,
            "risk_weight": args.risk_weight,
            "status": "DRY_RUN" if args.dry_run else "RUNNING",
            "keygraph_path": str(keygraph_path),
            "policy_path": str(policy_ckpt),
            "eval_csv": str(eval_csv),
            "log_path": str(log_path),
            "command": " ".join(shlex.quote(part) for part in cmd),
        }
        rows.append(row)
        if args.dry_run:
            continue
        fh = log_path.open("wb")
        proc = subprocess.Popen(
            ["setsid", *cmd],
            cwd=args.gas_repo,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        row["pid"] = proc.pid
        row["_start_time"] = time.time()
        active.append((proc, row, fh))
        (args.out_root / "logs" / f"{variant.name}.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        append_status(status_path, rows)

    append_status(status_path, rows)
    if not args.wait or args.dry_run:
        print(args.out_root)
        return

    while active:
        still: list[tuple[subprocess.Popen[Any], dict[str, Any], Any]] = []
        for proc, row, fh in active:
            code = proc.poll()
            if code is None:
                still.append((proc, row, fh))
                continue
            fh.close()
            row["returncode"] = code
            row["duration_sec"] = f"{time.time() - float(row.get('_start_time', time.time())):.3f}"
            row["status"] = "COMPLETE" if code == 0 and Path(str(row["eval_csv"])).exists() else "FAILED"
        active = still
        append_status(status_path, rows)
        if active:
            time.sleep(10)

    for row in rows:
        row.pop("_start_time", None)
    append_status(status_path, rows)
    print(write_summary(args.out_root, rows))


if __name__ == "__main__":
    main()
