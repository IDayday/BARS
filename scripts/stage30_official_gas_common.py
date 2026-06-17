#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


OFFICIAL_GAS_REPO = "https://github.com/qortmdgh4141/GAS.git"
OFFICIAL_HF_REPO = "qortmdgh4141/GAS"
ARCHIVED_PRE_STAGE30_STATUS = "ARCHIVED_INTERNAL_EXPLORATION_NOT_GAS_EVIDENCE"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE32_PROTOCOL_REGISTRY = REPO_ROOT / "configs" / "stage32_official_gas_protocol_registry.json"

_PROTOCOL_REGISTRY_CACHE: dict[Path, dict[str, dict[str, Any]]] = {}


@dataclass(frozen=True)
class OfficialGASArtifacts:
    env_name: str
    seed: int
    root: Path
    keygraph_path: Path
    policy_path: Path
    tdr_path: Path | None
    eval_csv: Path | None
    manifest_path: Path | None

    @property
    def dataset_embeddings_path(self) -> Path:
        return self.root / "features" / "dataset_embeddings.npy"

    @property
    def dataset_npz_path(self) -> Path:
        return Path(os.environ.get("OGBENCH_DATASET_DIR", "/mnt/project/offlinerl_datasets/ogbench")) / f"{self.env_name}.npz"


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(root: Path) -> str:
    excluded_dirs = {
        ".git",
        "__pycache__",
        "runs_stage22_tensorboard",
        "wandb",
        "exp_eval",
        "exp_graph",
        "exp_policy",
        "exp_tdr",
    }
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in excluded_dirs for part in rel.parts) or path.suffix == ".pyc":
            continue
        h.update(str(rel).encode("utf-8"))
        h.update(b"\0")
        h.update(file_sha256(path).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def maybe_git_sha(repo: Path) -> str:
    if not (repo / ".git").exists():
        return "UNAVAILABLE_IN_VENDOR_COPY"
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNAVAILABLE_GIT_ERROR"


def gas_source_identity(gas_repo: Path) -> dict[str, str]:
    return {
        "official_repo_url": OFFICIAL_GAS_REPO,
        "official_hf_repo": OFFICIAL_HF_REPO,
        "gas_repo_path": str(gas_repo),
        "official_repo_sha": maybe_git_sha(gas_repo),
        "gas_vendor_tree_sha256": tree_sha256(gas_repo) if gas_repo.exists() else "",
    }


def _safe_file_sha256(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return file_sha256(path)


def env_to_hf_slug(env_name: str) -> str:
    return env_name[:-3] if env_name.endswith("-v0") else env_name


def load_gas_protocol_registry(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    registry_path = Path(path or os.environ.get("STAGE32_PROTOCOL_REGISTRY") or DEFAULT_STAGE32_PROTOCOL_REGISTRY)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path
    registry_path = registry_path.resolve()
    if registry_path in _PROTOCOL_REGISTRY_CACHE:
        return _PROTOCOL_REGISTRY_CACHE[registry_path]
    if not registry_path.exists():
        _PROTOCOL_REGISTRY_CACHE[registry_path] = {}
        return {}
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    env_rows = raw.get("environments", raw if isinstance(raw, list) else [])
    registry = {str(row["env_name"]): dict(row) for row in env_rows}
    _PROTOCOL_REGISTRY_CACHE[registry_path] = registry
    return registry


def official_gas_protocol(env_name: str, *, registry_path: Path | str | None = None) -> dict[str, Any] | None:
    row = load_gas_protocol_registry(registry_path).get(env_name)
    return dict(row) if row else None


def selected_stage32_envs(*, registry_path: Path | str | None = None) -> list[str]:
    registry = load_gas_protocol_registry(registry_path)
    return [env for env, row in registry.items() if int(row.get("selected_paper_env", 1))]


def gas_agent_flag_args(env_name: str) -> list[str]:
    protocol = official_gas_protocol(env_name)
    if protocol:
        return [
            "--agent_config.encoder",
            str(protocol["encoder"]),
            "--agent_config.discount",
            str(protocol["discount"]),
            "--agent_config.tdr_expectile",
            str(protocol["tdr_expectile"]),
            "--agent_config.alpha",
            str(protocol["alpha"]),
            "--agent_config.batch_size",
            str(protocol["batch_size"]),
            "--agent_config.p_aug",
            str(protocol["p_aug"]),
            "--agent_config.way_steps",
            str(protocol["way_steps"]),
        ]
    slug = env_to_hf_slug(env_name)
    discount = "0.995" if "giant" in slug else "0.99"
    alpha = "0.01" if "explore" in slug else "1.0"
    expectile = "0.999"
    way_steps = "48" if ("scene" in slug or "kitchen" in slug) else "8"
    if "kitchen" in slug:
        alpha = "10.0"
        expectile = "0.95"
    return [
        "--agent_config.encoder",
        "not_used",
        "--agent_config.discount",
        discount,
        "--agent_config.tdr_expectile",
        expectile,
        "--agent_config.alpha",
        alpha,
        "--agent_config.batch_size",
        "1024",
        "--agent_config.p_aug",
        "0.0",
        "--agent_config.way_steps",
        way_steps,
    ]


def gas_config_overrides(env_name: str) -> dict[str, Any]:
    args = gas_agent_flag_args(env_name)
    out: dict[str, Any] = {}
    for key, value in zip(args[::2], args[1::2]):
        field = key.removeprefix("--agent_config.")
        if field in {"discount", "tdr_expectile", "alpha", "p_aug"}:
            out[field] = float(value)
        elif field in {"batch_size", "way_steps"}:
            out[field] = int(value)
        else:
            out[field] = value
    return out


def final_goal_threshold(env_name: str) -> int:
    protocol = official_gas_protocol(env_name)
    if protocol:
        return int(protocol["eval_final_goal_threshold"])
    return 1 if "kitchen" in env_name else 2


def read_artifact_manifest(art: OfficialGASArtifacts) -> dict[str, Any]:
    if not art.manifest_path or not art.manifest_path.exists():
        return {}
    try:
        return json.loads(art.manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def artifact_identity_fields(art: OfficialGASArtifacts) -> dict[str, Any]:
    manifest = read_artifact_manifest(art)
    return {
        "artifact_root": str(art.root),
        "env_name": art.env_name,
        "seed": art.seed,
        "keygraph_path": str(art.keygraph_path),
        "keygraph_sha256": manifest.get("keygraph_sha256") or _safe_file_sha256(art.keygraph_path),
        "policy_path": str(art.policy_path),
        "policy_sha256": manifest.get("policy_checkpoint_sha256") or _safe_file_sha256(art.policy_path),
        "tdr_path": str(art.tdr_path) if art.tdr_path else "",
        "tdr_sha256": manifest.get("tdr_checkpoint_sha256") or _safe_file_sha256(art.tdr_path),
        "official_eval_csv": str(art.eval_csv) if art.eval_csv else "",
        "artifact_manifest": str(art.manifest_path) if art.manifest_path else "",
    }


def protocol_lock_row(
    art: OfficialGASArtifacts,
    gas_repo: Path,
    *,
    stage: str,
    evidence_class: str,
    wrapper_status: str,
    command_line: str,
    task_id: str = "",
    episode_count: int | str = "",
    goal_threshold: int | str | None = None,
    subgoal_horizon: int | str = "",
    eval_on_cpu: int | str = "",
    gpu: int | str = "",
    source_identity: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage,
        "evidence_class": evidence_class,
        "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
        **(source_identity or gas_source_identity(gas_repo)),
        **artifact_identity_fields(art),
        "task_id": task_id,
        "episode_count": episode_count,
        "goal_threshold": final_goal_threshold(art.env_name) if goal_threshold is None else goal_threshold,
        "subgoal_horizon": subgoal_horizon,
        "evaluation_command": command_line,
        "wrapper_status": wrapper_status,
        "eval_on_cpu": eval_on_cpu,
        "gpu": gpu,
        "official_graph_modified": 0,
        "official_planner_modified": 0,
        "official_policy_modified": 0,
        "official_action_outputs_modified": 0,
    }
    if extra:
        row.update(extra)
    return row


def scan_official_artifacts(artifact_root: Path, envs: Sequence[str] | None, seeds: Sequence[int] | None) -> list[OfficialGASArtifacts]:
    rows: list[OfficialGASArtifacts] = []
    if envs:
        env_dirs = [artifact_root / env for env in envs]
    else:
        env_dirs = sorted(p for p in artifact_root.iterdir() if p.is_dir() and not p.name.startswith("visual-"))
    for env_dir in env_dirs:
        if not env_dir.exists():
            continue
        seed_dirs = [env_dir / f"seed{s}" for s in seeds] if seeds else sorted(p for p in env_dir.glob("seed*") if p.is_dir())
        for root in seed_dirs:
            try:
                seed = int(root.name.removeprefix("seed"))
            except ValueError:
                continue
            keygraph = root / "graph" / "keygraph.pkl"
            policies = sorted((root / "policy").glob("params_*.pkl"))
            tdrs = sorted((root / "tdr").glob("params_*.pkl"))
            if not keygraph.exists() or not policies:
                continue
            eval_csv = root / "policy" / "eval.csv"
            rows.append(
                OfficialGASArtifacts(
                    env_name=env_dir.name,
                    seed=seed,
                    root=root,
                    keygraph_path=keygraph,
                    policy_path=policies[-1],
                    tdr_path=tdrs[-1] if tdrs else None,
                    eval_csv=eval_csv if eval_csv.exists() else None,
                    manifest_path=(root / "manifest.json") if (root / "manifest.json").exists() else None,
                )
            )
    return rows


def read_official_eval_csv(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    row = rows[-1]
    out: dict[str, Any] = {"eval_csv_rows": len(rows)}
    for key, value in row.items():
        if key == "step" or "episode.success" in key or "overall_episode" in key:
            out[key.replace("eval/", "").replace(".", "_")] = value
    return out


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def safe_float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def mean(values: Iterable[Any]) -> float:
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def ci95(values: Iterable[Any]) -> tuple[float, float]:
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return float("nan"), float("nan")
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return mu, mu
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    half = 1.96 * math.sqrt(var / len(xs))
    return mu - half, mu + half


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw or raw.lower() == "auto":
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_seed_list(raw: str | None) -> list[int]:
    if not raw or raw.lower() == "auto":
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def configure_official_env(gpu: str | int = "0") -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("OGBENCH_DATASET_DIR", "/mnt/project/offlinerl_datasets/ogbench")
    env.setdefault("D4RL_DATASET_DIR", "/mnt/project/offlinerl_datasets/d4rl")
    env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    ogbench_package_root = REPO_ROOT / "external_src" / "tmd-release"
    if ogbench_package_root.exists():
        existing_pythonpath = [x for x in env.get("PYTHONPATH", "").split(":") if x]
        if str(ogbench_package_root) not in existing_pythonpath:
            env["PYTHONPATH"] = ":".join([str(ogbench_package_root), *existing_pythonpath])
    mujoco_bin = "/root/.mujoco/mujoco210/bin"
    if Path(mujoco_bin).exists() and mujoco_bin not in env.get("LD_LIBRARY_PATH", "").split(":"):
        env["LD_LIBRARY_PATH"] = f"{env.get('LD_LIBRARY_PATH', '')}:{mujoco_bin}".lstrip(":")
    if str(gpu).lower() in {"cpu", "-1", ""}:
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
    else:
        gpu_str = str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        first_gpu = gpu_str.split(",", 1)[0].strip()
        if first_gpu.isdigit():
            env["MUJOCO_EGL_DEVICE_ID"] = first_gpu
    return env


def ensure_ogbench_default_symlinks(env_name: str, dataset_dir: Path | None = None, default_dir: Path | None = None) -> list[dict[str, str]]:
    """Point OGBench's hard-coded default dataset path at local datasets.

    Official GAS calls `ogbench.make_env_and_datasets()` without a dataset_dir
    argument. OGBench captures `~/.ogbench/data` as a default argument at import
    time, so environment variables cannot redirect it. Symlinking avoids network
    downloads while keeping official GAS source and control logic unchanged.
    """
    dataset_dir = dataset_dir or Path(os.environ.get("OGBENCH_DATASET_DIR", "/mnt/project/offlinerl_datasets/ogbench"))
    default_dir = default_dir or Path.home() / ".ogbench" / "data"
    default_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for suffix in (".npz", "-val.npz"):
        src = dataset_dir / f"{env_name}{suffix}"
        dst = default_dir / f"{env_name}{suffix}"
        row = {"env_name": env_name, "source": str(src), "target": str(dst), "status": ""}
        if not src.exists():
            row["status"] = "missing_local_dataset"
            rows.append(row)
            continue
        if dst.exists() or dst.is_symlink():
            try:
                if dst.is_symlink() and Path(os.readlink(dst)).resolve() == src.resolve():
                    row["status"] = "existing_symlink"
                    rows.append(row)
                    continue
                if dst.resolve() == src.resolve():
                    row["status"] = "existing_same_file"
                    rows.append(row)
                    continue
            except Exception:
                pass
            if dst.is_symlink():
                dst.unlink()
            else:
                row["status"] = "target_exists_not_symlink"
                rows.append(row)
                continue
        try:
            os.symlink(src, dst)
            row["status"] = "created_symlink"
        except FileExistsError:
            try:
                if dst.resolve() == src.resolve():
                    row["status"] = "created_by_concurrent_worker"
                else:
                    row["status"] = "target_exists_not_symlink"
            except Exception:
                row["status"] = "target_exists_unresolved"
        rows.append(row)
    return rows


def recover_node_dataset_indices(
    nodes: Any,
    embeddings_path: Path,
    *,
    base_node_count: int | None = None,
    tolerance: float = 1e-5,
    batch_size: int = 4096,
) -> dict[int, dict[str, Any]]:
    """Map official keygraph node embeddings back to dataset embedding rows.

    This is diagnostic-only. It does not modify the official keygraph. Rows with
    no exact-enough match are marked as unavailable by omitting them.
    """
    import numpy as np

    if not embeddings_path.exists():
        return {}
    node_arr = np.asarray(nodes)
    limit = min(base_node_count if base_node_count is not None else len(node_arr), len(node_arr))
    node_arr = node_arr[:limit].astype(np.float32, copy=False)
    emb = np.load(embeddings_path, mmap_mode="r")
    node_norm = np.sum(node_arr * node_arr, axis=1)
    best_idx = np.full(limit, -1, dtype=np.int64)
    best_dist = np.full(limit, np.inf, dtype=np.float64)
    for start in range(0, emb.shape[0], batch_size):
        batch = np.asarray(emb[start : start + batch_size], dtype=np.float32)
        dists = np.sum(batch * batch, axis=1)[:, None] + node_norm[None, :] - 2.0 * batch.dot(node_arr.T)
        dists = np.maximum(dists, 0.0)
        local = np.argmin(dists, axis=0)
        vals = dists[local, np.arange(limit)]
        mask = vals < best_dist
        best_dist[mask] = vals[mask]
        best_idx[mask] = start + local[mask]
    out: dict[int, dict[str, Any]] = {}
    for node_idx, (dataset_idx, dist2) in enumerate(zip(best_idx.tolist(), best_dist.tolist())):
        dist = math.sqrt(float(dist2)) if math.isfinite(float(dist2)) else float("inf")
        if dataset_idx >= 0 and dist <= tolerance:
            out[node_idx] = {
                "dataset_idx": int(dataset_idx),
                "embedding_match_dist": dist,
                "embedding_match_tolerance": tolerance,
                "exact_embedding_match": int(dist <= 1e-5),
            }
    return out


def trajectory_ids_from_dataset(dataset_npz_path: Path) -> dict[int, int]:
    import numpy as np

    if not dataset_npz_path.exists():
        return {}
    data = np.load(dataset_npz_path, mmap_mode="r")
    if "terminals" not in data.files:
        return {}
    terminals = np.asarray(data["terminals"], dtype=bool)
    traj = np.zeros(len(terminals), dtype=np.int64)
    if len(terminals) > 1:
        traj[1:] = np.cumsum(terminals[:-1])
    return {int(i): int(t) for i, t in enumerate(traj.tolist())}


def load_dataset_arrays(dataset_npz_path: Path) -> Any:
    import numpy as np

    return np.load(dataset_npz_path, mmap_mode="r") if dataset_npz_path.exists() else None
