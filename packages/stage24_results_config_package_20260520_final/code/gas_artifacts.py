from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


REPO_ID = "qortmdgh4141/GAS"
DEFAULT_HF_ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")
OFFICIAL_PRETRAINED_SLUGS = {
    "antmaze-giant-navigate",
    "antmaze-giant-stitch",
    "antmaze-large-explore",
    "scene-play",
    "kitchen-partial",
    "visual-antmaze-giant-navigate",
    "visual-antmaze-giant-stitch",
    "visual-antmaze-large-explore",
    "visual-scene-play",
}


@dataclass
class GASArtifactSet:
    env_name: str
    seed: int
    root: Path
    tdr_dir: Path
    policy_dir: Path
    graph_dir: Path
    features_dir: Path
    tdr_checkpoint: Optional[Path]
    policy_checkpoint: Optional[Path]
    keygraph: Optional[Path]
    dataset_embeddings: Optional[Path]

    @property
    def complete(self) -> bool:
        return self.policy_checkpoint is not None and self.keygraph is not None

    def to_dict(self) -> Dict[str, str | int | bool | None]:
        return {
            "env_name": self.env_name,
            "seed": self.seed,
            "root": str(self.root),
            "tdr_checkpoint": str(self.tdr_checkpoint) if self.tdr_checkpoint else None,
            "policy_checkpoint": str(self.policy_checkpoint) if self.policy_checkpoint else None,
            "keygraph": str(self.keygraph) if self.keygraph else None,
            "dataset_embeddings": str(self.dataset_embeddings) if self.dataset_embeddings else None,
            "complete": self.complete,
        }


def _env_dir(env_name: str, root: str | os.PathLike[str], seed: int) -> Path:
    return Path(root) / env_name.replace("/", "_") / f"seed{seed}"


def env_to_hf_slug(env_name: str) -> str:
    slug = env_name
    if slug.endswith("-v0"):
        slug = slug[:-3]
    return slug.replace("visual-", "visual-")


def resolve_gas_artifacts(env_name: str, seed: int, root: str | os.PathLike[str]) -> GASArtifactSet:
    base = _env_dir(env_name, root, seed)
    tdr_dir = base / "tdr"
    policy_dir = base / "policy"
    graph_dir = base / "graph"
    features_dir = base / "features"
    tdr = find_latest_tdr_checkpoint(tdr_dir)
    policy = find_latest_policy_checkpoint(policy_dir)
    keygraph = find_keygraph(graph_dir)
    embeddings = features_dir / "dataset_embeddings.npy"
    return GASArtifactSet(
        env_name=env_name,
        seed=seed,
        root=base,
        tdr_dir=tdr_dir,
        policy_dir=policy_dir,
        graph_dir=graph_dir,
        features_dir=features_dir,
        tdr_checkpoint=tdr,
        policy_checkpoint=policy,
        keygraph=keygraph,
        dataset_embeddings=embeddings if embeddings.exists() else None,
    )


def _latest_params(paths: Iterable[Path]) -> Optional[Path]:
    candidates = []
    for p in paths:
        name = p.name
        try:
            step = int(name.split("params_")[-1].split(".")[0])
        except Exception:
            step = -1
        candidates.append((step, p.stat().st_mtime, p))
    if not candidates:
        return None
    return sorted(candidates)[-1][2]


def find_latest_tdr_checkpoint(path: str | os.PathLike[str]) -> Optional[Path]:
    p = Path(path)
    if p.is_file() and p.name.startswith("params_"):
        return p
    return _latest_params(p.glob("**/params_*.pkl")) if p.exists() else None


def find_latest_policy_checkpoint(path: str | os.PathLike[str]) -> Optional[Path]:
    return find_latest_tdr_checkpoint(path)


def find_keygraph(path: str | os.PathLike[str]) -> Optional[Path]:
    p = Path(path)
    if p.is_file() and p.name.endswith(".pkl"):
        return p
    if not p.exists():
        return None
    for name in ("keygraph.pkl", "keygraph_0.pkl"):
        hits = sorted(p.glob(f"**/{name}"))
        if hits:
            return hits[-1]
    hits = sorted(p.glob("**/*keygraph*.pkl"))
    return hits[-1] if hits else None


def file_sha256(path: str | os.PathLike[str], block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def _hf_endpoints() -> list[str]:
    configured = os.environ.get("BARS_HF_ENDPOINTS")
    if configured:
        raw = [x.strip() for x in configured.split(",") if x.strip()]
    else:
        explicit = os.environ.get("HF_ENDPOINT") or os.environ.get("BARS_HF_ENDPOINT")
        raw = [explicit] if explicit else []
        raw.extend(DEFAULT_HF_ENDPOINTS)
    out: list[str] = []
    seen = set()
    for endpoint in raw:
        if not endpoint:
            continue
        normalized = endpoint.rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out or list(DEFAULT_HF_ENDPOINTS)


def _hf_url(endpoint: str, slug: str, name: str) -> str:
    return f"{endpoint.rstrip('/')}/{REPO_ID}/resolve/main/{slug}/{name}"


def _download_file(url: str, dst: Path, timeout: int = 15) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if os.environ.get("BARS_DOWNLOAD_WITH_CURL", "1") != "0" and shutil.which("curl"):
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--connect-timeout",
            str(timeout),
            "--speed-time",
            str(int(os.environ.get("BARS_DOWNLOAD_SPEED_TIME", "60"))),
            "--speed-limit",
            str(int(os.environ.get("BARS_DOWNLOAD_SPEED_LIMIT", "1024"))),
            "-o",
            str(tmp),
            url,
        ]
        if tmp.exists() and tmp.stat().st_size > 0:
            cmd[1:1] = ["-C", "-"]
        try:
            subprocess.run(cmd, check=True)
            tmp.replace(dst)
            return True
        except Exception:
            pass
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dst)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _download_hf_file(slug: str, name: str, dst: Path) -> bool:
    for endpoint in _hf_endpoints():
        url = _hf_url(endpoint, slug, name)
        print(f"[gas_artifacts] download {slug}/{name} via {endpoint}", file=sys.stderr)
        if _download_file(url, dst):
            return True
    return False


def download_official_gas_if_available(env_name: str, root: str | os.PathLike[str]) -> bool:
    """Download official GAS graph/policy if the HF repo has this environment.

    The official checkpoint stores TDR and actor in one params file. For the
    standard BARS layout we copy the same file to both `tdr/` and `policy/`.
    """
    seed = 0
    artifacts = resolve_gas_artifacts(env_name, seed, root)
    if artifacts.complete:
        return True
    slug = env_to_hf_slug(env_name)
    if slug not in OFFICIAL_PRETRAINED_SLUGS:
        return False
    params_names = ["params_1000000.pkl", "params_500000.pkl"]
    if "kitchen" in slug or slug.startswith("visual-"):
        params_names = ["params_500000.pkl", "params_1000000.pkl"]

    graph_ok = artifacts.keygraph is not None
    if not graph_ok:
        graph_ok = _download_hf_file(slug, "keygraph.pkl", artifacts.graph_dir / "keygraph.pkl")
    policy_path = None
    if artifacts.policy_checkpoint is not None:
        policy_path = artifacts.policy_checkpoint
    else:
        for name in params_names:
            dst = artifacts.policy_dir / name
            if _download_hf_file(slug, name, dst):
                policy_path = dst
                break
    if graph_ok and policy_path is not None:
        artifacts.tdr_dir.mkdir(parents=True, exist_ok=True)
        tdr_dst = artifacts.tdr_dir / policy_path.name
        if not tdr_dst.exists():
            shutil.copy2(policy_path, tdr_dst)
        _write_manifest(resolve_gas_artifacts(env_name, seed, root), source="huggingface")
        return True
    return False


def _gas_script_env(gpu: int | str) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MUJOCO_GL", choose_mujoco_gl())
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("BARS_USE_TENSORBOARD", "1")
    env.setdefault("TENSORBOARD_LOGDIR", "runs_stage22_tensorboard")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def choose_mujoco_gl() -> str:
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    probe = "import mujoco; print('ok')"
    for gl in ("egl", "osmesa", ""):
        env = os.environ.copy()
        if gl:
            env["MUJOCO_GL"] = gl
        else:
            env.pop("MUJOCO_GL", None)
        try:
            subprocess.run([sys.executable, "-c", probe], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=True)
            return gl or "glfw"
        except Exception:
            continue
    return "osmesa"


def gas_agent_flag_args(env_name: str) -> list[str]:
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


def _run(cmd: list[str], cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")


def _copy_latest(src_root: Path, dst_dir: Path, kind: str) -> Optional[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    src = find_keygraph(src_root) if kind == "graph" else find_latest_policy_checkpoint(src_root)
    if src is None:
        return None
    dst = dst_dir / ("keygraph.pkl" if kind == "graph" else src.name)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def _write_manifest(artifacts: GASArtifactSet, source: str) -> None:
    artifacts.root.mkdir(parents=True, exist_ok=True)
    manifest = artifacts.to_dict()
    manifest["source"] = source
    for key in ("tdr_checkpoint", "policy_checkpoint", "keygraph", "dataset_embeddings"):
        value = manifest.get(key)
        if value and Path(value).exists():
            manifest[f"{key}_sha256"] = file_sha256(value)
    with open(artifacts.root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def train_gas_backbone_if_missing(
    env_name: str,
    seed: int,
    gpu: int | str,
    gas_repo_path: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    quick: bool = False,
    full: bool = False,
) -> GASArtifactSet:
    artifacts = resolve_gas_artifacts(env_name, seed, artifact_root)
    if artifacts.complete:
        return artifacts

    gas_repo = Path(gas_repo_path).resolve()
    if not (gas_repo / "pretrain_tdr.py").exists():
        raise FileNotFoundError(f"GAS repo not found at {gas_repo}")

    raw_root = artifacts.root.resolve() / "_raw_gas"
    logs = artifacts.root.resolve() / "logs"
    run_group = f"stage22_{env_to_hf_slug(env_name)}_seed{seed}"
    steps = 1_000_000 if full else (100_000 if quick else 1_000_000)
    save_interval = min(100_000, steps)
    common = [
        "--env_name",
        env_name,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--run_group",
        run_group,
    ] + gas_agent_flag_args(env_name)
    env = _gas_script_env(gpu)

    if artifacts.tdr_checkpoint is None:
        _run(
            [
                sys.executable,
                "pretrain_tdr.py",
                "--run_tdr_project",
                "Stage22_GAS_TDR",
                "--save_tdr_dir",
                str(raw_root / "tdr"),
                "--train_steps",
                str(steps),
                "--log_interval",
                "5000",
                "--save_interval",
                str(save_interval),
            ]
            + common,
            cwd=gas_repo,
            env=env,
            log_path=logs / "pretrain_tdr.log",
        )
        _copy_latest(raw_root / "tdr", artifacts.tdr_dir, "params")
        artifacts = resolve_gas_artifacts(env_name, seed, artifact_root)

    if artifacts.keygraph is None:
        if artifacts.tdr_checkpoint is None:
            raise RuntimeError("Cannot construct GAS keygraph without a TDR checkpoint")
        _run(
            [
                sys.executable,
                "construct_graph.py",
                "--save_graph_dir",
                str(raw_root / "graph"),
                "--te_threshold",
                "0.99",
                "--tdr_path",
                str(artifacts.tdr_checkpoint.resolve()),
            ]
            + common,
            cwd=gas_repo,
            env=env,
            log_path=logs / "construct_graph.log",
        )
        _copy_latest(raw_root / "graph", artifacts.graph_dir, "graph")
        artifacts = resolve_gas_artifacts(env_name, seed, artifact_root)

    if artifacts.policy_checkpoint is None:
        if artifacts.tdr_checkpoint is None:
            raise RuntimeError("Cannot train GAS policy without a TDR checkpoint")
        _run(
            [
                sys.executable,
                "train_policy.py",
                "--run_policy_project",
                "Stage22_GAS_Policy",
                "--save_policy_dir",
                str(raw_root / "policy"),
                "--train_steps",
                str(steps),
                "--log_interval",
                "5000",
                "--save_interval",
                str(save_interval),
                "--tdr_path",
                str(artifacts.tdr_checkpoint.resolve()),
            ]
            + common,
            cwd=gas_repo,
            env=env,
            log_path=logs / "train_policy.log",
        )
        _copy_latest(raw_root / "policy", artifacts.policy_dir, "params")
        artifacts = resolve_gas_artifacts(env_name, seed, artifact_root)

    if artifacts.complete:
        _write_manifest(artifacts, source="trained")
    return artifacts
