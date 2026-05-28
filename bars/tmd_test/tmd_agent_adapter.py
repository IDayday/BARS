from __future__ import annotations

import importlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_raw_ogbench_npz


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_tmd_paths(tmd_repo: str | os.PathLike[str] | None = None) -> Path:
    root = Path(tmd_repo) if tmd_repo is not None else _repo_root() / "external_src" / "tmd-release"
    impls = root / "impls"
    for p in (str(impls), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def _infer_env_name(path: Path) -> str | None:
    for part in reversed(path.parts):
        if "_sd" in part:
            return part.split("_sd", 1)[0]
        if part.endswith("-v0"):
            return part
    return None


def _load_agent_config(checkpoint_path: Path, config: dict[str, Any] | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    flags_path = checkpoint_path.parent / "flags.json"
    if flags_path.exists():
        with open(flags_path, "r", encoding="utf-8") as f:
            flags = json.load(f)
        cfg.update(flags.get("agent_config") or flags.get("agent") or {})
    if config:
        if "agent_config" in config and isinstance(config["agent_config"], dict):
            cfg.update(config["agent_config"])
        else:
            cfg.update({k: v for k, v in config.items() if k not in {"dataset_root", "env_name", "tmd_repo"}})
    if not cfg:
        _add_tmd_paths(config.get("tmd_repo") if config else None)
        tmd_mod = importlib.import_module("agents.tmd")
        cfg = dict(tmd_mod.get_config())
    cfg.setdefault("agent_name", "tmd")
    cfg.setdefault("frame_stack", None)
    cfg.setdefault("encoder", None)
    return cfg


class TMDAgentAdapter:
    def __init__(self, agent: Any, config: dict[str, Any], checkpoint_path: Path, tmd_repo: Path):
        self.agent = agent
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.tmd_repo = tmd_repo
        self._rng_counter = int(config.get("seed", 0))

    @classmethod
    def restore(cls, checkpoint_path: str, config: dict[str, Any] | None = None) -> "TMDAgentAdapter":
        cfg = dict(config or {})
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"TMD checkpoint not found: {ckpt}")
        tmd_repo = _add_tmd_paths(cfg.get("tmd_repo"))
        tmd_mod = importlib.import_module("agents.tmd")
        flax_serialization = importlib.import_module("flax.serialization")
        agent_config = _load_agent_config(ckpt, cfg)
        env_name = cfg.get("env_name") or _infer_env_name(ckpt) or "antmaze-medium-stitch-v0"
        dataset_root = cfg.get("dataset_root") or os.environ.get("BARS_TMD_TEST_DATASET_ROOT")
        data = load_raw_ogbench_npz(env_name, dataset_root=dataset_root, max_observations=4)
        ex_obs = np.asarray(data["observations"][:1], dtype=np.float32)
        ex_actions = np.asarray(data["actions"][:1], dtype=np.float32)
        seed = int(cfg.get("seed", 0))
        agent = tmd_mod.TMDAgent.create(seed, ex_obs, ex_actions, agent_config)
        with open(ckpt, "rb") as f:
            load_dict = pickle.load(f)
        agent = flax_serialization.from_state_dict(agent, load_dict["agent"])
        return cls(agent=agent, config=agent_config, checkpoint_path=ckpt, tmd_repo=tmd_repo)

    def get_psi(self, observations: Any) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[None, :]
        psi = np.asarray(self.agent.network.select("psi")(obs))
        if psi.ndim == 3:
            psi = psi.mean(axis=0)
        return psi[0] if squeeze else psi

    def distance_from_embeddings(self, src_embeds: Any, dst_embeds: Any) -> np.ndarray:
        src = np.asarray(src_embeds, dtype=np.float32)
        dst = np.asarray(dst_embeds, dtype=np.float32)
        src_squeeze = src.ndim == 1
        dst_squeeze = dst.ndim == 1
        if src_squeeze:
            src = src[None, :]
        if dst_squeeze:
            dst = dst[None, :]
        import jax.numpy as jnp
        dist = np.asarray(self.agent.distance(jnp.asarray(src)[:, None, :], jnp.asarray(dst)[None, :, :]))
        if src_squeeze and dst_squeeze:
            return np.asarray(dist[0, 0])
        if src_squeeze:
            return dist[0]
        if dst_squeeze:
            return dist[:, 0]
        return dist

    def distance_obs(self, src_obs: Any, dst_obs: Any) -> np.ndarray:
        return self.distance_from_embeddings(self.get_psi(src_obs), self.get_psi(dst_obs))

    def sample_actions(self, observations: Any, goals: Any, temperature: float = 0.0, seed: int | None = None) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        goal = np.asarray(goals, dtype=np.float32)
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[None, :]
        if goal.ndim == 1:
            goal = goal[None, :]
        import jax
        key = jax.random.PRNGKey(int(seed if seed is not None else self._next_seed()))
        action = np.asarray(self.agent.sample_actions(obs, goal, seed=key, temperature=temperature))
        return action[0] if squeeze else action

    def _next_seed(self) -> int:
        self._rng_counter += 1
        return self._rng_counter
