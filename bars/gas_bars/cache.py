from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional


def _stable_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def short_hash_bytes(data: bytes, n: int = 12) -> str:
    return hashlib.sha256(data).hexdigest()[:n]


def short_hash_json(data: Mapping[str, Any], n: int = 12) -> str:
    return short_hash_bytes(_stable_json(data).encode("utf-8"), n=n)


def file_hash(path: str | os.PathLike[str], n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def stage22_cache_key(
    env: str,
    seed: int,
    gas_checkpoint_hash: str,
    graph_hash: str,
    p_exec_model_hash: str = "none",
    planner_config_hash: str = "none",
) -> str:
    return short_hash_json(
        {
            "env": env,
            "seed": seed,
            "gas_checkpoint_hash": gas_checkpoint_hash,
            "graph_hash": graph_hash,
            "p_exec_model_hash": p_exec_model_hash,
            "planner_config_hash": planner_config_hash,
        },
        n=16,
    )


class Stage22Cache:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, key: str, suffix: str) -> Path:
        sub = self.root / kind / key[:2] / key
        sub.mkdir(parents=True, exist_ok=True)
        return sub / suffix

    def exists(self, kind: str, key: str, suffix: str) -> bool:
        return self.path(kind, key, suffix).exists()

    def write_json(self, kind: str, key: str, suffix: str, data: Mapping[str, Any]) -> Path:
        path = self.path(kind, key, suffix)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)
        return path

    def read_json(self, kind: str, key: str, suffix: str) -> Optional[dict[str, Any]]:
        path = self.path(kind, key, suffix)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def manifest_for(
        self,
        env: str,
        seed: int,
        gas_checkpoint: str | os.PathLike[str],
        graph_path: str | os.PathLike[str],
        p_exec_model: str | os.PathLike[str] | None = None,
        planner_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        gas_hash = file_hash(gas_checkpoint) if gas_checkpoint and Path(gas_checkpoint).exists() else "missing"
        graph_hash = file_hash(graph_path) if graph_path and Path(graph_path).exists() else "missing"
        p_hash = file_hash(p_exec_model) if p_exec_model and Path(p_exec_model).exists() else "none"
        cfg_hash = short_hash_json(planner_config or {})
        key = stage22_cache_key(env, seed, gas_hash, graph_hash, p_hash, cfg_hash)
        return {
            "key": key,
            "env": env,
            "seed": seed,
            "gas_checkpoint_hash": gas_hash,
            "graph_hash": graph_hash,
            "p_exec_model_hash": p_hash,
            "planner_config_hash": cfg_hash,
        }
