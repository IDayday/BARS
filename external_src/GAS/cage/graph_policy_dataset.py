from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np


@dataclass(frozen=True)
class GraphPolicyPair:
    """One graph-induced state-goal pair in TDR/phi space."""

    env_name: str
    seed: int
    s_ref: np.ndarray
    g_ref: np.ndarray
    d_phi: float
    state_ref_s: dict[str, Any] | None = None
    state_ref_g: dict[str, Any] | None = None
    probeable: bool = False
    source: str = "graph_planned"
    weight: float = 1.0
    metadata: dict[str, Any] | None = None

    @property
    def skill(self) -> np.ndarray:
        delta = self.g_ref - self.s_ref
        norm = np.linalg.norm(delta) + 1e-10
        return delta / norm


class GraphPolicyDataset:
    """
    Minimal dataset wrapper for future graph-induced policy training.

    GP0 does not train a policy. This object only stores graph-induced
    state-goal pairs in the same phi-space interface used by GAS low-level
    actor goals, so a later milestone can mix them into policy training.
    """

    def __init__(self, pairs: Iterable[GraphPolicyPair]):
        self.pairs = list(pairs)
        self.size = len(self.pairs)

    @classmethod
    def from_jsonl(cls, path: str | Path, *, require_vectors: bool = True) -> "GraphPolicyDataset":
        pairs: list[GraphPolicyPair] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if require_vectors and ("s_ref" not in row or "g_ref" not in row):
                    raise ValueError(f"Missing s_ref/g_ref vectors in {path}")
                if "s_ref" not in row or "g_ref" not in row:
                    continue
                metadata = {k: v for k, v in row.items() if k not in {"env_name", "seed", "s_ref", "g_ref", "d_phi", "state_ref_s", "state_ref_g"}}
                pairs.append(
                    GraphPolicyPair(
                        env_name=str(row.get("env_name", "")),
                        seed=int(row.get("seed", -1)),
                        s_ref=np.asarray(row["s_ref"], dtype=np.float32),
                        g_ref=np.asarray(row["g_ref"], dtype=np.float32),
                        d_phi=float(row.get("d_phi", np.linalg.norm(np.asarray(row["g_ref"]) - np.asarray(row["s_ref"])))),
                        state_ref_s=row.get("state_ref_s"),
                        state_ref_g=row.get("state_ref_g"),
                        probeable=bool(row.get("probeable", False)),
                        source=str(row.get("source", "graph_planned")),
                        weight=float(row.get("weight", 1.0)),
                        metadata=metadata,
                    )
                )
        return cls(pairs)

    def sample(self, batch_size: int, *, replace: bool | None = None, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty GraphPolicyDataset")
        rng = rng or np.random.default_rng()
        if replace is None:
            replace = batch_size > self.size
        idxs = rng.choice(self.size, size=batch_size, replace=replace)
        selected = [self.pairs[int(i)] for i in idxs]
        s_ref = np.stack([p.s_ref for p in selected]).astype(np.float32)
        g_ref = np.stack([p.g_ref for p in selected]).astype(np.float32)
        skills = np.stack([p.skill for p in selected]).astype(np.float32)
        d_phi = np.asarray([p.d_phi for p in selected], dtype=np.float32)
        weights = np.asarray([p.weight for p in selected], dtype=np.float32)
        probeable = np.asarray([p.probeable for p in selected], dtype=bool)
        return {
            "phi_obs": s_ref,
            "phi_actor_goals": g_ref,
            "actor_skills": skills,
            "d_phi": d_phi,
            "weights": weights,
            "probeable": probeable,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        if self.size == 0:
            return {
                "phi_obs": np.empty((0, 0), dtype=np.float32),
                "phi_actor_goals": np.empty((0, 0), dtype=np.float32),
                "actor_skills": np.empty((0, 0), dtype=np.float32),
                "d_phi": np.empty((0,), dtype=np.float32),
                "weights": np.empty((0,), dtype=np.float32),
                "probeable": np.empty((0,), dtype=bool),
            }
        return self.sample(self.size, replace=False, rng=np.random.default_rng(0))
