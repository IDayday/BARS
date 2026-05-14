from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import pickle

import flax
import jax
import numpy as np


def _as_plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): value[k] for k in value}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value)


@dataclass
class HIQLPolicyWrapper:
    agent: Any
    rng: Any
    low_dim_goals: bool = False
    discrete: int = 0
    temperature: float = 0.0

    def eval(self):
        return self

    def act(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)[None]
        goal = np.asarray(goal, dtype=np.float32)[None]
        self.rng, key = jax.random.split(self.rng)
        out = self.agent.sample_actions(
            observations=obs,
            goals=goal,
            low_dim_goals=bool(self.low_dim_goals),
            seed=key,
            temperature=float(self.temperature),
            discrete=int(self.discrete),
        )
        out = np.asarray(out, dtype=np.float32)
        return out[0] if out.ndim > 1 else out


def build_policy(obs_dim: int, action_dim: int, checkpoint_path: str, device: str = "cuda", **kwargs) -> HIQLPolicyWrapper:
    from src.agents import hiql as learner

    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"HIQL checkpoint not found: {ckpt}")
    with open(ckpt, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "agent" not in payload:
        raise ValueError("Expected official HIQL checkpoint payload with keys including 'agent'.")

    config = _as_plain_dict(payload.get("config"))
    ex_obs = np.zeros((1, int(obs_dim)), dtype=np.float32)
    ex_actions = np.zeros((1, int(action_dim)), dtype=np.float32)
    seed = int(kwargs.get("seed", 0))
    agent = learner.create_learner(
        seed,
        ex_obs,
        ex_actions,
        visual=int(kwargs.get("visual", 0)),
        encoder=str(kwargs.get("encoder", "impala")),
        discrete=int(kwargs.get("discrete", 0)),
        use_layer_norm=int(kwargs.get("use_layer_norm", 1)),
        rep_type=str(kwargs.get("rep_type", "concat")),
        **config,
    )
    agent = flax.serialization.from_state_dict(agent, payload["agent"])
    return HIQLPolicyWrapper(
        agent=agent,
        rng=jax.random.PRNGKey(seed),
        low_dim_goals=bool(kwargs.get("low_dim_goals", False)),
        discrete=int(kwargs.get("discrete", 0)),
        temperature=float(kwargs.get("temperature", 0.0)),
    )
