from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import importlib
import pickle

import flax
import jax
import jax.numpy as jnp
import numpy as np


def _config_to_plain_dict(value: Any) -> dict:
    if isinstance(value, Mapping):
        return {str(k): value[k] for k in value}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value)


@dataclass
class GASPolicyWrapper:
    agent: Any
    rng: Any

    def eval(self):
        return self

    def embed(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        phi = np.asarray(self.agent.get_phi(obs[None]), dtype=np.float32)
        return phi[0] if phi.ndim > 1 else phi

    def _skill_from_goal(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)[None]
        goal = np.asarray(goal, dtype=np.float32)[None]
        phi_obs = np.asarray(self.agent.get_phi(obs), dtype=np.float32)
        phi_goal = np.asarray(self.agent.get_phi(goal), dtype=np.float32)
        skill = phi_goal - phi_obs
        norm = np.linalg.norm(skill, axis=-1, keepdims=True) + 1e-8
        return (skill / norm).astype(np.float32)

    def act(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)[None]
        skill = self._skill_from_goal(obs[0], goal)
        self.rng, key = jax.random.split(self.rng)
        out = self.agent.sample_actions(observations=obs, goals=skill, temperature=1.0, seed=key)
        out = np.asarray(out, dtype=np.float32)
        return out[0] if out.ndim > 1 else out


def build_policy(obs_dim: int, action_dim: int, checkpoint_path: str, device: str = "cuda", **kwargs) -> GASPolicyWrapper:
    from M_utils.agents import agents_dict

    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"GAS checkpoint not found: {ckpt}")
    config_module_name = str(kwargs.get("config_module", "M_utils.agents.gas"))
    config_module = importlib.import_module(config_module_name)
    config = _config_to_plain_dict(config_module.get_config())
    for k, v in kwargs.get("config_overrides", {}).items():
        config[str(k)] = v
    ex_obs = np.zeros((1, int(obs_dim)), dtype=np.float32)
    ex_actions = np.zeros((1, int(action_dim)), dtype=np.float32)
    seed = int(kwargs.get("seed", 0))
    agent_class = agents_dict[config["agent_name"]]
    agent = agent_class.create(seed, ex_obs, ex_actions, config)
    with open(ckpt, "rb") as f:
        payload = pickle.load(f)
    state_dict = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    agent = flax.serialization.from_state_dict(agent, state_dict)
    return GASPolicyWrapper(agent=agent, rng=jax.random.PRNGKey(seed))
