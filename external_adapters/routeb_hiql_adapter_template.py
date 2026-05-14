"""Template adapter for using official HIQL policy inside BARS.

Copy this file to the root of the cloned HIQL repository or keep it in BARS and
set `external_policy.repo_path` so it can import official HIQL modules.

You must adapt the marked section to the exact checkpoint structure produced by
your HIQL run.  This template intentionally does not guess JAX/Flax checkpoint
internals because tiny deviations in HIQL loading can invalidate SOTA alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle
import numpy as np


@dataclass
class HIQLPolicyWrapper:
    agent: Any
    rng: Any = None
    deterministic: bool = True

    def embed(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        for name in ["get_phi", "encode", "encode_obs"]:
            method = getattr(self.agent, name, None)
            if method is None:
                continue
            try:
                z = np.asarray(method(obs[None]), dtype=np.float32)
                return z[0] if z.ndim > 1 else z
            except Exception:
                pass
        raise AttributeError("HIQL agent does not expose a reusable embedding method.")

    def act(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        # Official HIQL/OGBench agents often expose sample_actions or get_action.
        # Adapt this function to exactly match your cloned repo version.
        if hasattr(self.agent, 'sample_actions'):
            try:
                out = self.agent.sample_actions(observations=obs[None], goals=goal[None], seed=self.rng, temperature=0.0)
            except TypeError:
                out = self.agent.sample_actions(obs[None], goal[None])
        elif hasattr(self.agent, 'get_action'):
            out = self.agent.get_action(obs, goal)
        else:
            raise AttributeError('HIQL agent has no sample_actions/get_action; edit routeb_hiql_adapter_template.py')
        if isinstance(out, tuple):
            out = out[0]
        out = np.asarray(out, dtype=np.float32)
        return out[0] if out.ndim > 1 else out


def build_policy(obs_dim: int, action_dim: int, checkpoint_path: str, device: str = 'cuda', **kwargs) -> HIQLPolicyWrapper:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f'HIQL checkpoint not found: {ckpt}')

    # ------------------------------------------------------------------
    # EDIT THIS SECTION FOR THE EXACT OFFICIAL HIQL CHECKPOINT FORMAT.
    # Typical options:
    #   - import the official checkpoint loader and agent class;
    #   - restore the Flax TrainState/agent from checkpoint;
    #   - return HIQLPolicyWrapper(agent=agent, rng=...)
    # ------------------------------------------------------------------
    try:
        with open(ckpt, 'rb') as f:
            payload = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            'This template could not load the HIQL checkpoint with pickle. '
            'Replace build_policy with the official HIQL checkpoint loader.'
        ) from exc

    # Some local experiments save the whole agent under common keys.
    agent = payload.get('agent', payload.get('policy', payload)) if isinstance(payload, dict) else payload
    return HIQLPolicyWrapper(agent=agent)
