"""Template adapter for using official GAS low-level policy inside BARS.

Official GAS checkpoints contain JAX policy/value/TDR parameters.  Because exact
checkpoint structure may change, this adapter is a template.  Use the official
GAS evaluation code to load `params_*.pkl`, then expose an act(obs, goal) method
compatible with BARS.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle
import numpy as np


@dataclass
class GASPolicyWrapper:
    agent: Any

    def act(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if hasattr(self.agent, 'act'):
            return np.asarray(self.agent.act(obs, goal), dtype=np.float32)
        if hasattr(self.agent, 'sample_actions'):
            out = self.agent.sample_actions(observations=obs[None], goals=goal[None])
            if isinstance(out, tuple):
                out = out[0]
            out = np.asarray(out, dtype=np.float32)
            return out[0] if out.ndim > 1 else out
        raise AttributeError('GAS agent has no act/sample_actions; edit routeb_gas_policy_adapter_template.py')


def build_policy(obs_dim: int, action_dim: int, checkpoint_path: str, device: str = 'cuda', **kwargs) -> GASPolicyWrapper:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f'GAS checkpoint not found: {ckpt}')
    try:
        with open(ckpt, 'rb') as f:
            payload = pickle.load(f)
    except Exception as exc:
        raise RuntimeError('Replace this template with the official GAS checkpoint loader.') from exc
    agent = payload.get('agent', payload.get('policy', payload)) if isinstance(payload, dict) else payload
    return GASPolicyWrapper(agent=agent)
