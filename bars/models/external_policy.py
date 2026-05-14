from __future__ import annotations

"""Config-driven external policy adapter for HIQL/GAS/other strong backbones.

The adapter intentionally does not guess checkpoint formats.  It supports exact
source reuse by importing a user-specified factory/callable from the cloned
external repository.  This keeps BARS independent while allowing strong
backbone policies to be used without rewriting their implementations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import importlib
import os
import sys

import numpy as np


def _expand_path(value: str | None) -> str | None:
    if value is None:
        return None
    value = os.path.expandvars(str(value))
    return str(Path(value).expanduser()) if value else value


def _import_from_path(repo_path: str | None, dotted: str):
    repo_path = _expand_path(repo_path)
    if repo_path:
        p = str(Path(repo_path).expanduser().resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    module_name, attr = dotted.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


@dataclass
class ExternalPolicyAdapter:
    policy: Any
    act_method: str = 'act'
    obs_normalized: bool = False
    goal_normalized: bool = False
    action_clip: bool = True

    def eval(self):
        if hasattr(self.policy, 'eval'):
            self.policy.eval()
        return self

    def _maybe_norm(self, x: np.ndarray, normalizer, enabled: bool) -> np.ndarray:
        return normalizer.encode(x[None])[0] if enabled and normalizer is not None else x

    def embed(self, obs_np: np.ndarray) -> np.ndarray:
        """Return a backbone embedding for online graph-node lookup when available."""
        obs_np = np.asarray(obs_np, dtype=np.float32)
        for name in ['embed', 'encode', 'encode_obs', 'get_phi']:
            method = getattr(self.policy, name, None)
            if method is None:
                continue
            errors = []
            for call in [lambda: method(obs_np), lambda: method(obs_np[None])]:
                try:
                    z = call()
                    if isinstance(z, tuple):
                        z = z[0]
                    z = np.asarray(z, dtype=np.float32)
                    if z.ndim > 1:
                        z = z[0]
                    return z.astype(np.float32)
                except Exception as exc:
                    errors.append(type(exc).__name__)
        raise AttributeError('External policy does not expose an embedding method.')

    def act(self, obs_np: np.ndarray, goal_np: np.ndarray, obs_normalizer, action_low: Optional[np.ndarray] = None, action_high: Optional[np.ndarray] = None, device: str = 'cuda') -> np.ndarray:
        obs_np = np.asarray(obs_np, dtype=np.float32)
        goal_np = np.asarray(goal_np, dtype=np.float32)
        obs = self._maybe_norm(obs_np, obs_normalizer, self.obs_normalized)
        goal = self._maybe_norm(goal_np, obs_normalizer, self.goal_normalized)
        method = getattr(self.policy, self.act_method, None)
        if method is None:
            # Common JAX-style names.
            for name in ['sample_actions', 'get_action', 'predict', 'forward']:
                method = getattr(self.policy, name, None)
                if method is not None:
                    break
        if method is None:
            raise AttributeError(f'External policy has no act-like method: {self.act_method}')
        # Try common signatures in order.
        errors = []
        for call in [
            lambda: method(obs, goal),
            lambda: method(obs_np, goal_np),
            lambda: method(observations=obs[None], goals=goal[None]),
            lambda: method(obs=obs, goal=goal),
            lambda: method({'observations': obs[None], 'goals': goal[None]}),
        ]:
            try:
                action = call()
                if isinstance(action, tuple):
                    action = action[0]
                action = np.asarray(action, dtype=np.float32)
                if action.ndim > 1:
                    action = action[0]
                if self.action_clip and action_low is not None and action_high is not None:
                    action = np.clip(action, action_low, action_high)
                return action.astype(np.float32)
            except Exception as exc:
                errors.append(type(exc).__name__)
        raise RuntimeError(f'Could not call external policy action method; tried signatures failed with {errors}')


def build_external_policy_from_config(cfg: dict, dataset, device=None) -> ExternalPolicyAdapter:
    pcfg = cfg.get('policy', {})
    ecfg = cfg.get('external_policy', pcfg.get('external', {}))
    repo_path = _expand_path(ecfg.get('repo_path'))
    factory_path = ecfg.get('factory') or ecfg.get('factory_path')
    object_path = ecfg.get('object') or ecfg.get('object_path')
    checkpoint_path = _expand_path(ecfg.get('checkpoint_path'))
    kwargs = dict(ecfg.get('kwargs', {}))
    kwargs.setdefault('obs_dim', dataset.obs_dim)
    kwargs.setdefault('action_dim', dataset.action_dim)
    kwargs.setdefault('device', str(device) if device is not None else 'cuda')
    if checkpoint_path is not None:
        kwargs.setdefault('checkpoint_path', checkpoint_path)
    if factory_path:
        factory = _import_from_path(repo_path, factory_path)
        policy = factory(**kwargs)
    elif object_path:
        cls = _import_from_path(repo_path, object_path)
        policy = cls(**kwargs)
    else:
        raise ValueError('External policy requires external_policy.factory or external_policy.object')
    if checkpoint_path is not None and hasattr(policy, 'load'):
        policy.load(checkpoint_path)
    elif checkpoint_path is not None and hasattr(policy, 'load_checkpoint'):
        policy.load_checkpoint(checkpoint_path)
    return ExternalPolicyAdapter(
        policy=policy,
        act_method=str(ecfg.get('act_method', 'act')),
        obs_normalized=bool(ecfg.get('obs_normalized', False)),
        goal_normalized=bool(ecfg.get('goal_normalized', False)),
        action_clip=bool(ecfg.get('action_clip', True)),
    )
