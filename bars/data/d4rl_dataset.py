from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from .trajectories import OfflineDataset, TrajectorySlice

def _split_raw_trajectories(terminals: np.ndarray, timeouts: np.ndarray):
    starts, ends, start = [], [], 0; n = int(len(terminals)); done = np.logical_or(terminals.astype(bool), timeouts.astype(bool))
    for i in range(n):
        if done[i]:
            if i + 1 - start >= 2: starts.append(start); ends.append(i + 1)
            start = i + 1
    if n - start >= 2: starts.append(start); ends.append(n)
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)

def load_d4rl_dataset(env_name: str, dataset_limit: int = 0) -> Tuple[Any, OfflineDataset]:
    import gym, d4rl  # noqa: F401
    env = gym.make(env_name); raw: Dict[str, np.ndarray] = env.get_dataset()
    observations = raw['observations'].astype(np.float32); actions = raw['actions'].astype(np.float32)
    terminals = raw.get('terminals', np.zeros(len(observations), dtype=np.bool_)); timeouts = raw.get('timeouts', np.zeros(len(observations), dtype=np.bool_))
    if dataset_limit and dataset_limit > 0:
        observations, actions, terminals, timeouts = observations[:dataset_limit], actions[:dataset_limit], terminals[:dataset_limit], timeouts[:dataset_limit]
    starts, ends = _split_raw_trajectories(terminals, timeouts)
    obs_list=[]; act_list=[]; next_list=[]; traj_ids=[]; timesteps=[]; slices=[]; cursor=0
    for tid, (s, e) in enumerate(zip(starts, ends)):
        if e - s < 2: continue
        tr_obs = observations[s:e-1]; tr_next = observations[s+1:e]; tr_act = actions[s:e-1]; length = len(tr_obs)
        obs_list.append(tr_obs); next_list.append(tr_next); act_list.append(tr_act); traj_ids.append(np.full(length, tid, dtype=np.int32)); timesteps.append(np.arange(length, dtype=np.int32))
        slices.append(TrajectorySlice(tid, cursor, cursor + length, int(s), int(e))); cursor += length
    if not obs_list: raise RuntimeError(f'Could not build trajectories from {env_name}.')
    return env, OfflineDataset(np.concatenate(obs_list,0), np.concatenate(act_list,0), np.concatenate(next_list,0), np.concatenate(traj_ids,0), np.concatenate(timesteps,0), slices, env_name)
