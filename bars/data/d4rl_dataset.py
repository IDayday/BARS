from __future__ import annotations
import os
import time
import urllib.request
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import h5py
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

@contextmanager
def _file_lock(lock_path: str):
    import fcntl
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, 'w', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def _is_valid_hdf5_dataset(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    try:
        with h5py.File(path, 'r') as f:
            return all(k in f for k in ('observations', 'actions', 'rewards', 'terminals'))
    except Exception:
        return False

def _download_atomic(dataset_url: str, dataset_path: str) -> None:
    tmp_path = f'{dataset_path}.tmp.{os.getpid()}'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    try:
        urllib.request.urlretrieve(dataset_url, tmp_path)
        if not _is_valid_hdf5_dataset(tmp_path):
            raise IOError(f'Downloaded dataset is invalid or truncated: {tmp_path}')
        os.replace(tmp_path, dataset_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def prefetch_d4rl_dataset(env_name: str, retries: int = 3) -> str:
    import gym, d4rl  # noqa: F401
    env = gym.make(env_name)
    dataset_url = getattr(env, 'dataset_url', None) or getattr(env, '_dataset_url', None)
    dataset_path = getattr(env, 'dataset_filepath', None)
    if not dataset_url or not dataset_path:
        raise ValueError(f'Could not resolve D4RL dataset path for {env_name}.')
    lock_path = f'{dataset_path}.lock'
    with _file_lock(lock_path):
        if _is_valid_hdf5_dataset(dataset_path):
            return dataset_path
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                print(f'Preparing D4RL dataset for {env_name}: {dataset_path} (attempt {attempt}/{retries})')
                _download_atomic(dataset_url, dataset_path)
                if _is_valid_hdf5_dataset(dataset_path):
                    return dataset_path
                raise IOError(f'Validation failed after download: {dataset_path}')
            except Exception as exc:
                last_error = exc
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
                if attempt < retries:
                    time.sleep(min(5.0, float(attempt)))
        raise IOError(f'Failed to prepare D4RL dataset for {env_name}: {last_error}') from last_error

def load_d4rl_dataset(env_name: str, dataset_limit: int = 0) -> Tuple[Any, OfflineDataset]:
    import gym, d4rl  # noqa: F401
    env = gym.make(env_name); h5path = prefetch_d4rl_dataset(env_name); raw: Dict[str, np.ndarray] = env.get_dataset(h5path=h5path)
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
