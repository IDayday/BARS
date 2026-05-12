from __future__ import annotations
import numpy as np
from .trajectories import OfflineDataset, TrajectorySlice

def make_toy_dataset(num_traj: int = 64, length: int = 50, seed: int = 0) -> OfflineDataset:
    rng = np.random.default_rng(seed); obs=[]; acts=[]; nxt=[]; tids=[]; ts=[]; slices=[]; cursor=0
    for tid in range(num_traj):
        angle = rng.uniform(0, 2*np.pi); start = rng.normal(0,0.2,size=2); states=[]
        for t in range(length):
            frac = t / max(1,length-1); curve = np.array([np.cos(angle+frac*2.0), np.sin(angle+frac*2.0)])
            states.append(start + frac*curve + rng.normal(0,0.02,size=2))
        states = np.asarray(states, dtype=np.float32); actions = np.diff(states, axis=0).astype(np.float32); n=len(actions)
        obs.append(states[:-1]); nxt.append(states[1:]); acts.append(actions); tids.append(np.full(n,tid,dtype=np.int32)); ts.append(np.arange(n,dtype=np.int32)); slices.append(TrajectorySlice(tid,cursor,cursor+n,cursor,cursor+n+1)); cursor += n
    return OfflineDataset(np.concatenate(obs,0), np.concatenate(acts,0), np.concatenate(nxt,0), np.concatenate(tids,0), np.concatenate(ts,0), slices, 'toy')
