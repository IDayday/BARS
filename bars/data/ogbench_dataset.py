from __future__ import annotations

"""Best-effort OGBench / GAS-v0 dataset loader.

The implementation intentionally avoids hard-pinning a single OGBench API
version.  It first asks OGBench for an environment and datasets, then normalizes
common dataset dict layouts into BARS' OfflineDataset structure.
"""

import fcntl
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .trajectories import OfflineDataset, TrajectorySlice

_DEFAULT_DATASET_URL = "http://rail.eecs.berkeley.edu/datasets/ogbench"


def _default_dataset_dir() -> str:
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / "_data" / "ogbench")


def _npz_is_valid(path: str | os.PathLike[str]) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except Exception:
        return False


def _valid_marker_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.valid.json")


def _file_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _has_valid_marker(path: Path) -> bool:
    marker = _valid_marker_path(path)
    if not path.exists() or not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == _file_fingerprint(path)
    except Exception:
        return False


def _write_valid_marker(path: Path) -> None:
    _valid_marker_path(path).write_text(json.dumps(_file_fingerprint(path), sort_keys=True), encoding="utf-8")


def _npz_is_ready(path: Path) -> bool:
    if _has_valid_marker(path):
        return True
    if _npz_is_valid(path):
        _write_valid_marker(path)
        return True
    return False


class _FileLock:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+", encoding="utf-8")
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
        return False


def _download_atomic(url: str, dst: Path, retries: int = 5) -> None:
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    for attempt in range(1, retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            print(f"Downloading OGBench dataset {dst.name} (attempt {attempt}/{retries}) from {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if not _npz_is_valid(tmp):
                raise IOError(f"Downloaded file is not a valid npz: {tmp}")
            os.replace(tmp, dst)
            _write_valid_marker(dst)
            return
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt >= retries:
                raise
            time.sleep(min(60, 5 * attempt))


def _dataset_file_names(env_name: str) -> tuple[str, str]:
    # This matches OGBench's ordinary goal-conditioned antmaze naming.  The
    # special singletask/oraclerep variants drop helper tokens before download.
    parts = env_name.split("-")
    dataset_name = env_name
    if "singletask" in parts:
        pos = parts.index("singletask")
        dataset_name = "-".join(parts[:pos] + parts[-1:])
    elif "oraclerep" in parts:
        dataset_name = "-".join(parts[:-2] + parts[-1:])
    return f"{dataset_name}.npz", f"{dataset_name}-val.npz"


def ensure_ogbench_dataset_files(env_name: str, dataset_dir: str | None = None) -> tuple[str, str]:
    dataset_dir = os.path.expandvars(os.path.expanduser(dataset_dir or os.environ.get("OGBENCH_DATASET_DIR") or _default_dataset_dir()))
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get("OGBENCH_DATASET_URL", _DEFAULT_DATASET_URL).rstrip("/")
    train_name, val_name = _dataset_file_names(env_name)
    lock_path = root / f"{train_name}.lock"
    with _FileLock(lock_path):
        paths = []
        for file_name in (train_name, val_name):
            dst = root / file_name
            if _npz_is_ready(dst):
                paths.append(str(dst))
                continue
            if dst.exists():
                bad = dst.with_name(f"{dst.name}.bad-{int(time.time())}")
                os.replace(dst, bad)
            for stale in root.glob(f"{dst.name}.tmp.*"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            _download_atomic(f"{base_url}/{file_name}", dst)
            paths.append(str(dst))
        return paths[0], paths[1]


def _call_ogbench(
    env_name: str,
    dataset_path: str | None = None,
    dataset_dir: str | None = None,
    compact_dataset: bool = False,
):
    if dataset_path:
        dataset_path = os.path.expandvars(os.path.expanduser(str(dataset_path)))
        if os.path.isdir(dataset_path):
            dataset_dir = dataset_path
            dataset_path = None
    if dataset_dir is None:
        dataset_dir = os.environ.get("OGBENCH_DATASET_DIR") or _default_dataset_dir()
    if dataset_dir:
        dataset_dir = os.path.expandvars(os.path.expanduser(str(dataset_dir)))
    try:
        import ogbench  # type: ignore
    except ModuleNotFoundError:
        return _call_local_ogbench_npz(env_name, dataset_path=dataset_path, dataset_dir=dataset_dir)
    if dataset_path is None and dataset_dir:
        ensure_ogbench_dataset_files(env_name, dataset_dir)
    # Common recent API.
    if hasattr(ogbench, "make_env_and_datasets"):
        kwargs = {"compact_dataset": compact_dataset}
        if dataset_path:
            kwargs["dataset_path"] = dataset_path
        elif dataset_dir:
            kwargs["dataset_dir"] = dataset_dir
        out = ogbench.make_env_and_datasets(env_name, **kwargs)
        if isinstance(out, tuple):
            if len(out) == 3:
                return out[0], out[1], out[2]
            if len(out) == 2:
                return out[0], out[1], None
        raise RuntimeError(f"Unexpected ogbench.make_env_and_datasets return for {env_name}: {type(out)}")
    # Fallback API names occasionally used by wrappers.
    for name in ["make_env_and_dataset", "make_dataset"]:
        fn = getattr(ogbench, name, None)
        if fn is None:
            continue
        kwargs = {}
        if dataset_path:
            kwargs["dataset_path"] = dataset_path
        elif dataset_dir:
            kwargs["dataset_dir"] = dataset_dir
        out = fn(env_name, **kwargs)
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[1], out[2] if len(out) > 2 else None
    raise RuntimeError("Could not find a supported OGBench dataset API. Expected make_env_and_datasets.")


def _call_local_ogbench_npz(
    env_name: str,
    dataset_path: str | None = None,
    dataset_dir: str | None = None,
):
    """Load OGBench train/validation npz files without constructing a MuJoCo env.

    Graph-only audits and cache materialization need only offline arrays.  This
    fallback keeps those workflows independent of the optional `ogbench`,
    `dm_control`, and MuJoCo runtime stack when dataset files already exist.
    """
    if dataset_path:
        train_path = Path(os.path.expandvars(os.path.expanduser(str(dataset_path))))
        if train_path.is_dir():
            dataset_dir = str(train_path)
            train_path = None
    else:
        train_path = None
    val_path = None
    if train_path is None:
        if dataset_dir is None:
            dataset_dir = _default_dataset_dir()
        root = Path(os.path.expandvars(os.path.expanduser(str(dataset_dir))))
        train_name, val_name = _dataset_file_names(env_name)
        train_path = root / train_name
        val_path = root / val_name
    if not train_path.exists():
        raise ModuleNotFoundError(
            "No module named 'ogbench', and local OGBench npz was not found: "
            f"{train_path}"
        )
    train = {k: np.asarray(v) for k, v in np.load(train_path).items()}
    val = None
    if val_path is not None and val_path.exists():
        val = {k: np.asarray(v) for k, v in np.load(val_path).items()}
    return None, train, val


def _as_array(dataset: Dict[str, Any], *keys: str, required: bool = True) -> np.ndarray | None:
    for key in keys:
        if key in dataset:
            return np.asarray(dataset[key])
    if required:
        raise KeyError(f"Dataset missing any of keys {keys}")
    return None


def _split_by_done(n: int, terminals: np.ndarray | None, timeouts: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if terminals is None:
        terminals = np.zeros(n, dtype=bool)
    if timeouts is None:
        timeouts = np.zeros(n, dtype=bool)
    done = np.asarray(terminals).astype(bool) | np.asarray(timeouts).astype(bool)
    starts, ends, start = [], [], 0
    for i in range(n):
        if done[i]:
            if i + 1 - start >= 2:
                starts.append(start); ends.append(i + 1)
            start = i + 1
    if n - start >= 2:
        starts.append(start); ends.append(n)
    if not starts:
        starts = [0]; ends = [n]
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _normalise_dataset(raw: Dict[str, Any], env_name: str, dataset_limit: int = 0) -> OfflineDataset:
    obs = _as_array(raw, "observations", "obs", "states").astype(np.float32)
    actions = _as_array(raw, "actions", "action").astype(np.float32)
    next_obs = _as_array(raw, "next_observations", "next_obs", "next_states", required=False)
    terminals = _as_array(raw, "terminals", "dones", "done", required=False)
    timeouts = _as_array(raw, "timeouts", "truncations", "truncated", required=False)
    if dataset_limit and dataset_limit > 0:
        obs = obs[:dataset_limit]
        actions = actions[:dataset_limit]
        if next_obs is not None: next_obs = next_obs[:dataset_limit]
        if terminals is not None: terminals = terminals[:dataset_limit]
        if timeouts is not None: timeouts = timeouts[:dataset_limit]
    if next_obs is None:
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
    next_obs = np.asarray(next_obs, dtype=np.float32)
    n = min(len(obs), len(actions), len(next_obs))
    obs, actions, next_obs = obs[:n], actions[:n], next_obs[:n]
    if terminals is not None: terminals = np.asarray(terminals[:n]).astype(bool)
    if timeouts is not None: timeouts = np.asarray(timeouts[:n]).astype(bool)
    starts, ends = _split_by_done(n, terminals, timeouts)
    obs_list=[]; act_list=[]; next_list=[]; traj_ids=[]; timesteps=[]; slices=[]; cursor=0
    for tid, (s, e) in enumerate(zip(starts, ends)):
        if e - s < 2:
            continue
        # If next_observations is explicit, keep all transitions except a terminal duplicate if needed.
        tr_obs = obs[s:e-1]
        tr_next = next_obs[s:e-1]
        tr_act = actions[s:e-1]
        length = len(tr_obs)
        obs_list.append(tr_obs); next_list.append(tr_next); act_list.append(tr_act)
        traj_ids.append(np.full(length, tid, dtype=np.int32)); timesteps.append(np.arange(length, dtype=np.int32))
        slices.append(TrajectorySlice(tid, cursor, cursor + length, int(s), int(e))); cursor += length
    if not obs_list:
        raise RuntimeError(f"Could not build OGBench trajectories from {env_name}.")
    return OfflineDataset(
        np.concatenate(obs_list, 0), np.concatenate(act_list, 0), np.concatenate(next_list, 0),
        np.concatenate(traj_ids, 0), np.concatenate(timesteps, 0), slices, env_name,
    )


def load_ogbench_dataset(
    env_name: str,
    dataset_limit: int = 0,
    dataset_path: str | None = None,
    dataset_dir: str | None = None,
    split: str = "train",
    compact_dataset: bool = False,
) -> Tuple[Any, OfflineDataset]:
    env, train_dataset, val_dataset = _call_ogbench(
        env_name,
        dataset_path=dataset_path,
        dataset_dir=dataset_dir,
        compact_dataset=compact_dataset,
    )
    raw_dataset = val_dataset if str(split).lower() in {"val", "valid", "validation"} and val_dataset is not None else train_dataset
    # OGBench commonly returns a FrozenDict/dict with train data as the second output.
    if hasattr(raw_dataset, "unfreeze"):
        raw_dataset = raw_dataset.unfreeze()
    if not isinstance(raw_dataset, dict):
        try:
            raw_dataset = dict(raw_dataset)
        except Exception as exc:
            raise TypeError(f"Unsupported OGBench dataset type {type(raw_dataset)}") from exc
    return env, _normalise_dataset(raw_dataset, env_name, dataset_limit=dataset_limit)


def summarize_ogbench_dataset(dataset: OfflineDataset) -> Dict[str, Any]:
    return {
        "env_name": dataset.env_name,
        "size": int(dataset.size),
        "obs_dim": int(dataset.obs_dim),
        "action_dim": int(dataset.action_dim),
        "num_trajectories": int(len(dataset.traj_slices)),
    }
