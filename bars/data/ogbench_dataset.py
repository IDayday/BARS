from __future__ import annotations

"""Best-effort OGBench / GAS-v0 dataset loader.

The implementation intentionally avoids hard-pinning a single OGBench API
version.  It first asks OGBench for an environment and datasets, then normalizes
common dataset dict layouts into BARS' OfflineDataset structure.
"""

import fcntl
import json
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .trajectories import OfflineDataset, TrajectorySlice

_DEFAULT_DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"
_FALLBACK_DATASET_URL = "http://rail.eecs.berkeley.edu/datasets/ogbench"


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


def _dataset_base_urls() -> list[str]:
    raw = os.environ.get("OGBENCH_DATASET_ENDPOINTS") or os.environ.get("OGBENCH_DATASET_URL") or ""
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        cn_raw = os.environ.get("BARS_OGBENCH_CN_ENDPOINTS", "")
        urls = [u.strip().rstrip("/") for u in cn_raw.split(",") if u.strip()]
        urls.extend([_DEFAULT_DATASET_URL, _FALLBACK_DATASET_URL])
    return list(dict.fromkeys(urls))


def _shared_dataset_dirs(dataset_dir: Path) -> list[Path]:
    raw = os.environ.get("OGBENCH_DATASET_SHARED_DIRS", "")
    roots = [Path(os.path.expandvars(os.path.expanduser(x.strip()))) for x in raw.split(",") if x.strip()]
    shared_root = os.environ.get("BARS_SHARED_DATASET_ROOT")
    if shared_root:
        roots.append(Path(os.path.expandvars(os.path.expanduser(shared_root))) / "ogbench")
    roots = [p for p in roots if p != dataset_dir]
    out: list[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _materialize_from_shared(file_name: str, dst: Path, dataset_dir: Path) -> bool:
    for root in _shared_dataset_dirs(dataset_dir):
        src = root / file_name
        if not _npz_is_ready(src):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        _write_valid_marker(dst)
        print(f"Using shared OGBench dataset {src} -> {dst}", flush=True)
        return True
    return False


def _curl_download(url: str, tmp: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is not available")
    if _curl_parallel_download(url, tmp):
        return
    cmd = [
        curl,
        "-sS",
        "-fL",
        "--retry",
        os.environ.get("BARS_DOWNLOAD_RETRIES", "5"),
        "--retry-delay",
        os.environ.get("BARS_DOWNLOAD_RETRY_DELAY", "2"),
        "--connect-timeout",
        os.environ.get("BARS_DOWNLOAD_CONNECT_TIMEOUT", "30"),
        "--speed-time",
        os.environ.get("BARS_DOWNLOAD_SPEED_TIME", "60"),
        "--speed-limit",
        os.environ.get("BARS_DOWNLOAD_SPEED_LIMIT", "1024"),
        "-o",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)


def _aria2_download(url: str, tmp: Path) -> None:
    aria2 = shutil.which("aria2c")
    if aria2 is None:
        raise RuntimeError("aria2c is not available")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        aria2,
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--continue=true",
        "--max-tries",
        os.environ.get("BARS_DOWNLOAD_RETRIES", "5"),
        "--retry-wait",
        os.environ.get("BARS_DOWNLOAD_RETRY_DELAY", "2"),
        "--connect-timeout",
        os.environ.get("BARS_DOWNLOAD_CONNECT_TIMEOUT", "30"),
        "--timeout",
        os.environ.get("BARS_DOWNLOAD_TIMEOUT", "120"),
        "--min-split-size",
        os.environ.get("BARS_ARIA2_MIN_SPLIT_SIZE", "4M"),
        "--split",
        os.environ.get("BARS_ARIA2_SPLIT", "16"),
        "--max-connection-per-server",
        os.environ.get("BARS_ARIA2_CONNECTIONS", "16"),
        "--file-allocation",
        os.environ.get("BARS_ARIA2_FILE_ALLOCATION", "none"),
        "--dir",
        str(tmp.parent),
        "--out",
        tmp.name,
        url,
    ]
    subprocess.run(cmd, check=True)


def _curl_headers(url: str) -> dict[str, str]:
    curl = shutil.which("curl")
    if curl is None:
        return {}
    proc = subprocess.run(
        [curl, "-sIL", "--connect-timeout", os.environ.get("BARS_DOWNLOAD_CONNECT_TIMEOUT", "30"), url],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return {}
    headers: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return headers


def _curl_base_args() -> list[str]:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is not available")
    return [
        curl,
        "-sS",
        "-fL",
        "--retry",
        os.environ.get("BARS_DOWNLOAD_RETRIES", "5"),
        "--retry-delay",
        os.environ.get("BARS_DOWNLOAD_RETRY_DELAY", "2"),
        "--connect-timeout",
        os.environ.get("BARS_DOWNLOAD_CONNECT_TIMEOUT", "30"),
        "--speed-time",
        os.environ.get("BARS_DOWNLOAD_SPEED_TIME", "60"),
        "--speed-limit",
        os.environ.get("BARS_DOWNLOAD_SPEED_LIMIT", "1024"),
    ]


def _curl_parallel_download(url: str, tmp: Path) -> bool:
    chunks = int(os.environ.get("BARS_DOWNLOAD_PARALLEL_CHUNKS", "8"))
    min_size = int(os.environ.get("BARS_DOWNLOAD_PARALLEL_MIN_BYTES", str(64 * 1024 * 1024)))
    if chunks <= 1:
        return False
    headers = _curl_headers(url)
    if headers.get("accept-ranges", "").lower() != "bytes":
        return False
    try:
        size = int(headers.get("content-length", "0"))
    except ValueError:
        return False
    if size < min_size:
        return False
    chunks = max(1, min(chunks, size // (8 * 1024 * 1024) or 1))
    if chunks <= 1:
        return False
    parts_dir = tmp.with_name(f"{tmp.name}.parts")
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges: list[tuple[int, int]] = []
    block = (size + chunks - 1) // chunks
    for idx in range(chunks):
        start = idx * block
        if start >= size:
            break
        end = min(size - 1, start + block - 1)
        ranges.append((start, end))
    print(f"Parallel curl download {tmp.name}: {len(ranges)} ranges, {size} bytes", flush=True)
    base = _curl_base_args()
    parts: list[tuple[Path, int, int, int]] = []
    for idx, (start, end) in enumerate(ranges):
        parts.append((parts_dir / f"part-{idx:03d}", start, end, end - start + 1))
    procs: list[subprocess.Popen[Any]] = []
    try:
        rounds = int(os.environ.get("BARS_DOWNLOAD_PARALLEL_ROUNDS", "3"))
        for round_idx in range(1, rounds + 1):
            pending = [(part, start, end, expected) for part, start, end, expected in parts if not part.exists() or part.stat().st_size != expected]
            if not pending:
                break
            if round_idx > 1:
                print(f"Retrying {len(pending)} failed OGBench ranges for {tmp.name} ({round_idx}/{rounds})", flush=True)
            procs = []
            for part, start, end, _ in pending:
                cmd = base + ["-r", f"{start}-{end}", "-o", str(part), url]
                procs.append(subprocess.Popen(cmd))
            for proc in procs:
                proc.wait()
        pending = [(part, expected) for part, _, _, expected in parts if not part.exists() or part.stat().st_size != expected]
        if pending:
            detail = ", ".join(f"{part.name}:{part.stat().st_size if part.exists() else 0}/{expected}" for part, expected in pending[:4])
            raise RuntimeError(f"parallel range download failed for {tmp.name}: {detail}")
        with open(tmp, "wb") as out:
            for part, _, _, _ in parts:
                with open(part, "rb") as fh:
                    shutil.copyfileobj(fh, out, length=1024 * 1024)
        return tmp.stat().st_size == size
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        shutil.rmtree(parts_dir, ignore_errors=True)


def _urllib_download(url: str, tmp: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _download_atomic(file_name: str, dst: Path, retries: int = 3) -> None:
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    errors: list[str] = []
    for base_url in _dataset_base_urls():
        url = f"{base_url}/{file_name}"
        for attempt in range(1, retries + 1):
            try:
                if tmp.exists():
                    tmp.unlink()
                print(
                    f"Downloading OGBench dataset {dst.name} "
                    f"(attempt {attempt}/{retries}) from {url}",
                    flush=True,
                )
                try:
                    if os.environ.get("BARS_DOWNLOAD_WITH_ARIA2", "1") != "0":
                        _aria2_download(url, tmp)
                    else:
                        _curl_download(url, tmp)
                except Exception:
                    try:
                        _curl_download(url, tmp)
                    except Exception:
                        _urllib_download(url, tmp)
                if not _npz_is_valid(tmp):
                    raise IOError(f"Downloaded file is not a valid npz: {tmp}")
                os.replace(tmp, dst)
                _write_valid_marker(dst)
                return
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                if tmp.exists():
                    tmp.unlink()
                if attempt < retries:
                    time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"Failed to download {file_name}; tried {len(errors)} attempts: {' | '.join(errors[-3:])}")


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
            if not _materialize_from_shared(file_name, dst, root):
                if os.environ.get("BARS_OGBENCH_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
                    raise FileNotFoundError(
                        f"Missing OGBench dataset {dst}; BARS_OGBENCH_OFFLINE=1 prevents network download. "
                        f"Copy {file_name} into {root} or one of OGBENCH_DATASET_SHARED_DIRS."
                    )
                _download_atomic(file_name, dst)
            paths.append(str(dst))
        return paths[0], paths[1]


def _call_ogbench(
    env_name: str,
    dataset_path: str | None = None,
    dataset_dir: str | None = None,
    compact_dataset: bool = False,
):
    import ogbench  # type: ignore
    if dataset_path:
        dataset_path = os.path.expandvars(os.path.expanduser(str(dataset_path)))
        if os.path.isdir(dataset_path):
            dataset_dir = dataset_path
            dataset_path = None
    if dataset_dir is None:
        dataset_dir = os.environ.get("OGBENCH_DATASET_DIR") or _default_dataset_dir()
    if dataset_dir:
        dataset_dir = os.path.expandvars(os.path.expanduser(str(dataset_dir)))
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
