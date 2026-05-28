from __future__ import annotations

import sys
import os
from pathlib import Path


class _DummyRun:
    project = "OGBench"


def _setup_official_imports(repo_root: Path) -> None:
    tmd_root = repo_root / "external_src" / "tmd-release"
    impls = tmd_root / "impls"
    for path in (str(impls), str(tmd_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    _setup_official_imports(repo_root)
    os.chdir(repo_root / "external_src" / "tmd-release" / "impls")

    import wandb
    import main as official_main
    from absl import app

    original_make_env_and_datasets = official_main.make_env_and_datasets

    def make_env_and_datasets(dataset_name, frame_stack=None):
        dataset_dir = os.environ.get("OGBENCH_DATASET_DIR") or os.environ.get("BARS_TMD_TEST_DATASET_ROOT")
        if not dataset_dir:
            return original_make_env_and_datasets(dataset_name, frame_stack=frame_stack)
        import ogbench
        from utils.datasets import Dataset
        from utils.env_utils import FrameStackWrapper

        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name,
            dataset_dir=dataset_dir,
            compact_dataset=True,
        )
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
        if frame_stack is not None:
            env = FrameStackWrapper(env, frame_stack)
        env.reset()
        return env, train_dataset, val_dataset

    def fake_setup_wandb(*args, **kwargs):
        wandb.run = _DummyRun()
        return wandb.run

    official_main.make_env_and_datasets = make_env_and_datasets
    official_main.setup_wandb = fake_setup_wandb
    official_main.wandb.log = lambda *args, **kwargs: None
    app.run(official_main.main)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
