import glob
import os


def split_checkpoint_path(path):
    """Return (directory, epoch) for a GAS-style params_<epoch>.pkl checkpoint."""
    if path is None:
        raise ValueError("Checkpoint path is required.")
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "params_*.pkl")))
        if not candidates:
            raise FileNotFoundError(f"No params_*.pkl checkpoints found in {path}")
        path = candidates[-1]
    restore_dir = os.path.dirname(path)
    basename = os.path.basename(path)
    epoch = basename.split("_")[-1].split(".")[0]
    return restore_dir, epoch
