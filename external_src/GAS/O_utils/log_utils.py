import os
import json
import tempfile
import numpy as np
import ml_collections
import absl.flags as flags

from datetime import datetime
from PIL import Image, ImageEnhance


class _CompatMedia:
    def __init__(self, data=None, **kwargs):
        self.data = data
        self.kwargs = kwargs


class _CompatSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _NoOpRun:
    def log(self, metrics, step=None):
        return None

    def finish(self):
        return None


class _NoOpWandbCompat:
    Image = _CompatMedia
    Video = _CompatMedia
    Histogram = _CompatMedia
    Settings = _CompatSettings

    def __init__(self):
        self.run = _NoOpRun()

    def init(self, **kwargs):
        print("[GAS logging] WandB disabled; keeping CSV/stdout only.")
        return self.run

    def log(self, metrics, step=None):
        self.run.log(metrics, step=step)


class _TensorBoardRun:
    def __init__(self, log_dir, config=None):
        self.log_dir = os.path.abspath(log_dir)
        self.config = config or {}
        self._writer = None
        self._disabled_reason = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=self.log_dir)
        except Exception as exc:
            print(f"[GAS logging] TensorBoard unavailable ({exc!r}); keeping CSV/stdout only.")

    def _disable_writer(self, exc):
        if self._writer is None:
            return
        writer = self._writer
        self._writer = None
        try:
            writer.close()
        except Exception:
            pass
        if self._disabled_reason is None:
            self._disabled_reason = repr(exc)
            print(f"[GAS logging] TensorBoard writer disabled after error ({exc!r}); keeping CSV/stdout only.")

    def log(self, metrics, step=None):
        if self._writer is None:
            return
        step = int(step or 0)
        writer = self._writer
        try:
            for key, value in metrics.items():
                if isinstance(value, _CompatMedia):
                    continue
                try:
                    arr = np.asarray(value)
                except Exception:
                    continue
                if arr.shape == ():
                    writer.add_scalar(key, float(arr), step)
                elif arr.size > 0:
                    writer.add_histogram(key, arr.reshape(-1), step)
            writer.flush()
        except Exception as exc:
            self._disable_writer(exc)

    def finish(self):
        if self._writer is not None:
            writer = self._writer
            self._writer = None
            try:
                writer.flush()
                writer.close()
            except Exception as exc:
                if self._disabled_reason is None:
                    self._disabled_reason = repr(exc)
                    print(f"[GAS logging] TensorBoard writer disabled after error ({exc!r}); keeping CSV/stdout only.")


class _TensorBoardWandbCompat:
    Image = _CompatMedia
    Video = _CompatMedia
    Histogram = _CompatMedia
    Settings = _CompatSettings

    def __init__(self):
        self.run = None

    def init(self, **kwargs):
        root = os.environ.get("TENSORBOARD_LOGDIR", "runs_stage22_tensorboard")
        project = kwargs.get("project") or "gas"
        group = kwargs.get("group") or "default"
        name = kwargs.get("name") or "run"
        log_dir = os.path.join(root, project, group, name)
        os.makedirs(log_dir, exist_ok=True)
        self.run = _TensorBoardRun(log_dir=log_dir, config=kwargs.get("config"))
        print(f"[GAS logging] Using TensorBoard log dir: {log_dir}")
        return self.run

    def log(self, metrics, step=None):
        if self.run is not None:
            self.run.log(metrics, step=step)


def _load_logging_backend():
    use_tb = os.environ.get("BARS_USE_TENSORBOARD", "").lower() in {"1", "true", "yes"}
    disable_wandb = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
    if use_tb:
        return _TensorBoardWandbCompat()
    if disable_wandb:
        return _NoOpWandbCompat()
    try:
        import wandb as _wandb

        return _wandb
    except Exception as exc:
        print(f"[GAS logging] wandb unavailable ({exc!r}); keeping CSV/stdout only.")
        return _NoOpWandbCompat()


wandb = _load_logging_backend()


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""
    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            
            
def get_exp_name(seed):
    """Return the experiment name."""
    g_start_time = int(datetime.now().timestamp())
    g_start_time = datetime.fromtimestamp(g_start_time).strftime('%Y-%m-%d_%H-%M-%S')  
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    exp_name += f'_{g_start_time}'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    return exp_name


def setup_save_directory(exp_name, env_name, run_group, save_dir):
    """Create the experiment save directory and store run configuration."""
    save_dir = os.path.join(save_dir, run_group, env_name + "_" + exp_name)
    os.makedirs(save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, indent=4)
    return save_dir


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {}
    for k in flags.FLAGS:
        if '.' in k:
            continue
        try:
            flag_dict[k] = getattr(flags.FLAGS, k)
        except Exception:
            continue
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(project, group, name, entity=None, mode='online'):
    """Set up the GAS logging backend.

    Stage22 disables external WandB monitoring by default and routes the
    small scalar stream through a WandB-compatible TensorBoard shim.  The
    public function name is kept so the official GAS scripts remain intact.
    """
    env_mode = os.environ.get('WANDB_MODE', '').lower()
    if os.environ.get('WANDB_DISABLED', '').lower() in {'1', 'true', 'yes'} or env_mode in {'disabled', 'offline'}:
        mode = 'disabled'
    tags = [group]
    wandb_output_dir = tempfile.mkdtemp()
    name = name.format(**get_flag_dict())
    unique_identifier = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{name}_{unique_identifier}"
    init_kwargs = dict(entity=entity, config=get_flag_dict(), 
                       project=project, tags=tags, group=group, dir=wandb_output_dir, name=name, id=experiment_id, resume=None,
                       settings=wandb.Settings(start_method='thread', _disable_stats=False,), mode=mode, save_code=True,)
    run = wandb.init(**init_kwargs)
    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]
    _, t, h, w, c = v.shape
    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols
    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))
    return v
            
            
def get_wandb_video(renders=None, n_cols=None, fps=15):
    """
    Return a Weights & Biases video.
    It takes a list of videos and reshapes them into a single video with the specified number of columns.
    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8
        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)
        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)
        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)
    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)
    return wandb.Video(renders, fps=fps, format='mp4')
