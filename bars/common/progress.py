from __future__ import annotations

import os
from typing import Dict, Optional


def tqdm_kwargs(cfg: Optional[Dict] = None) -> Dict:
    cfg = cfg or {}
    pcfg = cfg.get("progress", {}) if isinstance(cfg, dict) else {}
    env_disable = str(os.environ.get("BARS_DISABLE_TQDM", "")).strip().lower()
    disable = False
    if "enabled" in pcfg:
        disable = not bool(pcfg.get("enabled"))
    if env_disable in {"1", "true", "yes", "on"}:
        disable = True
    elif env_disable in {"0", "false", "no", "off"}:
        disable = False
    mininterval = float(os.environ.get("BARS_TQDM_MININTERVAL", pcfg.get("mininterval", 5.0)))
    return {"dynamic_ncols": not disable, "disable": disable, "mininterval": mininterval}
