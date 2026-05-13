from __future__ import annotations
import os
from typing import Any, Dict, Optional

def save_checkpoint(path: str, model, optimizer=None, **extra: Any) -> None:
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {'model': model.state_dict(), **extra}
    if optimizer is not None: payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, path)

def load_checkpoint(path: str, model, optimizer=None, map_location: Optional[str] = None) -> Dict[str, Any]:
    import torch
    try:
        payload = torch.load(path, map_location=map_location or 'cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        payload = torch.load(path, map_location=map_location or 'cpu')
    model.load_state_dict(payload['model'])
    if optimizer is not None and 'optimizer' in payload: optimizer.load_state_dict(payload['optimizer'])
    return payload
