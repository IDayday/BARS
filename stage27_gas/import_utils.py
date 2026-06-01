from __future__ import annotations

import importlib
from typing import Any


def import_object(spec: str) -> Any:
    """Import object from 'module:object' or 'module.object'."""
    if ":" in spec:
        module_name, obj_name = spec.split(":", 1)
    else:
        module_name, obj_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj
