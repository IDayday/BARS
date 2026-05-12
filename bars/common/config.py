from __future__ import annotations
import copy, json, os
from typing import Any, Dict, Iterable, Mapping

def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)

def deep_update(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def parse_value(raw: str) -> Any:
    if raw.lower() in {'true','false'}:
        return raw.lower() == 'true'
    if raw.lower() in {'none','null'}:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw

def apply_dotlist(config: Dict[str, Any], dotlist: Iterable[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    for item in dotlist:
        if '=' not in item:
            raise ValueError(f'Override must be key=value, got: {item}')
        key, value_raw = item.split('=', 1)
        value = parse_value(value_raw)
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg

def flatten_dict(d: Mapping[str, Any], prefix: str = '') -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten_dict(v, kk))
        else:
            out[kk] = v
    return out

def cfg_get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in path.split('.'):
        if not isinstance(node, Mapping) or p not in node:
            return default
        node = node[p]
    return node
