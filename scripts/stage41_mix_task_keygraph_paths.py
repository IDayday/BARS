#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from bars.gas_bars.support_keygraph import load_keygraph_pickle, save_keygraph_pickle


def _parse_name_path(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty method name in {value!r}")
        if name in out:
            raise ValueError(f"duplicate method name: {name}")
        out[name] = Path(path)
    return out


def _parse_task_method(values: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected TASK_ID=METHOD_NAME, got {value!r}")
        task_raw, method = value.split("=", 1)
        task_id = int(task_raw)
        method = method.strip()
        if not method:
            raise ValueError(f"empty method name in {value!r}")
        out[task_id] = method
    return out


def _lookup_mapping_key(mapping: dict[Any, Any], task_id: int) -> Any:
    if task_id in mapping:
        return task_id
    task_str = str(task_id)
    if task_str in mapping:
        return task_str
    for key in mapping:
        try:
            if int(key) == int(task_id):
                return key
        except Exception:
            continue
    raise KeyError(task_id)


def mix_task_keygraph_paths(
    base_keygraph: Any,
    method_keygraphs: dict[str, Any],
    task_method_map: dict[int, str],
) -> tuple[Any, list[dict[str, Any]]]:
    """Return a GAS keygraph with task cached paths copied from method keygraphs.

    GAS rollout reads ``task_paths_dict`` and ``task_paths_dist_dict`` from the
    keygraph for every task.  Keeping the base graph and nodes fixed while
    replacing these per-task caches lets us test task-conditioned graph routing
    without changing the low-level policy interface.
    """
    out = copy.deepcopy(base_keygraph)
    if not hasattr(out, "task_paths_dict") or not hasattr(out, "task_paths_dist_dict"):
        raise ValueError("base keygraph is missing GAS task path caches")

    summary: list[dict[str, Any]] = []
    for task_id, method_name in sorted(task_method_map.items()):
        if method_name not in method_keygraphs:
            raise KeyError(f"task {task_id} references unknown method {method_name!r}")
        src = method_keygraphs[method_name]
        if not hasattr(src, "task_paths_dict") or not hasattr(src, "task_paths_dist_dict"):
            raise ValueError(f"method {method_name!r} keygraph is missing GAS task path caches")

        src_paths_key = _lookup_mapping_key(src.task_paths_dict, task_id)
        src_dists_key = _lookup_mapping_key(src.task_paths_dist_dict, task_id)
        dst_paths_key = _lookup_mapping_key(out.task_paths_dict, task_id)
        dst_dists_key = _lookup_mapping_key(out.task_paths_dist_dict, task_id)

        paths = copy.deepcopy(src.task_paths_dict[src_paths_key])
        dists = copy.deepcopy(src.task_paths_dist_dict[src_dists_key])
        out.task_paths_dict[dst_paths_key] = paths
        out.task_paths_dist_dict[dst_dists_key] = dists
        summary.append(
            {
                "task_id": int(task_id),
                "method": method_name,
                "num_cached_start_nodes": int(len(paths)),
                "mean_cached_distance": float(sum(map(float, dists.values())) / max(1, len(dists))),
                "source_task_paths_key": str(src_paths_key),
                "destination_task_paths_key": str(dst_paths_key),
            }
        )

    setattr(out, "bars_task_method_map", {str(k): v for k, v in sorted(task_method_map.items())})
    setattr(out, "bars_task_path_mixer", "stage41_task_conditioned_cached_paths")
    return out, summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-keygraph", required=True, type=Path)
    parser.add_argument(
        "--method-keygraph",
        action="append",
        default=[],
        help="Repeat as NAME=PATH. Include original=... if a task should keep original paths.",
    )
    parser.add_argument(
        "--task-method",
        action="append",
        default=[],
        help="Repeat as TASK_ID=METHOD_NAME, for example 1=hybrid_w5p00_forward.",
    )
    parser.add_argument("--output-keygraph", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    method_paths = _parse_name_path(args.method_keygraph)
    task_method_map = _parse_task_method(args.task_method)
    if not task_method_map:
        raise ValueError("at least one --task-method is required")

    base = load_keygraph_pickle(args.base_keygraph)
    methods = {name: load_keygraph_pickle(path) for name, path in method_paths.items()}
    mixed, rows = mix_task_keygraph_paths(base, methods, task_method_map)

    save_keygraph_pickle(mixed, args.output_keygraph)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_keygraph": str(args.base_keygraph),
        "output_keygraph": str(args.output_keygraph),
        "method_keygraphs": {name: str(path) for name, path in method_paths.items()},
        "task_method_map": {str(k): v for k, v in sorted(task_method_map.items())},
        "graph_source": "base_keygraph",
        "task_path_rows": rows,
        "note": "Only GAS task cached paths are mixed; low-level policy and node embeddings are unchanged.",
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
