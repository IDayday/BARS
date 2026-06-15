#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.support_keygraph import (  # noqa: E402
    load_edge_scores_csv,
    load_keygraph_pickle,
    patch_gas_keygraph_with_support,
    write_patch_outputs,
)


def _load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _cfg(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Patch an official GAS keygraph using BARS/GAS edge support scores. "
            "The output keygraph keeps the GAS policy interface unchanged and only "
            "changes graph edge weights or prunes unsupported non-goal edges."
        )
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--keygraph-path", default="")
    parser.add_argument("--edge-scores-csv", default="")
    parser.add_argument("--out-keygraph-path", default="")
    parser.add_argument("--mode", choices=["annotate", "penalize", "prune", "penalize_and_prune"], default="")
    parser.add_argument("--support-column", default="")
    parser.add_argument("--min-support", type=float, default=None)
    parser.add_argument("--unsupported-penalty", type=float, default=None)
    parser.add_argument("--risk-column", default="")
    parser.add_argument("--risk-weight", type=float, default=None)
    parser.add_argument("--missing-score-policy", choices=["protect", "penalize", "prune"], default="")
    parser.add_argument("--protect-goal-edges", type=int, default=None)
    parser.add_argument("--no-recompute-task-paths", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    keygraph_path = args.keygraph_path or _cfg(cfg, "keygraph_path")
    edge_scores_csv = args.edge_scores_csv or _cfg(cfg, "edge_scores_csv")
    out_keygraph_path = args.out_keygraph_path or _cfg(cfg, "out_keygraph_path")
    if not keygraph_path:
        raise ValueError("--keygraph-path or config.keygraph_path is required")
    if not edge_scores_csv:
        raise ValueError("--edge-scores-csv or config.edge_scores_csv is required")
    if not out_keygraph_path:
        raise ValueError("--out-keygraph-path or config.out_keygraph_path is required")

    mode = args.mode or _cfg(cfg, "patch.mode", "penalize")
    support_column = args.support_column or _cfg(cfg, "patch.support_column", "local_support")
    min_support = args.min_support if args.min_support is not None else float(_cfg(cfg, "patch.min_support", 1.0))
    unsupported_penalty = (
        args.unsupported_penalty
        if args.unsupported_penalty is not None
        else _cfg(cfg, "patch.unsupported_penalty", None)
    )
    risk_column = args.risk_column if args.risk_column else _cfg(cfg, "patch.risk_column", "r_exec")
    if str(risk_column).lower() in {"", "none", "null"}:
        risk_column = None
    risk_weight = args.risk_weight if args.risk_weight is not None else float(_cfg(cfg, "patch.risk_weight", 0.0))
    missing_score_policy = args.missing_score_policy or _cfg(cfg, "patch.missing_score_policy", "protect")
    protect_goal_edges = (
        bool(args.protect_goal_edges)
        if args.protect_goal_edges is not None
        else bool(_cfg(cfg, "patch.protect_goal_edges", True))
    )
    recompute_task_paths = not args.no_recompute_task_paths and bool(_cfg(cfg, "patch.recompute_task_paths", True))

    key_graph = load_keygraph_pickle(keygraph_path)
    edge_scores = load_edge_scores_csv(edge_scores_csv)
    result = patch_gas_keygraph_with_support(
        key_graph,
        edge_scores,
        mode=mode,
        support_column=support_column,
        min_support=min_support,
        unsupported_penalty=unsupported_penalty,
        risk_column=risk_column,
        risk_weight=risk_weight,
        missing_score_policy=missing_score_policy,
        protect_goal_edges=protect_goal_edges,
        recompute_task_paths=recompute_task_paths,
    )
    paths = write_patch_outputs(result, out_keygraph_path)
    print(json.dumps({"paths": paths, "summary": result.summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
