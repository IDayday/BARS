from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bars.external.gas_artifacts import (
    OFFICIAL_PRETRAINED_SLUGS,
    env_to_hf_slug,
    gas_agent_flag_args,
    resolve_gas_artifacts,
)
from fetch_public_baseline_targets import MAX_EPISODE_STEPS, TARGETS_PP, lower_bound_pp


ROUND_ID = "003"
PRIMARY_QUESTION = (
    "Can we certify a public-quality GAS backbone using official/full-budget artifacts, "
    "then certify the BARS adapter against the official GAS evaluation loop?"
)
SECONDARY_QUESTION = (
    "If medium official artifacts remain unavailable, can certification shift to official-artifact "
    "OGBench tasks without weakening the research claim?"
)
OFFICIAL_REPO_URL = "https://huggingface.co/qortmdgh4141/GAS"
OFFICIAL_TREE_URL = f"{OFFICIAL_REPO_URL}/tree/main"
AUDIT_ENVS = [
    "antmaze-medium-stitch-v0",
    "antmaze-medium-navigate-v0",
    "antmaze-large-stitch-v0",
    "antmaze-large-navigate-v0",
    "antmaze-giant-stitch-v0",
    "antmaze-giant-navigate-v0",
    "antmaze-large-explore-v0",
    "scene-play-v0",
    "kitchen-partial-v0",
]
CERTIFICATION_PRIORITY = [
    "antmaze-giant-stitch-v0",
    "antmaze-large-explore-v0",
    "scene-play-v0",
    "antmaze-giant-navigate-v0",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def split_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def split_int_csv(value: str) -> list[int]:
    return [int(x) for x in split_csv(value)]


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def checkpoint_step(path: str | Path | None) -> int | None:
    if not path:
        return None
    match = re.search(r"params_(\d+)\.pkl", Path(path).name)
    return int(match.group(1)) if match else None


def env_state_or_visual(env: str) -> str:
    return "visual" if env.startswith("visual-") else "state"


def official_policy_name(env: str) -> str:
    slug = env_to_hf_slug(env)
    if slug.startswith("visual-") or slug == "kitchen-partial":
        return "params_500000.pkl"
    return "params_1000000.pkl"


def required_train_steps(env: str) -> int:
    slug = env_to_hf_slug(env)
    if slug.startswith("visual-") or slug == "kitchen-partial":
        return 500_000
    return 1_000_000


def official_artifact_url(env: str) -> str:
    return f"{OFFICIAL_TREE_URL}/{env_to_hf_slug(env)}"


def policy_file_url(env: str) -> str:
    return f"{OFFICIAL_REPO_URL}/resolve/main/{env_to_hf_slug(env)}/{official_policy_name(env)}"


def graph_file_url(env: str) -> str:
    return f"{OFFICIAL_REPO_URL}/resolve/main/{env_to_hf_slug(env)}/keygraph.pkl"


def is_official_slug(env: str) -> bool:
    return env_to_hf_slug(env) in OFFICIAL_PRETRAINED_SLUGS


def way_steps(env: str) -> int:
    args = gas_agent_flag_args(env)
    for i, arg in enumerate(args):
        if arg == "--agent_config.way_steps" and i + 1 < len(args):
            return int(args[i + 1])
    return 8


def max_episode_steps(env: str) -> int:
    if env == "kitchen-partial-v0":
        return 280
    return int(MAX_EPISODE_STEPS.get(env) or 1000)


def public_target(env: str) -> tuple[float | None, float | None, float | None]:
    target = TARGETS_PP.get(env, {}).get("GAS")
    if not target:
        return None, None, None
    mean, std = target
    return mean, std, lower_bound_pp(mean, std)


def local_artifact_record(env: str, seed: int, root: str | Path) -> dict[str, Any]:
    artifacts = resolve_gas_artifacts(env, seed, root)
    manifest = read_json(artifacts.root / "manifest.json", {})
    policy = str(artifacts.policy_checkpoint) if artifacts.policy_checkpoint else ""
    tdr = str(artifacts.tdr_checkpoint) if artifacts.tdr_checkpoint else ""
    graph = str(artifacts.keygraph) if artifacts.keygraph else ""
    step = checkpoint_step(policy)
    local_files = list(artifacts.root.glob("**/*")) if artifacts.root.exists() else []
    partial_tmp = sorted(str(p) for p in artifacts.root.glob("**/*.tmp")) if artifacts.root.exists() else []
    source = str(manifest.get("source") or "unknown")
    if artifacts.complete and source == "huggingface" and step and step >= required_train_steps(env):
        status = "OFFICIAL_FULL_BUDGET"
    elif artifacts.complete and step is not None and step < required_train_steps(env):
        status = "LOCAL_UNDERTRAINED"
    elif artifacts.complete and step is not None and step >= required_train_steps(env):
        status = "LOCAL_FULL_BUDGET_UNCERTIFIED_SOURCE"
    elif is_official_slug(env) and local_files:
        status = "OFFICIAL_PARTIAL_LOCAL"
    elif is_official_slug(env):
        status = "OFFICIAL_AVAILABLE_NOT_LOCAL"
    else:
        status = "OFFICIAL_NOT_FOUND"
    return {
        "env": env,
        "seed": seed,
        "root": str(artifacts.root),
        "complete": artifacts.complete,
        "source": source,
        "policy_checkpoint": policy,
        "tdr_checkpoint": tdr,
        "graph_checkpoint": graph,
        "local_train_steps": step,
        "artifact_status": status,
        "partial_tmp": ";".join(partial_tmp),
    }


def certification_recommended_action(env: str, record: dict[str, Any]) -> str:
    status = str(record.get("artifact_status", ""))
    if status == "OFFICIAL_FULL_BUDGET":
        return "run_official_eval_then_adapter_certification"
    if status == "LOCAL_UNDERTRAINED":
        return "do_not_certify; acquire_official_artifact_or_run_full_budget_training"
    if status == "OFFICIAL_PARTIAL_LOCAL":
        return "resume_or_redownload_official_artifact_before_certification"
    if status == "OFFICIAL_AVAILABLE_NOT_LOCAL":
        return "download_official_artifact_before_certification"
    if status == "LOCAL_FULL_BUDGET_UNCERTIFIED_SOURCE":
        return "verify_lineage_before_certification"
    return "official_artifact_unavailable; use_full_budget_training_plan"


def official_eval_score_from_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"score": None, "score_pp": None, "num_task_ids": 0, "task_metric_columns": [], "rows": 0}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"score": None, "score_pp": None, "num_task_ids": 0, "task_metric_columns": [], "rows": 0}
    if len(df) == 0:
        return {"score": None, "score_pp": None, "num_task_ids": 0, "task_metric_columns": [], "rows": 0}
    cols = list(df.columns)
    overall_cols = [c for c in cols if c.endswith("overall_episode.success")]
    task_cols = [
        c
        for c in cols
        if c.startswith("eval/")
        and c.endswith("episode.success")
        and "overall_episode.success" not in c
    ]
    score = None
    if overall_cols:
        score = as_float(df[overall_cols[-1]].iloc[-1])
    elif task_cols:
        vals = [as_float(df[c].iloc[-1]) for c in task_cols]
        vals = [v for v in vals if v is not None]
        score = sum(vals) / len(vals) if vals else None
    elif "success" in df:
        score = as_float(df["success"].mean())
    return {
        "score": score,
        "score_pp": 100.0 * score if score is not None else None,
        "num_task_ids": len(task_cols),
        "task_metric_columns": task_cols,
        "rows": int(len(df)),
    }


def adapter_score_from_csv(path: Path, task_ids: list[int], episodes_per_task: int) -> dict[str, Any]:
    if not path.exists():
        return {"score": None, "score_pp": None, "rows": 0, "complete": False, "task_counts": {}}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"score": None, "score_pp": None, "rows": 0, "complete": False, "task_counts": {}}
    if len(df) == 0 or "success" not in df:
        return {"score": None, "score_pp": None, "rows": int(len(df)), "complete": False, "task_counts": {}}
    filtered = df[df["task_id"].astype(int).isin(task_ids)] if "task_id" in df else df
    task_counts = {}
    if "task_id" in filtered:
        task_counts = {str(int(k)): int(v) for k, v in filtered.groupby(filtered["task_id"].astype(int)).size().items()}
    expected = len(task_ids) * episodes_per_task
    score = float(filtered["success"].mean()) if len(filtered) else None
    complete = len(filtered) >= expected and all(task_counts.get(str(t), 0) >= episodes_per_task for t in task_ids)
    return {
        "score": score,
        "score_pp": 100.0 * score if score is not None else None,
        "rows": int(len(filtered)),
        "complete": complete,
        "task_counts": task_counts,
        "steps_mean": float(filtered["steps"].mean()) if "steps" in filtered and len(filtered) else "",
    }


def normalize_task_id_list(value: str | list[int]) -> str:
    if isinstance(value, list):
        return ",".join(str(int(x)) for x in value)
    parts = split_int_csv(str(value))
    return ",".join(str(x) for x in parts)
