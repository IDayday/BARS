#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PUBLIC_SOURCE = "GAS ICML 2025 / OpenReview PDF, Table 1"
PUBLIC_SOURCE_URL = "https://openreview.net/pdf?id=73EwiOrN8W"
OFFICIAL_CODE_URL = "https://github.com/qortmdgh4141/GAS"
OFFICIAL_ARTIFACT_URL = "https://huggingface.co/qortmdgh4141/GAS/tree/main"

ALGORITHMS = ["GCBC", "GCIQL", "QRL", "CRL", "HGCBC", "HHILP", "HIQL", "GAS"]

TARGETS_PP: dict[str, dict[str, tuple[float, float]]] = {
    "antmaze-medium-navigate-v0": {
        "GCBC": (33.1, 5.6),
        "GCIQL": (74.6, 4.8),
        "QRL": (81.9, 8.2),
        "CRL": (95.3, 1.0),
        "HGCBC": (58.1, 5.5),
        "HHILP": (96.3, 0.4),
        "HIQL": (95.3, 1.3),
        "GAS": (96.3, 1.3),
    },
    "antmaze-large-navigate-v0": {
        "GCBC": (23.4, 3.2),
        "GCIQL": (32.6, 4.7),
        "QRL": (74.9, 4.4),
        "CRL": (85.5, 5.3),
        "HGCBC": (44.3, 4.1),
        "HHILP": (86.8, 3.6),
        "HIQL": (89.9, 2.2),
        "GAS": (93.2, 0.5),
    },
    "antmaze-giant-navigate-v0": {
        "GCBC": (0.0, 0.0),
        "GCIQL": (0.1, 0.4),
        "QRL": (14.3, 3.6),
        "CRL": (15.0, 5.7),
        "HGCBC": (7.2, 1.7),
        "HHILP": (53.1, 2.6),
        "HIQL": (67.3, 5.5),
        "GAS": (77.6, 2.9),
    },
    "antmaze-medium-stitch-v0": {
        "GCBC": (43.2, 7.7),
        "GCIQL": (26.6, 6.8),
        "QRL": (67.0, 10.6),
        "CRL": (57.0, 7.9),
        "HGCBC": (65.9, 5.7),
        "HHILP": (96.0, 1.2),
        "HIQL": (92.0, 2.8),
        "GAS": (98.1, 1.2),
    },
    "antmaze-large-stitch-v0": {
        "GCBC": (2.3, 3.6),
        "GCIQL": (9.6, 3.1),
        "QRL": (20.2, 1.7),
        "CRL": (14.4, 5.9),
        "HGCBC": (10.7, 5.8),
        "HHILP": (34.1, 3.0),
        "HIQL": (71.7, 4.8),
        "GAS": (96.3, 0.9),
    },
    "antmaze-giant-stitch-v0": {
        "GCBC": (0.0, 0.0),
        "GCIQL": (0.0, 0.0),
        "QRL": (0.4, 0.3),
        "CRL": (0.0, 0.0),
        "HGCBC": (0.0, 0.0),
        "HHILP": (0.0, 0.0),
        "HIQL": (1.0, 1.2),
        "GAS": (88.3, 3.6),
    },
    "antmaze-large-explore-v0": {
        "GCBC": (0.0, 0.0),
        "GCIQL": (0.6, 0.5),
        "QRL": (0.3, 1.0),
        "CRL": (0.0, 0.0),
        "HGCBC": (0.0, 0.0),
        "HHILP": (2.4, 1.9),
        "HIQL": (2.9, 4.3),
        "GAS": (94.2, 3.0),
    },
    "scene-play-v0": {
        "GCBC": (5.4, 0.9),
        "GCIQL": (50.4, 1.4),
        "QRL": (5.1, 1.7),
        "CRL": (19.2, 3.0),
        "HGCBC": (4.6, 1.3),
        "HHILP": (43.4, 5.2),
        "HIQL": (40.0, 9.6),
        "GAS": (73.6, 8.0),
    },
}

MAX_EPISODE_STEPS = {
    "antmaze-medium-navigate-v0": 1000,
    "antmaze-large-navigate-v0": 1000,
    "antmaze-giant-navigate-v0": 2000,
    "antmaze-medium-stitch-v0": 200,
    "antmaze-large-stitch-v0": 200,
    "antmaze-giant-stitch-v0": 200,
    "antmaze-large-explore-v0": 500,
    "scene-play-v0": 1000,
}

OFFICIAL_GAS_PRETRAINED_SLUGS = {
    "antmaze-giant-navigate",
    "antmaze-giant-stitch",
    "antmaze-large-explore",
    "scene-play",
    "kitchen-partial",
    "visual-antmaze-giant-navigate",
    "visual-antmaze-giant-stitch",
    "visual-antmaze-large-explore",
    "visual-scene-play",
}


def env_to_slug(env: str) -> str:
    return env[:-3] if env.endswith("-v0") else env


def lower_bound_pp(mean_pp: float, std_pp: float) -> float:
    return mean_pp - max(2.0 * std_pp, 5.0)


def gas_required_hyperparameters(env: str) -> dict[str, Any]:
    slug = env_to_slug(env)
    return {
        "encoder": "not_used",
        "discount": 0.995 if "giant" in slug else 0.99,
        "tdr_expectile": 0.999,
        "alpha": 0.01 if "explore" in slug else 1.0,
        "batch_size": 1024,
        "p_aug": 0.0,
        "way_steps": 48 if "scene" in slug else 8,
        "te_threshold": 0.99,
    }


def gas_required_train_steps(env: str) -> int:
    return 1_000_000


def public_eval_protocol(env: str) -> dict[str, Any]:
    return {
        "num_task_goals": 5,
        "rollouts_per_goal": 50,
        "num_seeds": 4,
        "max_episode_steps": MAX_EPISODE_STEPS.get(env),
        "success_threshold": 2,
        "goal_sampling": "five test-time goals from OGBench task set",
    }


def target_rows(envs: list[str] | None = None, algorithms: list[str] | None = None) -> list[dict[str, Any]]:
    env_list = envs or list(TARGETS_PP)
    alg_list = algorithms or ALGORITHMS
    rows: list[dict[str, Any]] = []
    for env in env_list:
        targets = TARGETS_PP.get(env, {})
        for algorithm in alg_list:
            if algorithm not in targets:
                continue
            mean, std = targets[algorithm]
            rows.append(
                {
                    "env": env,
                    "suite": "ogbench",
                    "algorithm": algorithm,
                    "public_source": PUBLIC_SOURCE,
                    "public_source_url": PUBLIC_SOURCE_URL,
                    "public_metric": "normalized_return_pp",
                    "public_mean_pp": mean,
                    "public_std_pp": std,
                    "lower_bound_pp": lower_bound_pp(mean, std),
                    "exact_public_target_available": True,
                    "public_eval_protocol": public_eval_protocol(env),
                    "required_train_steps": gas_required_train_steps(env) if algorithm == "GAS" else None,
                    "required_batch_size": 1024 if algorithm == "GAS" else None,
                    "required_hyperparameters": gas_required_hyperparameters(env) if algorithm == "GAS" else {},
                    "official_checkpoint_available": env_to_slug(env) in OFFICIAL_GAS_PRETRAINED_SLUGS if algorithm == "GAS" else None,
                    "official_source_url": OFFICIAL_ARTIFACT_URL if algorithm == "GAS" else "",
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = row.copy()
            for key, value in list(flat.items()):
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, sort_keys=True)
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", default=",".join(TARGETS_PP))
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--out-jsonl", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    envs = [x.strip() for x in args.envs.split(",") if x.strip()]
    algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()]
    rows = target_rows(envs, algorithms)
    if args.out_jsonl:
        write_jsonl(Path(args.out_jsonl), rows)
    if args.out_csv:
        write_csv(Path(args.out_csv), rows)
    print(json.dumps({"rows": len(rows), "source": PUBLIC_SOURCE_URL}, sort_keys=True))


if __name__ == "__main__":
    main()
