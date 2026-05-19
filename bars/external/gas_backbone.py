from __future__ import annotations

import contextlib
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from .gas_artifacts import (
    GASArtifactSet,
    choose_mujoco_gl,
    download_official_gas_if_available,
    gas_agent_flag_args,
    resolve_gas_artifacts,
    train_gas_backbone_if_missing,
)


@contextlib.contextmanager
def gas_python_path(gas_repo_path: str | os.PathLike[str]):
    path = str(Path(gas_repo_path).resolve())
    inserted = False
    if path not in sys.path:
        sys.path.insert(0, path)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


@dataclass
class GASBackbone:
    env_name: Optional[str] = None
    seed: int = 0
    artifact_root: Path = Path("artifacts/gas")
    gas_repo_path: Path = Path("external_src/GAS")
    gpu: int | str = 0
    artifacts: Optional[GASArtifactSet] = None
    config: Any = None
    env: Any = None
    train_dataset: Any = None
    val_dataset: Any = None
    agent: Any = None
    key_graph: Any = None
    actor_fn: Optional[Callable[..., Any]] = None
    rng: Any = None
    episode_logs: list[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load_or_train(
        cls,
        env_name: str,
        seed: int,
        artifact_root: str | os.PathLike[str],
        gas_repo_path: str | os.PathLike[str],
        gpu: int | str,
        prefer_pretrained: bool = True,
        train_if_missing: bool = True,
        quick: bool = False,
    ) -> "GASBackbone":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("MUJOCO_GL", choose_mujoco_gl())
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if str(gpu).lower() in {"", "cpu", "-1"} else str(gpu)
        if prefer_pretrained:
            download_official_gas_if_available(env_name, artifact_root)
            if seed != 0:
                src = resolve_gas_artifacts(env_name, 0, artifact_root)
                dst = resolve_gas_artifacts(env_name, seed, artifact_root)
                if src.complete and not dst.complete:
                    for src_dir, dst_dir in ((src.tdr_dir, dst.tdr_dir), (src.policy_dir, dst.policy_dir), (src.graph_dir, dst.graph_dir)):
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        for p in src_dir.glob("*"):
                            if p.is_file() and not (dst_dir / p.name).exists():
                                shutil.copy2(p, dst_dir / p.name)
        artifacts = resolve_gas_artifacts(env_name, seed, artifact_root)
        if not artifacts.complete and train_if_missing:
            artifacts = train_gas_backbone_if_missing(
                env_name=env_name,
                seed=seed,
                gpu=gpu,
                gas_repo_path=gas_repo_path,
                artifact_root=artifact_root,
                quick=quick,
                full=not quick,
            )
        bb = cls(
            env_name=env_name,
            seed=seed,
            artifact_root=Path(artifact_root),
            gas_repo_path=Path(gas_repo_path),
            gpu=gpu,
            artifacts=artifacts,
        )
        if artifacts.keygraph is not None:
            bb.load_keygraph(artifacts.keygraph)
        if artifacts.policy_checkpoint is not None:
            bb.load_policy(artifacts.policy_checkpoint)
        return bb

    def _get_config(self) -> Any:
        with gas_python_path(self.gas_repo_path):
            from M_utils.agents.gas import get_config

            config = get_config()
        # Keep these aligned with the official command table.
        args = gas_agent_flag_args(self.env_name or "")
        for k, v in zip(args[::2], args[1::2]):
            name = k.replace("--agent_config.", "")
            if name in {"batch_size", "way_steps"}:
                config[name] = int(v)
            elif name in {"discount", "tdr_expectile", "alpha", "p_aug"}:
                config[name] = float(v)
            else:
                config[name] = v
        return config

    def load_env_and_dataset(self) -> tuple[Any, Any, Any]:
        if self.env is not None:
            return self.env, self.train_dataset, self.val_dataset
        if self.env_name is None:
            raise ValueError("env_name is required before loading a GAS environment")
        with gas_python_path(self.gas_repo_path):
            if self.env_name in ["kitchen-partial-v0"]:
                from D_utils.d4rl_env_utils import d4rl_make_env_and_dataset

                env, train_dataset = d4rl_make_env_and_dataset(self.env_name, self.seed)
                val_dataset = None
            else:
                from O_utils.env_utils import make_env_and_datasets

                env, train_dataset, val_dataset = make_env_and_datasets(self.env_name, self.seed)
        self.env, self.train_dataset, self.val_dataset = env, train_dataset, val_dataset
        return env, train_dataset, val_dataset

    def load_keygraph(self, keygraph_path: str | os.PathLike[str]) -> Any:
        keygraph_path = Path(keygraph_path)
        with gas_python_path(self.gas_repo_path):
            try:
                from K_utils.keygraph_utils import KeyGraph

                kg = KeyGraph()
                kg.load_keygraph(str(keygraph_path.parent), keygraph_path.stem)
            except Exception:
                with open(keygraph_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    kg = type("LoadedGASKeyGraph", (), {})()
                    for k, v in data.items():
                        setattr(kg, k, v)
                else:
                    kg = data
        self.key_graph = kg
        return kg

    def load_policy(self, policy_path: str | os.PathLike[str]) -> Any:
        policy_path = Path(policy_path)
        self.config = self._get_config()
        env, train_dataset, _ = self.load_env_and_dataset()
        with gas_python_path(self.gas_repo_path):
            from M_utils.agents import agents_dict
            from M_utils.flax_utils import restore_agent
            from O_utils.datasets import Dataset, GCDataset
            from O_utils.evaluation import supply_rng
            import jax

            gc_dataset = GCDataset(Dataset.create(**train_dataset), self.config)
            example_batch = gc_dataset.sample(1)
            agent_class = agents_dict[self.config["agent_name"]]
            agent = agent_class.create(
                self.seed,
                example_batch["observations"],
                example_batch["actions"],
                self.config,
            )
            epoch = policy_path.name.split("_")[-1].split(".")[0]
            agent = restore_agent(agent, str(policy_path.parent), epoch)
            self.rng = jax.random.PRNGKey(self.seed)
            self.actor_fn = supply_rng(agent.sample_actions, rng=self.rng)
        self.agent = agent
        return agent

    def get_phi(self, observation_batch: np.ndarray) -> np.ndarray:
        if self.agent is None:
            raise RuntimeError("GAS policy/TDR is not loaded")
        x = np.asarray(observation_batch)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        phi = np.asarray(self.agent.get_phi(x))
        return phi[0] if squeeze else phi

    def sample_action(self, observation: np.ndarray, phi_subgoal_or_skill: np.ndarray, final_goal: bool = False) -> np.ndarray:
        if self.actor_fn is None:
            raise RuntimeError("GAS actor is not loaded")
        obs = np.asarray(observation)
        target = np.asarray(phi_subgoal_or_skill, dtype=np.float32)
        phi_obs = self.get_phi(obs)
        delta = target - phi_obs
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-10:
            skill = np.zeros_like(delta, dtype=np.float32)
        else:
            skill = (delta / norm).astype(np.float32)
        action = self.actor_fn(observations=obs, goals=skill, temperature=0.0)
        return np.clip(np.asarray(action), -1.0, 1.0)

    def sample_action_from_skill(self, observation: np.ndarray, skill: np.ndarray) -> np.ndarray:
        if self.actor_fn is None:
            raise RuntimeError("GAS actor is not loaded")
        action = self.actor_fn(observations=np.asarray(observation), goals=np.asarray(skill), temperature=0.0)
        return np.clip(np.asarray(action), -1.0, 1.0)

    def setup_task_env(self, env: Any, env_name: str, task_id: int, seed: int, render_goal: bool = False):
        if env_name in ["kitchen-partial-v0"]:
            with gas_python_path(self.gas_repo_path):
                from O_utils.evaluation import setup_task_env

                return setup_task_env(env, env_name, task_id, render_goal, seed)
        observation, info = env.reset(seed=seed, options=dict(task_id=task_id, render_goal=render_goal))
        goal = info.get("goal")
        if goal is None:
            raise RuntimeError(f"{env_name} reset did not provide info['goal']; refusing random goal fallback")
        goal_rendered = info.get("goal_rendered") if render_goal else None
        return env, observation, goal, 0.0, False, goal_rendered

    def step_env(self, env: Any, env_name: str, action: np.ndarray):
        if env_name in ["kitchen-partial-v0"]:
            next_observation, reward, done, info = env.step(action)
            return next_observation[:30], reward, bool(done), info
        out = env.step(action)
        if len(out) == 5:
            next_observation, reward, terminated, truncated, info = out
            return next_observation, reward, bool(terminated or truncated), info
        next_observation, reward, done, info = out
        return next_observation, reward, bool(done), info

    def get_task_ids(self, env: Any) -> list[int]:
        if self.env_name in ["kitchen-partial-v0"]:
            return [1]
        infos = getattr(getattr(env, "unwrapped", env), "task_infos", None)
        if infos is None:
            infos = getattr(env, "task_infos", None)
        if infos is None:
            return [1]
        return list(range(1, len(infos) + 1))

    def export_dataset_embeddings(self, out_path: str | os.PathLike[str], batch_size: int = 1024) -> Path:
        if self.agent is None:
            raise RuntimeError("Load a GAS policy/TDR before exporting embeddings")
        _, train_dataset, _ = self.load_env_and_dataset()
        observations = np.asarray(train_dataset["observations"])
        phis = []
        for st in range(0, len(observations), batch_size):
            phis.append(self.get_phi(observations[st : st + batch_size]))
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.concatenate(phis, axis=0).astype(np.float32))
        meta = {"env_name": self.env_name, "seed": self.seed, "num_embeddings": int(len(observations))}
        with open(out.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return out

    def evaluate_episode(
        self,
        env: Any,
        env_name: str,
        task_id: int,
        episode_id: int,
        seed: int,
        planner_fn: Callable[[np.ndarray, np.ndarray], Any],
        max_steps: int = 1000,
        final_goal_threshold: float = 2.0,
    ) -> Dict[str, Any]:
        start_time = time.time()
        env, observation, goal, _, done, _ = self.setup_task_env(env, env_name, task_id, seed, render_goal=False)
        actual_goal_phi = self.get_phi(goal)
        initial_phi = self.get_phi(observation)
        plan = planner_fn(initial_phi, actual_goal_phi)
        total_reward = 0.0
        steps = 0
        success = False
        path = list(getattr(plan, "subgoal_phis", []) or [])
        subgoal_idx = 0
        while not done and steps < max_steps:
            phi_obs = self.get_phi(observation)
            if subgoal_idx >= len(path) or np.linalg.norm(actual_goal_phi - phi_obs) <= final_goal_threshold:
                target_phi = actual_goal_phi
            else:
                target_phi = np.asarray(path[subgoal_idx])
                if np.linalg.norm(target_phi - phi_obs) <= final_goal_threshold:
                    subgoal_idx += 1
                    continue
            action = self.sample_action(observation, target_phi)
            observation, reward, done, info = self.step_env(env, env_name, action)
            total_reward += float(reward)
            steps += 1
            ep_info = info.get("episode", {}) if isinstance(info, dict) else {}
            success = bool(
                success
                or ep_info.get("success", False)
                or info.get("success", False)
                or info.get("goal_achieved", False)
                or info.get("is_success", False)
            )
        final_phi = self.get_phi(observation)
        row = {
            "env_name": env_name,
            "seed": seed,
            "task_id": task_id,
            "episode_id": episode_id,
            "success": int(success),
            "return": total_reward,
            "steps": steps,
            "duration_sec": time.time() - start_time,
            "actual_goal_raw": np.asarray(goal).reshape(-1).tolist(),
            "actual_goal_phi": np.asarray(actual_goal_phi).reshape(-1).tolist(),
            "initial_goal_dist_phi": float(np.linalg.norm(actual_goal_phi - initial_phi)),
            "final_goal_dist_phi": float(np.linalg.norm(actual_goal_phi - final_phi)),
        }
        self.episode_logs.append(row)
        return row
