import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from copy import deepcopy

from M_utils.encoders import GCEncoder, encoder_modules
from M_utils.flax_utils import ModuleDict, TrainState
from M_utils.networks import GCActor, GCValue

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class GASTMDLowAgent(flax.struct.PyTreeNode):
    """GAS-style low-level policy conditioned on TMD psi direction + distance."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        skill_dim = int(config.get("skill_dim", config["tmd_latent_dim"] + 1))
        ex_skill = np.zeros((1, skill_dim), dtype=np.float32)
        action_dim = ex_actions.shape[-1]

        encoders = {}
        if config["encoder"] == "not_used":
            print("Using state-based observations (no encoder).")
        elif config["encoder"] in encoder_modules:
            print(f"Using pixel-based observations with encoder: {config['encoder']}")
            encoder_module = encoder_modules[config["encoder"]]
            encoders["value"] = GCEncoder(state_encoder=encoder_module())
            encoders["critic"] = GCEncoder(state_encoder=encoder_module())
            encoders["actor"] = GCEncoder(state_encoder=encoder_module())
        else:
            raise ValueError(f"Unknown encoder: {config['encoder']}")

        value_def = GCValue(
            ensemble=False,
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            gc_encoder=encoders.get("value"),
        )
        critic_def = GCValue(
            ensemble=True,
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            gc_encoder=encoders.get("critic"),
        )
        actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            final_fc_init_scale=config["final_fc_init_scale"],
            state_dependent_std=config["state_dependent_std"],
            const_std=config["const_std"],
            gc_encoder=encoders.get("actor"),
            log_std_min=config["log_std_min"],
            log_std_max=config["log_std_max"],
            tanh_squash=config["tanh_squash"],
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_skill, None, True)),
            critic=(critic_def, (ex_observations, ex_skill, ex_actions, True)),
            target_critic=(deepcopy(critic_def), (ex_observations, ex_skill, ex_actions, True)),
            actor=(actor_def, (ex_observations, ex_skill, 1.0, True)),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_params = network_def.init(init_rng, **{k: v[1] for k, v in network_info.items()})["params"]
        network_tx = optax.adam(learning_rate=config["lr"])
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params["modules_target_critic"] = network.params["modules_critic"]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        q1, q2 = self.network.select("target_critic")(
            batch["observations"],
            batch["value_skills"],
            batch["actions"],
            goal_encoded=True,
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("value")(
            batch["observations"],
            batch["value_skills"],
            None,
            goal_encoded=True,
            params=grad_params,
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config["expectile"]).mean()
        return value_loss, {"value_loss": value_loss, "v_mean": v.mean(), "v_max": v.max(), "v_min": v.min()}

    def critic_loss(self, batch, grad_params):
        next_v = self.network.select("value")(
            batch["next_observations"],
            batch["value_skills"],
            None,
            goal_encoded=True,
        )
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v
        q1, q2 = self.network.select("critic")(
            batch["observations"],
            batch["value_skills"],
            batch["actions"],
            goal_encoded=True,
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        return critic_loss, {"critic_loss": critic_loss, "q_mean": q.mean(), "q_max": q.max(), "q_min": q.min()}

    def actor_loss(self, batch, grad_params, rng=None):
        dist = self.network.select("actor")(
            batch["observations"],
            batch["actor_skills"],
            temperature=1.0,
            goal_encoded=True,
            params=grad_params,
        )
        if self.config["const_std"]:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
        q1, q2 = self.network.select("critic")(batch["observations"], batch["actor_skills"], q_actions, goal_encoded=True)
        q = jnp.minimum(q1, q2)
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch["actions"])
        bc_loss = -(self.config["alpha"] * log_prob).mean()
        actor_loss = q_loss + bc_loss
        return actor_loss, {
            "actor_loss": actor_loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
            "q_mean": q.mean(),
            "q_abs_mean": jnp.abs(q).mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_critic_actor_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)
        epsilon = 1e-10

        batch_size, rep_dim = batch["psi_obs"].shape
        random_directions = jax.random.normal(rng, (batch_size, rep_dim))
        random_directions = random_directions / (jnp.linalg.norm(random_directions, axis=1, keepdims=True) + epsilon)
        random_distances = jax.random.uniform(rng, (batch_size, 1))
        batch["value_skills"] = jnp.concatenate([random_directions, random_distances], axis=-1)
        batch["rewards"] = ((batch["psi_next_obs"] - batch["psi_obs"]) * random_directions).sum(axis=1)
        batch["masks"] = jnp.ones(batch_size)

        value_loss, value_info = self.value_loss(batch, grad_params)
        critic_loss, critic_info = self.critic_loss(batch, grad_params)

        actor_direction = batch["psi_actor_goals"] - batch["psi_obs"]
        actor_direction = actor_direction / (jnp.linalg.norm(actor_direction, axis=1, keepdims=True) + epsilon)
        distance_scalar = jnp.clip(batch["tmd_actor_dist"][:, None] / self.config["edge_distance_threshold"], 0.0, 1.0)
        batch["actor_skills"] = jnp.concatenate([actor_direction, distance_scalar], axis=-1)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)

        for k, v in value_info.items():
            info[f"value/{k}"] = v
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v
        return value_loss + critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def critic_actor_update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def critic_actor_loss_fn(grad_params):
            return self.total_critic_actor_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=critic_actor_loss_fn)
        self.target_update(new_network, "critic")
        return self.replace(network=new_network, rng=new_rng), info

    def sample_actions(self, observations, goals=None, temperature=1.0, seed=None):
        dist = self.network.select("actor")(observations, goals, temperature=temperature, goal_encoded=True)
        actions = dist.mode() if seed is None or temperature == 0.0 else dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="gas_tmd_low",
            encoder="not_used",
            value_hidden_dims=(512, 512, 512),
            actor_hidden_dims=(512, 512, 512),
            tmd_latent_dim=512,
            skill_dim=513,
            layer_norm=True,
            state_dependent_std=False,
            const_std=True,
            tanh_squash=False,
            log_std_min=-5,
            log_std_max=2,
            final_fc_init_scale=1e-2,
            discount=0.995,
            expectile=0.7,
            alpha=1.0,
            lr=3e-4,
            tau=0.005,
            batch_size=1024,
            p_aug=0.0,
            way_steps=8,
            edge_distance_threshold=1.0,
        )
    )
