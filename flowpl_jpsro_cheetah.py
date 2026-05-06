#!/usr/bin/env python3
from __future__ import annotations

import sys

if sys.version_info < (3, 9):
    sys.exit(
        f"This script requires Python 3.9+ (got {sys.version_info.major}.{sys.version_info.minor})."
    )

import argparse
import os
import pickle
import time
from dataclasses import dataclass, asdict
from typing import NamedTuple

import jax
jax.tree_map = jax.tree.map

import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import serialization
from functools import partial

JAXMARL_PATH = os.path.join(os.path.dirname(__file__), "JaxMARL")
sys.path.insert(0, JAXMARL_PATH)
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper


@dataclass
class Config:
    jpsro_iters: int = 8
    embed_dim: int = 8
    hidden_dim: int = 128

    br_steps: int = 400
    num_envs: int = 64
    num_steps: int = 128
    update_epochs: int = 6
    num_minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0

    flow_steps: int = 8
    t_embed_dim: int = 16
    policy_output_scale: float = 0.5
    flow_explore_noise: float = 0.1
    flow_init_noise: float = 1.0
    discretize_t: bool = True
    nft_beta: float = 1.0
    nft_adv_clip_max: float = 5.0
    nft_adaptive_weighting: bool = True
    nft_old_policy_ema: float = 0.0
    kl_reg_beta: float = 0.0

    ep_eval_episodes: int = 32
    ep_eval_steps: int = 200

    viz_rollout_steps: int = 300

    seed: int = 42
    save_dir: str = "results_flowpl_cheetah"

    wandb: bool = False
    wandb_project: str = "flowpl-jpsro-cheetah"
    wandb_entity: str = ""
    wandb_mode: str = "online"


def sinusoidal_t_embed(t: jnp.ndarray, embed_dim: int) -> jnp.ndarray:
    half = embed_dim // 2
    freqs = jnp.power(2.0, jnp.arange(half, dtype=jnp.float32))
    scaled = t * freqs[None, :]
    return jnp.concatenate([jnp.cos(scaled), jnp.sin(scaled)], axis=-1)


class VelocityNet(nn.Module):
    action_dim: int
    hidden_dim: int = 128
    t_embed_dim: int = 16

    @nn.compact
    def __call__(self, obs, strategy_embed, x_t, t):
        t_emb = sinusoidal_t_embed(t, self.t_embed_dim)
        x = jnp.concatenate([obs, strategy_embed, x_t, t_emb], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.silu(x)
        v = nn.Dense(self.action_dim, kernel_init=orthogonal(0.1),
                     bias_init=constant(0.0))(x)
        return v


class ValueNet(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs, strategy_embed):
        x = jnp.concatenate([obs, strategy_embed], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(v, axis=-1)


class FlowActorCritic(nn.Module):
    action_dim: int
    hidden_dim: int = 128
    t_embed_dim: int = 16

    def setup(self):
        self.velocity_net = VelocityNet(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            t_embed_dim=self.t_embed_dim,
        )
        self.value_net = ValueNet(hidden_dim=self.hidden_dim)

    def __call__(self, obs, strategy_embed, x_t, t):
        return self.velocity_net(obs, strategy_embed, x_t, t), \
               self.value_net(obs, strategy_embed)

    def velocity(self, obs, strategy_embed, x_t, t):
        return self.velocity_net(obs, strategy_embed, x_t, t)

    def value(self, obs, strategy_embed):
        return self.value_net(obs, strategy_embed)


def flow_sample(
    apply_fn,
    params,
    obs: jnp.ndarray,
    strategy_embed: jnp.ndarray,
    rng: jax.Array,
    flow_steps: int,
    action_dim: int,
    init_noise: float,
    explore_noise: float,
    output_scale: float,
):
    batch_size = obs.shape[0]
    rng_init, rng_kick = jax.random.split(rng)
    x_init = jax.random.normal(rng_init, (batch_size, action_dim)) * init_noise

    t_grid = jnp.linspace(1.0, 0.0, flow_steps + 1)

    def _step(carry, step_i):
        x, rng_k = carry
        rng_k, rng_ki = jax.random.split(rng_k)
        t_cur = jnp.full((batch_size, 1), t_grid[step_i])
        dt = t_grid[step_i + 1] - t_grid[step_i]
        v = apply_fn(
            params, obs, strategy_embed, x, t_cur, method=FlowActorCritic.velocity
        )
        v = v * output_scale
        x_new = x + dt * v
        if explore_noise > 0.0:
            kick = jax.random.normal(rng_ki, x.shape) * explore_noise * jnp.sqrt(jnp.abs(dt))
            x_new = x_new + kick
        return (x_new, rng_k), None

    (x_final, _), _ = jax.lax.scan(
        _step, (x_init, rng_kick), jnp.arange(flow_steps))
    return x_final


def flow_velocity(apply_fn, params, obs, strategy_embed, x_t, t):
    return apply_fn(params, obs, strategy_embed, x_t, t,
                    method=FlowActorCritic.velocity)


def flow_value(apply_fn, params, obs, strategy_embed):
    return apply_fn(params, obs, strategy_embed, method=FlowActorCritic.value)


class Transition(NamedTuple):
    done: jnp.ndarray
    raw_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray


def cce_solver_cooperative(payoff_tensor: np.ndarray) -> np.ndarray:
    sigma = np.zeros_like(payoff_tensor)
    best = np.unravel_index(np.argmax(payoff_tensor), payoff_tensor.shape)
    sigma[best] = 1.0
    return sigma


def cce_marginal(sigma_joint: np.ndarray, player: int) -> np.ndarray:
    if player == 0:
        return sigma_joint.sum(axis=0)
    else:
        return sigma_joint.sum(axis=1)


def weighted_coplayer_embed(
    coplayer_embeds: jnp.ndarray,
    sigma_coplayer: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.einsum('j,jd->d', sigma_coplayer, coplayer_embeds)


def train_br(
    rng: jax.Array,
    env,
    network: FlowActorCritic,
    init_params,
    cop_params,
    cop_network: FlowActorCritic,
    cop_embeds: jnp.ndarray,
    sigma_cop: jnp.ndarray,
    br_embed: jnp.ndarray,
    player_idx: int,
    cfg: Config,
    action_dim_p: int,
    action_dim_c: int,
):
    agents = env.agents
    player_agent = agents[player_idx]
    cop_agent = agents[1 - player_idx]

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=init_params, tx=tx)
    old_params = jax.tree.map(jnp.copy, init_params)

    t_schedule = jnp.linspace(1.0, 0.0, cfg.flow_steps + 1)[:-1]

    def _env_step(runner_state, unused):
        train_state, old_params, env_state, last_obs, rng = runner_state
        rng, rng_p, rng_c, rng_step, rng_idx = jax.random.split(rng, 5)

        p_obs = last_obs[player_agent]
        c_obs = last_obs[cop_agent]

        embed_batch = jnp.broadcast_to(br_embed, (cfg.num_envs, br_embed.shape[-1]))

        x_p_raw = flow_sample(
            network.apply, train_state.params, p_obs, embed_batch, rng_p,
            cfg.flow_steps, action_dim_p,
            cfg.flow_init_noise, cfg.flow_explore_noise, cfg.policy_output_scale,
        )
        p_action = jnp.clip(x_p_raw, -1.0, 1.0)

        val_p = flow_value(network.apply, train_state.params, p_obs, embed_batch)

        num_strats = sigma_cop.shape[0]
        strat_idx = jax.random.choice(
            rng_idx, num_strats, shape=(cfg.num_envs,), p=sigma_cop)
        c_embed = cop_embeds[strat_idx]
        x_c_raw = flow_sample(
            cop_network.apply, cop_params, c_obs, c_embed, rng_c,
            cfg.flow_steps, action_dim_c,
            cfg.flow_init_noise, 0.0, cfg.policy_output_scale,
        )
        c_action = jnp.clip(x_c_raw, -1.0, 1.0)

        if player_idx == 0:
            actions = {agents[0]: p_action, agents[1]: c_action}
        else:
            actions = {agents[0]: c_action, agents[1]: p_action}

        rng_steps = jax.random.split(rng_step, cfg.num_envs)
        next_obs, next_state, rewards, dones, info = jax.vmap(env.step)(
            rng_steps, env_state, actions)

        transition = Transition(
            done=dones[player_agent],
            raw_action=x_p_raw,
            value=val_p,
            reward=rewards[player_agent],
            obs=p_obs,
        )
        return (train_state, old_params, next_state, next_obs, rng), transition

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(carry, transition):
            gae, next_value = carry
            delta = (transition.reward
                     + cfg.gamma * next_value * (1 - transition.done)
                     - transition.value)
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - transition.done) * gae
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True, unroll=8)
        return advantages, advantages + traj_batch.value

    def _update_step(runner_state, unused):
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, cfg.num_steps)
        train_state, old_params, env_state, last_obs, rng = runner_state

        p_obs_last = last_obs[player_agent]
        embed_batch = jnp.broadcast_to(br_embed, (cfg.num_envs, br_embed.shape[-1]))
        last_val = flow_value(network.apply, train_state.params,
                              p_obs_last, embed_batch)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        ema = cfg.nft_old_policy_ema
        if ema > 0.0:
            old_params = jax.tree.map(
                lambda o, n: ema * o + (1.0 - ema) * n,
                old_params, train_state.params)
        else:
            old_params = jax.tree.map(jnp.copy, train_state.params)

        def _update_epoch(update_state, unused):
            def _update_minibatch(train_state, batch_info):
                traj_batch, advantages, targets, rng_mb = batch_info

                def _loss_fn(params, traj_batch, gae, targets, rng_mb):
                    rng_noise, rng_t = jax.random.split(rng_mb)
                    B = traj_batch.obs.shape[0]

                    embed_mb = jnp.broadcast_to(
                        br_embed, (B, br_embed.shape[-1]))

                    value_pred = flow_value(
                        network.apply, params, traj_batch.obs, embed_mb)
                    value_loss = 0.5 * jnp.mean((value_pred - targets) ** 2)

                    x0 = traj_batch.raw_action

                    noise = jax.random.normal(rng_noise, x0.shape) * cfg.flow_init_noise

                    if cfg.discretize_t:
                        idx = jax.random.randint(
                            rng_t, (B,), 0, cfg.flow_steps)
                        t = t_schedule[idx].reshape(B, 1)
                    else:
                        t = jax.random.uniform(rng_t, (B, 1), minval=1e-3, maxval=1.0)

                    x_t = (1.0 - t) * x0 + t * noise

                    v_old = flow_velocity(
                        network.apply, old_params, traj_batch.obs, embed_mb, x_t, t)
                    v_old = jax.lax.stop_gradient(v_old * cfg.policy_output_scale)

                    v_cur_raw = flow_velocity(
                        network.apply, params, traj_batch.obs, embed_mb, x_t, t)
                    v_cur = v_cur_raw * cfg.policy_output_scale

                    adv_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    adv_clip = jnp.clip(
                        adv_norm, -cfg.nft_adv_clip_max, cfg.nft_adv_clip_max)
                    r = jnp.clip(
                        (adv_clip / cfg.nft_adv_clip_max) / 2.0 + 0.5, 0.0, 1.0)
                    r_col = r[:, None]

                    beta = cfg.nft_beta
                    v_pos = beta * v_cur + (1.0 - beta) * v_old
                    v_neg = (1.0 + beta) * v_old - beta * v_cur

                    x0_pos = x_t - t * v_pos
                    x0_neg = x_t - t * v_neg

                    if cfg.nft_adaptive_weighting:
                        w_pos = jax.lax.stop_gradient(
                            jnp.clip(jnp.abs(x0_pos - x0).mean(
                                axis=-1, keepdims=True), a_min=1e-5))
                        w_neg = jax.lax.stop_gradient(
                            jnp.clip(jnp.abs(x0_neg - x0).mean(
                                axis=-1, keepdims=True), a_min=1e-5))
                        pos_loss = ((x0_pos - x0) ** 2 / w_pos).mean(axis=-1)
                        neg_loss = ((x0_neg - x0) ** 2 / w_neg).mean(axis=-1)
                    else:
                        pos_loss = ((x0_pos - x0) ** 2).mean(axis=-1)
                        neg_loss = ((x0_neg - x0) ** 2).mean(axis=-1)

                    r_flat = r_col.squeeze(-1)
                    nft_per_sample = (r_flat * pos_loss
                                       + (1.0 - r_flat) * neg_loss) / beta
                    policy_loss = nft_per_sample.mean() * cfg.nft_adv_clip_max

                    kl_loss = jnp.mean((v_cur - v_old) ** 2)
                    total = (policy_loss
                             + cfg.vf_coef * value_loss
                             + cfg.kl_reg_beta * kl_loss)

                    return total, (policy_loss, value_loss,
                                   pos_loss.mean(), neg_loss.mean(), kl_loss)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss_val, aux), grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets, rng_mb)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (loss_val, *aux)

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, rng_perm, rng_mb = jax.random.split(rng, 3)
            batch_size = cfg.num_steps * cfg.num_envs
            perm = jax.random.permutation(rng_perm, batch_size)
            batch = jax.tree.map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]),
                (traj_batch, advantages, targets))
            shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), batch)
            rng_mbs = jax.random.split(rng_mb, cfg.num_minibatches)
            minibatches = jax.tree.map(
                lambda x: x.reshape([cfg.num_minibatches, -1] + list(x.shape[1:])),
                shuffled) + (rng_mbs,)
            train_state, losses = jax.lax.scan(
                _update_minibatch, train_state, minibatches)
            return (train_state, traj_batch, advantages, targets, rng), losses

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, _ = jax.lax.scan(
            _update_epoch, update_state, None, cfg.update_epochs)
        train_state = update_state[0]
        rng = update_state[-1]

        mean_return = traj_batch.reward.sum(axis=0).mean()
        return (train_state, old_params, env_state, last_obs, rng), mean_return

    rng, rng_reset = jax.random.split(rng)
    reset_rngs = jax.random.split(rng_reset, cfg.num_envs)
    obs, env_state = jax.vmap(env.reset)(reset_rngs)

    rng, rng_train = jax.random.split(rng)
    runner_state = (train_state, old_params, env_state, obs, rng_train)
    runner_state, returns = jax.lax.scan(
        _update_step, runner_state, None, cfg.br_steps)

    return runner_state[0], {"returns": returns}


def eval_joint_strategy(
    rng, env, network_0, params_0, network_1, params_1,
    embed_0, embed_1, num_envs, num_steps, cfg: Config,
    action_dim_0: int, action_dim_1: int,
):
    agents = env.agents

    def _step(carry, unused):
        env_state, obs, rng = carry
        rng, rng_0, rng_1, rng_step = jax.random.split(rng, 4)

        e0 = jnp.broadcast_to(embed_0, (num_envs, embed_0.shape[-1]))
        e1 = jnp.broadcast_to(embed_1, (num_envs, embed_1.shape[-1]))

        x0_raw = flow_sample(
            network_0.apply, params_0, obs[agents[0]], e0, rng_0,
            cfg.flow_steps, action_dim_0, cfg.flow_init_noise, 0.0,
            cfg.policy_output_scale,
        )
        x1_raw = flow_sample(
            network_1.apply, params_1, obs[agents[1]], e1, rng_1,
            cfg.flow_steps, action_dim_1, cfg.flow_init_noise, 0.0,
            cfg.policy_output_scale,
        )

        a0 = jnp.clip(x0_raw, -1.0, 1.0)
        a1 = jnp.clip(x1_raw, -1.0, 1.0)

        actions = {agents[0]: a0, agents[1]: a1}
        rng_steps = jax.random.split(rng_step, num_envs)
        next_obs, next_state, rewards, dones, info = jax.vmap(env.step)(
            rng_steps, env_state, actions)
        return (next_state, next_obs, rng), rewards[agents[0]]

    rng, rng_reset = jax.random.split(rng)
    reset_rngs = jax.random.split(rng_reset, num_envs)
    obs, env_state = jax.vmap(env.reset)(reset_rngs)

    _, rewards = jax.lax.scan(_step, (env_state, obs, rng), None, num_steps)
    return rewards.sum(axis=0).mean()


def rollout_trajectory(
    rng, env_raw, network_0, params_0, network_1, params_1,
    embed_0, embed_1, num_steps, cfg: Config,
    action_dim_0: int, action_dim_1: int,
):
    agents = env_raw.agents

    def _step(carry, unused):
        env_state, obs, rng = carry
        rng, rng_0, rng_1, rng_step = jax.random.split(rng, 4)

        e0 = embed_0[None]
        e1 = embed_1[None]
        obs0 = obs[agents[0]][None]
        obs1 = obs[agents[1]][None]

        x0_raw = flow_sample(
            network_0.apply, params_0, obs0, e0, rng_0,
            cfg.flow_steps, action_dim_0, cfg.flow_init_noise, 0.0,
            cfg.policy_output_scale,
        )
        x1_raw = flow_sample(
            network_1.apply, params_1, obs1, e1, rng_1,
            cfg.flow_steps, action_dim_1, cfg.flow_init_noise, 0.0,
            cfg.policy_output_scale,
        )
        a0 = jnp.clip(x0_raw[0], -1.0, 1.0)
        a1 = jnp.clip(x1_raw[0], -1.0, 1.0)

        actions = {agents[0]: a0, agents[1]: a1}
        next_obs, next_state, rewards, dones, info = env_raw.step(
            rng_step, env_state, actions)
        return (next_state, next_obs, rng), next_state.pipeline_state

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env_raw.reset(rng_reset)
    init_pipeline = env_state.pipeline_state

    _, traj_pipeline_states = jax.lax.scan(
        _step, (env_state, obs, rng), None, num_steps)

    return init_pipeline, traj_pipeline_states


def save_checkpoint(save_dir, iteration, params, embeds, payoff_matrix, sigma):
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"iter_{iteration:03d}.pkl")
    data = {
        "iteration": iteration,
        "params_p0": serialization.to_bytes(params[0]),
        "params_p1": serialization.to_bytes(params[1]),
        "embeds_p0": [np.array(e) for e in embeds[0]],
        "embeds_p1": [np.array(e) for e in embeds[1]],
        "payoff_matrix": payoff_matrix,
        "sigma": sigma,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def save_trajectory(save_dir, label, init_ps, traj_ps):
    traj_dir = os.path.join(save_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    path = os.path.join(traj_dir, f"{label}.pkl")

    q_init = np.array(init_ps.q)
    qd_init = np.array(init_ps.qd)
    q_traj = np.array(traj_ps.q)
    qd_traj = np.array(traj_ps.qd)
    x_pos_init = np.array(init_ps.x.pos)
    x_pos_traj = np.array(traj_ps.x.pos)
    x_rot_init = np.array(init_ps.x.rot)
    x_rot_traj = np.array(traj_ps.x.rot)

    data = {
        "q": np.concatenate([q_init[None], q_traj], axis=0),
        "qd": np.concatenate([qd_init[None], qd_traj], axis=0),
        "x_pos": np.concatenate([x_pos_init[None], x_pos_traj], axis=0),
        "x_rot": np.concatenate([x_rot_init[None], x_rot_traj], axis=0),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def run_flowpl_jpsro(cfg: Config):
    print("=" * 70)
    print("  FlowPL-JPSRO (MANFT flow policy) for Multi-Agent HalfCheetah")
    print("=" * 70)

    rng = jax.random.PRNGKey(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            config=asdict(cfg),
            mode=cfg.wandb_mode,
            name=f"flowpl-jpsro-seed{cfg.seed}",
        )

    env_raw = jaxmarl.make("halfcheetah_2x3")
    env = LogWrapper(env_raw, replace_info=True)
    agents = env.agents

    obs_dims = {a: env.observation_space(a).shape[0] for a in agents}
    act_dims = {a: env.action_space(a).shape[0] for a in agents}
    print(f"  Agents: {agents}")
    print(f"  Obs dims: {obs_dims}, Act dims: {act_dims}")
    print(f"  Flow: steps={cfg.flow_steps}, t_embed_dim={cfg.t_embed_dim}, "
          f"output_scale={cfg.policy_output_scale}")

    networks = [
        FlowActorCritic(
            action_dim=act_dims[agents[p]],
            hidden_dim=cfg.hidden_dim,
            t_embed_dim=cfg.t_embed_dim,
        )
        for p in range(2)
    ]
    action_dims = [act_dims[agents[p]] for p in range(2)]

    rng, *init_rngs = jax.random.split(rng, 3)
    dummy_obs = [jnp.zeros((1, obs_dims[agents[p]])) for p in range(2)]
    dummy_embed = jnp.zeros((1, cfg.embed_dim))
    dummy_xt = [jnp.zeros((1, action_dims[p])) for p in range(2)]
    dummy_t = jnp.zeros((1, 1))
    params = [
        networks[p].init(
            init_rngs[p], dummy_obs[p], dummy_embed, dummy_xt[p], dummy_t)
        for p in range(2)
    ]

    rng, rng_e0, rng_e1 = jax.random.split(rng, 3)
    embeds = [
        [jax.random.normal(rng_e0, (cfg.embed_dim,)) * 0.1],
        [jax.random.normal(rng_e1, (cfg.embed_dim,)) * 0.1],
    ]

    param_snapshots = [
        [jax.tree.map(jnp.copy, params[0])],
        [jax.tree.map(jnp.copy, params[1])],
    ]

    payoff_history = []
    sigma_history = []
    returns_history = []
    global_br_step = 0

    @jax.jit
    def _eval_joint(rng, params_0, params_1, embed_0, embed_1):
        return eval_joint_strategy(
            rng, env, networks[0], params_0, networks[1], params_1,
            embed_0, embed_1, cfg.ep_eval_episodes, cfg.ep_eval_steps, cfg,
            action_dims[0], action_dims[1])

    @jax.jit
    def _rollout(rng, params_0, params_1, embed_0, embed_1):
        return rollout_trajectory(
            rng, env_raw, networks[0], params_0, networks[1], params_1,
            embed_0, embed_1, cfg.viz_rollout_steps, cfg,
            action_dims[0], action_dims[1])

    print("\n  Evaluating initial payoff...")
    rng, rng_ep = jax.random.split(rng)
    initial_payoff = float(_eval_joint(
        rng_ep, params[0], params[1], embeds[0][0], embeds[1][0]))
    print(f"  Initial payoff (strat 0 vs strat 0): {initial_payoff:.2f}")

    if cfg.wandb:
        import wandb
        wandb.log({"jpsro/initial_payoff": initial_payoff}, step=0)

    payoff_matrix = np.array([[initial_payoff]])
    sigma = cce_solver_cooperative(payoff_matrix)

    rng, rng_traj = jax.random.split(rng)
    init_ps, traj_ps = _rollout(
        rng_traj, params[0], params[1], embeds[0][0], embeds[1][0])
    traj_path = save_trajectory(cfg.save_dir, "iter_000_s0_vs_s0", init_ps, traj_ps)
    print(f"  Trajectory saved: {traj_path}")

    for t in range(1, cfg.jpsro_iters + 1):
        iter_t0 = time.time()
        print(f"\n{'-' * 60}")
        print(f"  JPSRO Iteration {t}/{cfg.jpsro_iters}")
        print(f"{'-' * 60}")

        ref_params = [jax.tree.map(jnp.copy, params[p]) for p in range(2)]

        for p in range(2):
            cop = 1 - p
            sigma_cop = cce_marginal(sigma, p)
            sigma_cop_jnp = jnp.array(sigma_cop, dtype=jnp.float32)

            num_cop_strats = len(embeds[cop])
            if sigma_cop_jnp.shape[0] < num_cop_strats:
                sigma_cop_jnp = jnp.concatenate([
                    sigma_cop_jnp,
                    jnp.zeros(num_cop_strats - sigma_cop_jnp.shape[0])])
            elif sigma_cop_jnp.shape[0] > num_cop_strats:
                sigma_cop_jnp = sigma_cop_jnp[:num_cop_strats]
            sigma_cop_jnp = sigma_cop_jnp / (sigma_cop_jnp.sum() + 1e-8)

            cop_embeds_arr = jnp.stack(embeds[cop])
            br_embed = weighted_coplayer_embed(cop_embeds_arr, sigma_cop_jnp)

            print(f"\n  Player {p}: flow-BR vs co-player CCE marginal "
                  f"(support: {int((sigma_cop > 1e-6).sum())} strategies)")

            br_train = jax.jit(partial(
                train_br,
                env=env,
                network=networks[p],
                cop_network=networks[cop],
                player_idx=p,
                cfg=cfg,
                action_dim_p=action_dims[p],
                action_dim_c=action_dims[cop],
            ))

            t0 = time.time()
            br_state, br_info = br_train(
                rng=rng,
                init_params=params[p],
                cop_params=ref_params[cop],
                cop_embeds=cop_embeds_arr,
                sigma_cop=sigma_cop_jnp,
                br_embed=br_embed,
            )
            jax.block_until_ready(br_state.params)
            elapsed = time.time() - t0

            rng, _ = jax.random.split(rng)

            br_returns = np.array(br_info["returns"])
            print(f"    BR training: {elapsed:.1f}s, "
                  f"final return: {br_returns[-1]:.2f}, "
                  f"max return: {br_returns.max():.2f}")
            returns_history.append(br_returns)

            if cfg.wandb:
                import wandb
                for step_i, ret_val in enumerate(br_returns):
                    wandb.log({
                        f"br/p{p}_return": float(ret_val),
                        f"br/p{p}_iter": t,
                        "br/global_step": global_br_step,
                    }, step=global_br_step)
                    global_br_step += 1

            params[p] = br_state.params

            rng, rng_ne = jax.random.split(rng)
            new_embed = jax.random.normal(rng_ne, (cfg.embed_dim,)) * 0.1
            embeds[p].append(new_embed)
            param_snapshots[p].append(jax.tree.map(jnp.copy, params[p]))

        n0, n1 = len(embeds[0]), len(embeds[1])
        print(f"\n  Evaluating payoff tensor ({n0} x {n1})...")

        payoff_matrix = np.zeros((n0, n1))
        for i in range(n0):
            for j in range(n1):
                rng, rng_eval = jax.random.split(rng)
                p0_params = param_snapshots[0][min(i, len(param_snapshots[0]) - 1)]
                p1_params = param_snapshots[1][min(j, len(param_snapshots[1]) - 1)]
                payoff_matrix[i, j] = float(_eval_joint(
                    rng_eval, p0_params, p1_params, embeds[0][i], embeds[1][j]))

        print(f"  Payoff matrix G[i,j]:\n{np.round(payoff_matrix, 1)}")
        payoff_history.append(payoff_matrix.copy())

        sigma = cce_solver_cooperative(payoff_matrix)
        best = np.unravel_index(np.argmax(payoff_matrix), payoff_matrix.shape)
        print(f"  CCE: best joint = {best}, payoff = {payoff_matrix[best]:.2f}")
        sigma_history.append(sigma.copy())

        cce_val = payoff_matrix[best]
        gap_0 = max(0, payoff_matrix[:, best[1]].max() - cce_val)
        gap_1 = max(0, payoff_matrix[best[0], :].max() - cce_val)
        cce_gap = max(gap_0, gap_1)
        print(f"  CCE gap: {cce_gap:.3f}")

        iter_elapsed = time.time() - iter_t0

        if cfg.wandb:
            import wandb
            wandb.log({
                "jpsro/iteration": t,
                "jpsro/best_payoff": float(cce_val),
                "jpsro/cce_gap": float(cce_gap),
                "jpsro/num_strategies_p0": n0,
                "jpsro/num_strategies_p1": n1,
                "jpsro/best_joint_p0": int(best[0]),
                "jpsro/best_joint_p1": int(best[1]),
                "jpsro/iter_time_s": iter_elapsed,
            }, step=global_br_step)

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(payoff_matrix, cmap='viridis', aspect='auto')
                ax.set_xlabel("P1 strategy")
                ax.set_ylabel("P0 strategy")
                ax.set_title(f"Payoff (iter {t})")
                plt.colorbar(im, ax=ax)
                for ii in range(n0):
                    for jj in range(n1):
                        c = 'white' if payoff_matrix[ii, jj] < payoff_matrix.mean() else 'black'
                        ax.text(jj, ii, f"{payoff_matrix[ii, jj]:.0f}",
                                ha='center', va='center', fontsize=7, color=c)
                plt.tight_layout()
                wandb.log({f"jpsro/payoff_matrix": wandb.Image(fig)},
                          step=global_br_step)
                plt.close(fig)
            except Exception:
                pass

        ckpt_path = save_checkpoint(
            cfg.save_dir, t, params, embeds, payoff_matrix, sigma)
        print(f"  Checkpoint saved: {ckpt_path}")

        rng, rng_traj = jax.random.split(rng)
        best_p0_params = param_snapshots[0][min(best[0], len(param_snapshots[0]) - 1)]
        best_p1_params = param_snapshots[1][min(best[1], len(param_snapshots[1]) - 1)]
        init_ps, traj_ps = _rollout(
            rng_traj, best_p0_params, best_p1_params,
            embeds[0][best[0]], embeds[1][best[1]])
        label = f"iter_{t:03d}_s{best[0]}_vs_s{best[1]}"
        traj_path = save_trajectory(cfg.save_dir, label, init_ps, traj_ps)
        print(f"  Trajectory saved: {traj_path}")

    print(f"\n{'=' * 70}")
    print(f"  FlowPL-JPSRO Complete")
    print(f"{'=' * 70}")
    print(f"  Strategies: P0={len(embeds[0])}, P1={len(embeds[1])}")
    if payoff_history:
        final_pm = payoff_history[-1]
        print(f"  Final payoff matrix:\n{np.round(final_pm, 1)}")
        best = np.unravel_index(np.argmax(final_pm), final_pm.shape)
        print(f"  Best joint: {best}, payoff={final_pm[best]:.2f}")

    run_data_path = os.path.join(cfg.save_dir, "run_data.pkl")
    with open(run_data_path, "wb") as f:
        pickle.dump({
            "config": asdict(cfg),
            "payoff_history": payoff_history,
            "sigma_history": sigma_history,
            "returns_history": returns_history,
            "embeds_p0": [np.array(e) for e in embeds[0]],
            "embeds_p1": [np.array(e) for e in embeds[1]],
        }, f)
    print(f"  Run data saved: {run_data_path}")

    _plot_results(cfg, payoff_history, returns_history)

    if cfg.wandb:
        import wandb
        wandb.finish()

    return {
        "payoff_history": payoff_history,
        "sigma_history": sigma_history,
        "returns_history": returns_history,
        "embeds": embeds,
        "params": params,
        "param_snapshots": param_snapshots,
    }


def _plot_results(cfg, payoff_history, returns_history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        for idx, ret in enumerate(returns_history):
            p = idx % 2
            it = idx // 2 + 1
            ax.plot(ret, label=f"P{p} iter {it}", alpha=0.7)
        ax.set_xlabel("BR training step")
        ax.set_ylabel("Episode return (per update)")
        ax.set_title("Flow-BR Training Returns (MANFT)")
        ax.legend(fontsize=6, ncol=2)

        ax = axes[1]
        best_payoffs = [np.max(pm) for pm in payoff_history]
        ax.plot(range(1, len(best_payoffs) + 1), best_payoffs, 'bo-')
        ax.set_xlabel("JPSRO iteration")
        ax.set_ylabel("Best joint payoff")
        ax.set_title("Payoff Improvement")

        ax = axes[2]
        if payoff_history:
            final_pm = payoff_history[-1]
            im = ax.imshow(final_pm, cmap='viridis', aspect='auto', origin='upper')
            n0, n1 = final_pm.shape
            ax.set_xticks(np.arange(n1))
            ax.set_yticks(np.arange(n0))
            ax.set_xticklabels([f"s{j}" for j in range(n1)], fontsize=8)
            ax.set_yticklabels([f"s{i}" for i in range(n0)], fontsize=8)
            ax.set_xlabel("P1 (front leg) strategy index j")
            ax.set_ylabel("P0 (rear leg) strategy index i")
            ax.set_title(
                "G[i,j] = payoff for pure joint (i, j)\n"
                "mixed payoff = sum_{i,j} sigma(i,j) G[i,j]"
            )
            plt.colorbar(im, ax=ax)
            for i in range(final_pm.shape[0]):
                for j in range(final_pm.shape[1]):
                    color = 'white' if final_pm[i, j] < final_pm.mean() else 'black'
                    ax.text(j, i, f"{final_pm[i, j]:.0f}",
                            ha='center', va='center', fontsize=7, color=color)

        plt.tight_layout()
        save_path = os.path.join(cfg.save_dir, "flowpl_jpsro_cheetah.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")
        plt.close(fig)

        if cfg.wandb:
            import wandb
            wandb.log({"summary/training_plot": wandb.Image(save_path)})

    except Exception as e:
        print(f"  Plotting failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpsro-iters", type=int, default=8)
    parser.add_argument("--br-steps", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--t-embed-dim", type=int, default=16)
    parser.add_argument("--policy-output-scale", type=float, default=0.5)
    parser.add_argument("--flow-explore-noise", type=float, default=0.1)
    parser.add_argument("--flow-init-noise", type=float, default=1.0)
    parser.add_argument("--nft-beta", type=float, default=1.0)
    parser.add_argument("--nft-adv-clip-max", type=float, default=5.0)
    parser.add_argument("--nft-old-policy-ema", type=float, default=0.0)
    parser.add_argument("--nft-adaptive-weighting", action="store_true", default=True)
    parser.add_argument("--kl-reg-beta", type=float, default=0.0)
    parser.add_argument("--discretize-t", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results_flowpl_cheetah")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="flowpl-jpsro-cheetah")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    cfg = Config(
        jpsro_iters=args.jpsro_iters,
        br_steps=args.br_steps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        flow_steps=args.flow_steps,
        t_embed_dim=args.t_embed_dim,
        policy_output_scale=args.policy_output_scale,
        flow_explore_noise=args.flow_explore_noise,
        flow_init_noise=args.flow_init_noise,
        nft_beta=args.nft_beta,
        nft_adv_clip_max=args.nft_adv_clip_max,
        nft_old_policy_ema=args.nft_old_policy_ema,
        nft_adaptive_weighting=args.nft_adaptive_weighting,
        kl_reg_beta=args.kl_reg_beta,
        discretize_t=args.discretize_t,
        seed=args.seed,
        save_dir=args.save_dir,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )
    run_flowpl_jpsro(cfg)
    print("\nDone!")


if __name__ == "__main__":
    main()
