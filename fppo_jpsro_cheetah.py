#!/usr/bin/env python3

from __future__ import annotations

import sys
if sys.version_info < (3, 9):
    sys.exit(
        "fppo_jpsro_cheetah.py requires Python 3.9+ (uses dataclasses, PEP-526 "
        "annotations, JAX). Detected Python {0}.{1}. "
        "Activate the venv first:  source JaxMARL/.venv/bin/activate  "
        "or run explicitly:  python3 fppo_jpsro_cheetah.py ...".format(
            sys.version_info[0], sys.version_info[1]))

import argparse
import os
import pickle
import time
from dataclasses import dataclass, asdict, replace
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

def clamp_ste(x, lo=None, hi=None):

    y = x
    if lo is not None:
        y = jnp.maximum(y, lo)
    if hi is not None:
        y = jnp.minimum(y, hi)
    return x + jax.lax.stop_gradient(y - x)

@dataclass
class Config:
    jpsro_iters: int = 8
    embed_dim: int = 8

    hidden_dim: int = 256

    br_steps: int = 25
    num_envs: int = 256
    num_steps: int = 1000
    update_epochs: int = 3
    num_minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 1.0
    max_grad_norm: float = 2.0

    flow_steps: int = 10
    t_embed_dim: int = 16
    policy_output_scale: float = 0.25
    flow_explore_noise: float = 0.0

    feather_std: float = 0.0

    flow_init_noise: float = 1.0
    discretize_t: bool = True

    n_samples_per_action: int = 8

    ratio_clip: float = 0.1
    cfm_loss_clip: float = 3.0
    bc_loss_scale: float = 0.0
    output_mode: str = "u_but_supervise_as_eps"

    cfm_diff_clamp_ste: bool = True
    cfm_loss_clamp_max: float = 0.0
    neg_adv_cur_cfm_clamp_max: float = 0.0

    use_adaptive_lr: bool = True
    desired_kl: float = 0.02
    kl_lr_factor: float = 1.5
    lr_min: float = 1e-5
    lr_max: float = 1e-2

    value_clip: float = 0.0

    global_adv_norm: bool = True

    reg_beta: float = 1.0
    reg_n_slots: int = 4
    distill_beta: float = 1.0

    decouple_encoder: bool = True

    br_head_layers: int = 2
    pop_head_layers: int = 2

    embed_lr: float = 3e-4

    sigma_cce_prob: float = 0.5
    sigma_dirichlet_alpha: float = 1.0

    br_steps_scale: int = 3

    best_ema_alpha: float = 0.9

    memory_dim: int = 256
    use_prev_cop_action: bool = True

    ep_eval_episodes: int = 32
    ep_eval_steps: int = 200

    viz_rollout_steps: int = 300

    seed: int = 42
    save_dir: str = "results_fppo_cheetah"

    wandb: bool = False
    wandb_project: str = "fppo-jpsro-cheetah"
    wandb_entity: str = ""
    wandb_mode: str = "online"

def sinusoidal_t_embed(t: jnp.ndarray, embed_dim: int) -> jnp.ndarray:

    half = embed_dim // 2
    freqs = jnp.power(2.0, jnp.arange(half, dtype=jnp.float32))
    scaled = t * freqs[None, :]
    return jnp.concatenate([jnp.cos(scaled), jnp.sin(scaled)], axis=-1)

class ObsEncoder(nn.Module):

    hidden_dim: int = 128
    memory_dim: int = 128
    use_prev_cop_action: bool = True

    @nn.compact
    def __call__(self, obs, prev_cop_act, h_prev):
        if self.use_prev_cop_action:
            x_in = jnp.concatenate([obs, prev_cop_act], axis=-1)
        else:
            x_in = obs
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x_in)
        x = nn.tanh(x)
        if self.memory_dim > 0:

            new_carry, _ = nn.GRUCell(features=self.memory_dim,
                                      kernel_init=orthogonal(np.sqrt(2)),
                                      bias_init=constant(0.0))(h_prev, x)
            return new_carry
        return x

class VelocityHead(nn.Module):

    action_dim: int
    hidden_dim: int = 128
    t_embed_dim: int = 16
    head_layers: int = 2

    @nn.compact
    def __call__(self, h_obs, cond, x_t, t):
        t_emb = sinusoidal_t_embed(t, self.t_embed_dim)
        x = jnp.concatenate([h_obs, cond, x_t, t_emb], axis=-1)
        for _ in range(self.head_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                         bias_init=constant(0.0))(x)
            x = nn.silu(x)
        v = nn.Dense(self.action_dim, kernel_init=orthogonal(0.1),
                     bias_init=constant(0.0))(x)
        return v

class ValueHead(nn.Module):

    hidden_dim: int = 128

    @nn.compact
    def __call__(self, h_obs, cond):
        x = jnp.concatenate([h_obs, cond], axis=-1)
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
    cond_dim: int
    cop_action_dim: int
    hidden_dim: int = 128
    t_embed_dim: int = 16
    memory_dim: int = 128
    use_prev_cop_action: bool = True
    br_head_layers: int = 2
    pop_head_layers: int = 2

    def setup(self):
        self.encoder = ObsEncoder(
            hidden_dim=self.hidden_dim,
            memory_dim=self.memory_dim,
            use_prev_cop_action=self.use_prev_cop_action,
        )
        self.vel_br = VelocityHead(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            t_embed_dim=self.t_embed_dim,
            head_layers=self.br_head_layers,
        )
        self.vel_pop = VelocityHead(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            t_embed_dim=self.t_embed_dim,
            head_layers=self.pop_head_layers,
        )
        self.val_br = ValueHead(hidden_dim=self.hidden_dim)
        self.val_pop = ValueHead(hidden_dim=self.hidden_dim)

    def __call__(self, obs, prev_cop_act, h_prev, cond_br, cond_pop, x_t, t):

        h = self.encoder(obs, prev_cop_act, h_prev)
        v_br = self.vel_br(h, cond_br, x_t, t)
        v_pop = self.vel_pop(h, cond_pop, x_t, t)
        vv_br = self.val_br(h, cond_br)
        vv_pop = self.val_pop(h, cond_pop)
        return v_br, v_pop, vv_br, vv_pop

    def encode(self, obs, prev_cop_act, h_prev):

        return self.encoder(obs, prev_cop_act, h_prev)

    def velocity_br(self, obs, prev_cop_act, h_prev, cond_br, x_t, t):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.vel_br(h, cond_br, x_t, t)

    def velocity_pop(self, obs, prev_cop_act, h_prev, cond_pop, x_t, t):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.vel_pop(h, cond_pop, x_t, t)

    def velocity_pop_sg(self, obs, prev_cop_act, h_prev, cond_pop, x_t, t):

        h = self.encoder(obs, prev_cop_act, h_prev)
        h = jax.lax.stop_gradient(h)
        return self.vel_pop(h, cond_pop, x_t, t)

    def value_br(self, obs, prev_cop_act, h_prev, cond_br):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.val_br(h, cond_br)

    def value_pop(self, obs, prev_cop_act, h_prev, cond_pop):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.val_pop(h, cond_pop)

def flow_sample(
    apply_fn,
    params,
    obs: jnp.ndarray,
    prev_cop_act: jnp.ndarray,
    h_prev: jnp.ndarray,
    cond: jnp.ndarray,
    rng: jax.Array,
    flow_steps: int,
    action_dim: int,
    init_noise: float,
    explore_noise: float,
    output_scale: float,
    method,
    feather_std: float = 0.0,
):

    batch_size = obs.shape[0]
    rng_init, rng_kick, rng_feather = jax.random.split(rng, 3)
    x_init = jax.random.normal(rng_init, (batch_size, action_dim)) * init_noise

    t_grid = jnp.linspace(1.0, 0.0, flow_steps + 1)

    def _step(carry, step_i):
        x, rng_k = carry
        rng_k, rng_ki = jax.random.split(rng_k)
        t_cur = jnp.full((batch_size, 1), t_grid[step_i])
        dt = t_grid[step_i + 1] - t_grid[step_i]
        v = apply_fn(params, obs, prev_cop_act, h_prev, cond, x, t_cur,
                     method=method)
        v = v * output_scale
        x_new = x + dt * v
        if explore_noise > 0.0:
            kick = jax.random.normal(rng_ki, x.shape) * explore_noise * jnp.sqrt(jnp.abs(dt))
            x_new = x_new + kick
        return (x_new, rng_k), None

    (x_final, _), _ = jax.lax.scan(
        _step, (x_init, rng_kick), jnp.arange(flow_steps))

    if feather_std > 0.0:
        x_final = x_final + jax.random.normal(rng_feather, x_final.shape) * feather_std
    return x_final

def flow_velocity(apply_fn, params, obs, prev_cop_act, h_prev, cond, x_t, t,
                  method):
    return apply_fn(params, obs, prev_cop_act, h_prev, cond, x_t, t,
                    method=method)

def flow_value(apply_fn, params, obs, prev_cop_act, h_prev, cond, method):
    return apply_fn(params, obs, prev_cop_act, h_prev, cond, method=method)

def compute_cfm_loss(
    apply_fn,
    params,
    obs: jnp.ndarray,
    prev_cop_act: jnp.ndarray,
    h_prev: jnp.ndarray,
    cond: jnp.ndarray,
    action: jnp.ndarray,
    loss_eps: jnp.ndarray,
    loss_t: jnp.ndarray,
    output_scale: float,
    output_mode: str,
    method,
) -> jnp.ndarray:

    B = obs.shape[0]
    n = loss_eps.shape[1]
    d = loss_eps.shape[-1]

    action_exp = action[:, None, :]
    x_t = loss_t * loss_eps + (1.0 - loss_t) * action_exp

    obs_exp = jnp.broadcast_to(obs[:, None, :], (B, n, obs.shape[-1]))
    pca_exp = jnp.broadcast_to(
        prev_cop_act[:, None, :], (B, n, prev_cop_act.shape[-1]))
    hp_exp = jnp.broadcast_to(
        h_prev[:, None, :], (B, n, h_prev.shape[-1]))
    cond_exp = jnp.broadcast_to(cond[:, None, :], (B, n, cond.shape[-1]))

    obs_flat = obs_exp.reshape(-1, obs.shape[-1])
    pca_flat = pca_exp.reshape(-1, prev_cop_act.shape[-1])
    hp_flat = hp_exp.reshape(-1, h_prev.shape[-1])
    cond_flat = cond_exp.reshape(-1, cond.shape[-1])
    x_t_flat = x_t.reshape(-1, d)
    t_flat = loss_t.reshape(-1, 1)

    v = apply_fn(params, obs_flat, pca_flat, hp_flat, cond_flat, x_t_flat,
                 t_flat, method=method)
    v = v * output_scale
    v = v.reshape(B, n, d)

    if output_mode == "u":
        v_gt = loss_eps - action_exp
        sq_err = (v - v_gt) ** 2
    elif output_mode == "u_but_supervise_as_eps":
        x0_pred = x_t - loss_t * v
        x1_pred = x0_pred + v
        sq_err = (loss_eps - x1_pred) ** 2
    else:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    sq_err = jnp.clip(sq_err, 0.0, 1e6)
    return sq_err.mean(axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    raw_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    prev_cop_act: jnp.ndarray
    h_prev: jnp.ndarray
    g_embed: jnp.ndarray
    loss_eps: jnp.ndarray
    loss_t: jnp.ndarray
    initial_cfm_loss: jnp.ndarray
    used_cce: jnp.ndarray

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
    ref_params,
    cop_params,
    cop_network: FlowActorCritic,
    cop_embeds: jnp.ndarray,
    sigma_cce: jnp.ndarray,
    old_embeds: jnp.ndarray,
    player_idx: int,
    cfg: Config,
    action_dim_p: int,
    action_dim_c: int,
    reg_n_slots_actual: int,
):

    agents = env.agents
    player_agent = agents[player_idx]
    cop_agent = agents[1 - player_idx]
    has_old_slots = old_embeds.shape[0] > 0 and reg_n_slots_actual > 0
    num_cop_strats = cop_embeds.shape[0]
    dirichlet_alpha = jnp.full((num_cop_strats,), cfg.sigma_dirichlet_alpha)

    mem_dim_eff = cfg.memory_dim if cfg.memory_dim > 0 else cfg.hidden_dim

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=cfg.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=init_params, tx=tx)

    t_schedule = jnp.linspace(1.0, 0.0, cfg.flow_steps + 1)[:-1]

    def _sample_sigma_batch(rng):

        rng_mix, rng_dir = jax.random.split(rng)
        use_cce = (jax.random.uniform(rng_mix, (cfg.num_envs,))
                   < cfg.sigma_cce_prob)
        sigma_dir = jax.random.dirichlet(
            rng_dir, dirichlet_alpha, (cfg.num_envs,))
        sigma_cce_b = jnp.broadcast_to(
            sigma_cce[None, :], (cfg.num_envs, num_cop_strats))
        sigma_b = jnp.where(use_cce[:, None], sigma_cce_b, sigma_dir)
        return sigma_b, use_cce

    def _env_step(runner_state, unused):
        (train_state, env_state, last_obs,
         h_prev_p, h_prev_c, prev_act_cop_of_p, prev_act_cop_of_c,
         rng) = runner_state
        rng, rng_p, rng_c, rng_step, rng_sig, rng_idx, rng_eps, rng_t = \
            jax.random.split(rng, 8)

        net_params = train_state.params["net"]

        p_obs = last_obs[player_agent]
        c_obs = last_obs[cop_agent]

        sigma_b, use_cce_b = _sample_sigma_batch(rng_sig)
        g_batch = sigma_b @ cop_embeds

        x_p_raw = flow_sample(
            network.apply, net_params, p_obs, prev_act_cop_of_p, h_prev_p,
            g_batch, rng_p,
            cfg.flow_steps, action_dim_p,
            cfg.flow_init_noise, cfg.flow_explore_noise,
            cfg.policy_output_scale,
            method=FlowActorCritic.velocity_br,
            feather_std=cfg.feather_std,
        )
        p_action = jnp.clip(x_p_raw, -1.0, 1.0)
        val_p = flow_value(
            network.apply, net_params, p_obs, prev_act_cop_of_p, h_prev_p,
            g_batch, method=FlowActorCritic.value_br)

        n = cfg.n_samples_per_action
        loss_eps = (jax.random.normal(rng_eps, (cfg.num_envs, n, action_dim_p))
                    * cfg.flow_init_noise)
        if cfg.discretize_t:
            idx_t = jax.random.randint(rng_t, (cfg.num_envs, n), 0, cfg.flow_steps)
            loss_t = t_schedule[idx_t][..., None]
        else:
            loss_t = jax.random.uniform(
                rng_t, (cfg.num_envs, n, 1), minval=1e-3, maxval=1.0)

        initial_cfm = compute_cfm_loss(
            network.apply, jax.lax.stop_gradient(net_params),
            p_obs, prev_act_cop_of_p, h_prev_p,
            g_batch, x_p_raw, loss_eps, loss_t,
            cfg.policy_output_scale, cfg.output_mode,
            method=FlowActorCritic.velocity_br,
        )

        u_gumbel = jax.random.uniform(rng_idx, (cfg.num_envs, num_cop_strats),
                                      minval=1e-20, maxval=1.0)
        gumbel = -jnp.log(-jnp.log(u_gumbel))
        logits_b = jnp.log(jnp.clip(sigma_b, 1e-20, 1.0))
        strat_idx = jnp.argmax(logits_b + gumbel, axis=-1)
        c_embed = cop_embeds[strat_idx]
        x_c_raw = flow_sample(
            cop_network.apply, cop_params, c_obs, prev_act_cop_of_c, h_prev_c,
            c_embed, rng_c,
            cfg.flow_steps, action_dim_c,
            cfg.flow_init_noise, 0.0, cfg.policy_output_scale,
            method=FlowActorCritic.velocity_pop,
        )
        c_action = jnp.clip(x_c_raw, -1.0, 1.0)

        h_new_p = network.apply(
            net_params, p_obs, prev_act_cop_of_p, h_prev_p,
            method=FlowActorCritic.encode)
        h_new_c = cop_network.apply(
            cop_params, c_obs, prev_act_cop_of_c, h_prev_c,
            method=FlowActorCritic.encode)

        if player_idx == 0:
            actions = {agents[0]: p_action, agents[1]: c_action}
        else:
            actions = {agents[0]: c_action, agents[1]: p_action}

        rng_steps = jax.random.split(rng_step, cfg.num_envs)
        next_obs, next_state, rewards, dones, info = jax.vmap(env.step)(
            rng_steps, env_state, actions)

        done_p = dones[player_agent][:, None].astype(jnp.float32)
        done_c = dones[cop_agent][:, None].astype(jnp.float32)
        h_prev_p_next = h_new_p * (1.0 - done_p)
        h_prev_c_next = h_new_c * (1.0 - done_c)

        prev_act_cop_of_p_next = c_action * (1.0 - done_p)
        prev_act_cop_of_c_next = p_action * (1.0 - done_c)

        transition = Transition(
            done=dones[player_agent],
            raw_action=x_p_raw,
            value=val_p,
            reward=rewards[player_agent],
            obs=p_obs,
            prev_cop_act=prev_act_cop_of_p,
            h_prev=h_prev_p,
            g_embed=g_batch,
            loss_eps=loss_eps,
            loss_t=loss_t,
            initial_cfm_loss=initial_cfm,
            used_cce=use_cce_b,
        )
        return (train_state, next_state, next_obs,
                h_prev_p_next, h_prev_c_next,
                prev_act_cop_of_p_next, prev_act_cop_of_c_next,
                rng), transition

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
        (rs_core, best_ema, best_prev_ema, best_params) = runner_state
        rs_core, traj_batch = jax.lax.scan(_env_step, rs_core, None, cfg.num_steps)
        (train_state, env_state, last_obs,
         h_prev_p, h_prev_c, prev_act_cop_of_p, prev_act_cop_of_c,
         rng) = rs_core

        p_obs_last = last_obs[player_agent]
        net_params = train_state.params["net"]
        rng, rng_bs = jax.random.split(rng)
        sigma_bs, _ = _sample_sigma_batch(rng_bs)
        g_bs = sigma_bs @ cop_embeds
        last_val = flow_value(
            network.apply, net_params, p_obs_last,
            prev_act_cop_of_p, h_prev_p, g_bs,
            method=FlowActorCritic.value_br)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        if cfg.global_adv_norm:
            adv_flat = advantages.reshape(-1)
            adv_mean = adv_flat.mean()
            adv_std = adv_flat.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        def _update_epoch(update_state, unused):
            def _update_minibatch(train_state, batch_info):
                traj_batch, advantages, targets, rng_mb = batch_info

                def _loss_fn(params, traj_batch, gae, targets, rng_mb):
                    net_params = params["net"]
                    own_embed = params["embed"]
                    B = traj_batch.obs.shape[0]
                    d_emb = own_embed.shape[-1]
                    d_act = traj_batch.raw_action.shape[-1]
                    d_obs = traj_batch.obs.shape[-1]
                    d_pca = traj_batch.prev_cop_act.shape[-1]
                    d_hp = traj_batch.h_prev.shape[-1]
                    own_embed_b = jnp.broadcast_to(own_embed, (B, d_emb))

                    value_pred = flow_value(
                        network.apply, net_params,
                        traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, traj_batch.g_embed,
                        method=FlowActorCritic.value_br)
                    if cfg.value_clip > 0.0:
                        value_pred_clipped = traj_batch.value + jnp.clip(
                            value_pred - traj_batch.value,
                            -cfg.value_clip, cfg.value_clip)
                        vl_unclipped = (value_pred - targets) ** 2
                        vl_clipped = (value_pred_clipped - targets) ** 2
                        value_loss = 0.5 * jnp.mean(
                            jnp.maximum(vl_unclipped, vl_clipped))
                    else:
                        value_loss = 0.5 * jnp.mean((value_pred - targets) ** 2)

                    if cfg.global_adv_norm:
                        adv_norm = gae
                    else:
                        adv_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    adv_exp = adv_norm[:, None]

                    current_cfm = compute_cfm_loss(
                        network.apply, net_params,
                        traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, traj_batch.g_embed,
                        traj_batch.raw_action,
                        traj_batch.loss_eps, traj_batch.loss_t,
                        cfg.policy_output_scale, cfg.output_mode,
                        method=FlowActorCritic.velocity_br,
                    )

                    cfm_old = traj_batch.initial_cfm_loss
                    cfm_cur = current_cfm
                    if cfg.cfm_loss_clamp_max > 0.0:
                        cfm_old = jnp.minimum(cfm_old, cfg.cfm_loss_clamp_max)
                        cfm_cur = jnp.minimum(cfm_cur, cfg.cfm_loss_clamp_max)

                    if cfg.neg_adv_cur_cfm_clamp_max > 0.0:
                        adv_neg_mask = (adv_exp < 0.0)
                        cfm_cur_neg = jnp.minimum(
                            cfm_cur, cfg.neg_adv_cur_cfm_clamp_max)
                        cfm_cur = jnp.where(adv_neg_mask, cfm_cur_neg, cfm_cur)

                    cfm_diff = cfm_old - cfm_cur

                    if cfg.cfm_diff_clamp_ste:
                        cfm_diff_c = clamp_ste(cfm_diff, hi=cfg.cfm_loss_clip)
                    else:
                        cfm_diff_c = jnp.clip(
                            cfm_diff, -cfg.cfm_loss_clip, cfg.cfm_loss_clip)
                    rho_s = jnp.exp(cfm_diff_c)

                    logratio = cfm_diff_c.mean(axis=-1)
                    ratio_mean_b = rho_s.mean(axis=-1)
                    approx_kl = jnp.mean((ratio_mean_b - 1.0) - logratio)

                    eps_clip = cfg.ratio_clip

                    surr1 = adv_exp * rho_s
                    surr2 = adv_exp * jnp.clip(
                        rho_s, 1.0 - eps_clip, 1.0 + eps_clip)
                    psi_ppo = jnp.minimum(surr1, surr2)

                    psi_spo = (rho_s * adv_exp
                               - jnp.abs(adv_exp) / (2.0 * eps_clip)
                               * (rho_s - 1.0) ** 2)
                    psi = jnp.where(adv_exp >= 0, psi_ppo, psi_spo)
                    policy_loss = -psi.mean()

                    rho_mean = rho_s.mean()
                    rho_std = rho_s.std()
                    rho_min = rho_s.min()
                    rho_max = rho_s.max()
                    clip_frac = (jnp.abs(rho_s - 1.0) > eps_clip).mean()

                    bc_loss = current_cfm.mean()

                    pop_method = (FlowActorCritic.velocity_pop_sg
                                  if cfg.decouple_encoder
                                  else FlowActorCritic.velocity_pop)
                    distill_cfm = compute_cfm_loss(
                        network.apply, net_params,
                        traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, own_embed_b,
                        traj_batch.raw_action,
                        traj_batch.loss_eps, traj_batch.loss_t,
                        cfg.policy_output_scale, cfg.output_mode,
                        method=pop_method,
                    )

                    cce_mask_b = traj_batch.used_cce.astype(jnp.float32)
                    cce_mask_bn = cce_mask_b[:, None]
                    mask_sum = jnp.clip(cce_mask_b.sum(), 1.0)
                    distill_loss = ((distill_cfm * cce_mask_bn).sum()
                                    / (mask_sum * distill_cfm.shape[-1]))

                    if has_old_slots:
                        rng_slot, rng_re, rng_rt = jax.random.split(rng_mb, 3)
                        K = reg_n_slots_actual
                        T_old = old_embeds.shape[0]

                        idx = jax.random.randint(rng_slot, (K,), 0, T_old)
                        tau_embeds = old_embeds[idx]

                        obs_e = jnp.broadcast_to(
                            traj_batch.obs[:, None, :], (B, K, d_obs))
                        pca_e = jnp.broadcast_to(
                            traj_batch.prev_cop_act[:, None, :], (B, K, d_pca))
                        hp_e = jnp.broadcast_to(
                            traj_batch.h_prev[:, None, :], (B, K, d_hp))
                        tau_e = jnp.broadcast_to(
                            tau_embeds[None, :, :], (B, K, d_emb))
                        x0_e = jnp.broadcast_to(
                            traj_batch.raw_action[:, None, :], (B, K, d_act))

                        eps_r = (jax.random.normal(rng_re, (B, K, d_act))
                                 * cfg.flow_init_noise)
                        if cfg.discretize_t:
                            idx_t = jax.random.randint(
                                rng_rt, (B, K), 0, cfg.flow_steps)
                            t_r = t_schedule[idx_t][..., None]
                        else:
                            t_r = jax.random.uniform(
                                rng_rt, (B, K, 1), minval=1e-3, maxval=1.0)

                        x_t_r = t_r * eps_r + (1.0 - t_r) * x0_e

                        obs_flat = obs_e.reshape(-1, d_obs)
                        pca_flat = pca_e.reshape(-1, d_pca)
                        hp_flat = hp_e.reshape(-1, d_hp)
                        emb_flat = tau_e.reshape(-1, d_emb)
                        xt_flat = x_t_r.reshape(-1, d_act)
                        t_flat = t_r.reshape(-1, 1)

                        reg_pop_method = (FlowActorCritic.velocity_pop_sg
                                          if cfg.decouple_encoder
                                          else FlowActorCritic.velocity_pop)
                        v_cur = network.apply(
                            net_params, obs_flat, pca_flat, hp_flat,
                            emb_flat, xt_flat, t_flat,
                            method=reg_pop_method)
                        v_ref = network.apply(
                            ref_params, obs_flat, pca_flat, hp_flat,
                            emb_flat, xt_flat, t_flat,
                            method=FlowActorCritic.velocity_pop)
                        v_cur = v_cur * cfg.policy_output_scale
                        v_ref = jax.lax.stop_gradient(
                            v_ref * cfg.policy_output_scale)
                        reg_loss = jnp.mean((v_cur - v_ref) ** 2)
                    else:
                        reg_loss = jnp.asarray(0.0)

                    total = (policy_loss
                             + cfg.vf_coef * value_loss
                             + cfg.bc_loss_scale * bc_loss
                             + cfg.distill_beta * distill_loss
                             + cfg.reg_beta * reg_loss)

                    return total, (policy_loss, value_loss, bc_loss,
                                   distill_loss, reg_loss, approx_kl,
                                   rho_mean, rho_std, rho_min, rho_max,
                                   clip_frac, current_cfm.mean())

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss_val, aux), grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets,
                    rng_mb)
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
        update_state, epoch_losses = jax.lax.scan(
            _update_epoch, update_state, None, cfg.update_epochs)
        train_state = update_state[0]
        rng = update_state[-1]

        kl_mean = jnp.mean(epoch_losses[6])

        if cfg.use_adaptive_lr:
            clip_s, inject_s = train_state.opt_state
            current_lr = inject_s.hyperparams["learning_rate"]
            lr_dn = current_lr / cfg.kl_lr_factor
            lr_up = current_lr * cfg.kl_lr_factor
            new_lr = jnp.where(
                kl_mean > cfg.desired_kl * 2.0, lr_dn,
                jnp.where(
                    (kl_mean < cfg.desired_kl / 2.0) & (kl_mean > 0.0),
                    lr_up, current_lr))
            new_lr = jnp.clip(new_lr, cfg.lr_min, cfg.lr_max)
            new_hp = {**inject_s.hyperparams, "learning_rate": new_lr}
            new_inject = inject_s._replace(hyperparams=new_hp)
            train_state = train_state.replace(
                opt_state=(clip_s, new_inject))
            lr_for_stats = new_lr
        else:
            lr_for_stats = jnp.asarray(cfg.lr, dtype=jnp.float32)

        stats = {
            "policy_loss": jnp.mean(epoch_losses[1]),
            "value_loss":  jnp.mean(epoch_losses[2]),
            "distill_loss": jnp.mean(epoch_losses[4]),
            "reg_loss":    jnp.mean(epoch_losses[5]),
            "approx_kl":   kl_mean,
            "rho_mean":    jnp.mean(epoch_losses[7]),
            "rho_std":     jnp.mean(epoch_losses[8]),
            "rho_min":     jnp.min(epoch_losses[9]),
            "rho_max":     jnp.max(epoch_losses[10]),
            "clip_frac":   jnp.mean(epoch_losses[11]),
            "lr":          lr_for_stats,
        }

        mean_return = traj_batch.reward.sum(axis=0).mean()

        alpha = cfg.best_ema_alpha
        new_prev_ema = alpha * best_prev_ema + (1.0 - alpha) * mean_return
        is_better = new_prev_ema > best_ema
        new_best_ema = jnp.where(is_better, new_prev_ema, best_ema)
        new_best_params = jax.tree.map(
            lambda b, c: jnp.where(is_better, c, b),
            best_params, train_state.params)

        rs_core_new = (train_state, env_state, last_obs,
                       h_prev_p, h_prev_c,
                       prev_act_cop_of_p, prev_act_cop_of_c,
                       rng)
        return ((rs_core_new, new_best_ema, new_prev_ema, new_best_params),
                (mean_return, stats))

    rng, rng_reset = jax.random.split(rng)
    reset_rngs = jax.random.split(rng_reset, cfg.num_envs)
    obs, env_state = jax.vmap(env.reset)(reset_rngs)

    h_prev_p0 = jnp.zeros((cfg.num_envs, mem_dim_eff))
    h_prev_c0 = jnp.zeros((cfg.num_envs, mem_dim_eff))
    prev_act_cp0 = jnp.zeros((cfg.num_envs, action_dim_c))
    prev_act_cc0 = jnp.zeros((cfg.num_envs, action_dim_p))

    rng, rng_train = jax.random.split(rng)
    rs_core0 = (train_state, env_state, obs,
                h_prev_p0, h_prev_c0, prev_act_cp0, prev_act_cc0,
                rng_train)
    best_ema0 = jnp.asarray(-1e9, dtype=jnp.float32)
    best_prev_ema0 = jnp.asarray(0.0, dtype=jnp.float32)
    best_params0 = jax.tree.map(jnp.copy, init_params)
    runner_state = (rs_core0, best_ema0, best_prev_ema0, best_params0)
    runner_state, (returns, stats_per_step) = jax.lax.scan(
        _update_step, runner_state, None, cfg.br_steps)

    (rs_core_final, best_ema_final, _, best_params_final) = runner_state
    train_state_final = rs_core_final[0]

    return train_state_final, {
        "returns": returns,
        "stats": stats_per_step,
        "best_params": best_params_final,
        "best_ema_return": best_ema_final,
    }

def eval_joint_strategy(
    rng, env, network_0, params_0, network_1, params_1,
    embed_0, embed_1, num_envs, num_steps, cfg: Config,
    action_dim_0: int, action_dim_1: int,
):

    agents = env.agents
    mem_dim = cfg.memory_dim if cfg.memory_dim > 0 else cfg.hidden_dim

    def _step(carry, unused):
        (env_state, obs, h0, h1, pa_of_0, pa_of_1, rng) = carry
        rng, rng_0, rng_1, rng_step = jax.random.split(rng, 4)

        e0 = jnp.broadcast_to(embed_0, (num_envs, embed_0.shape[-1]))
        e1 = jnp.broadcast_to(embed_1, (num_envs, embed_1.shape[-1]))

        obs0 = obs[agents[0]]
        obs1 = obs[agents[1]]

        x0_raw = flow_sample(
            network_0.apply, params_0, obs0, pa_of_0, h0, e0, rng_0,
            cfg.flow_steps, action_dim_0, cfg.flow_init_noise, 0.0,
            cfg.policy_output_scale,
            method=FlowActorCritic.velocity_pop,
        )
        x1_raw = flow_sample(
            network_1.apply, params_1, obs1, pa_of_1, h1, e1, rng_1,
            cfg.flow_steps, action_dim_1, cfg.flow_init_noise, 0.0,
            cfg.policy_output_scale,
            method=FlowActorCritic.velocity_pop,
        )

        a0 = jnp.clip(x0_raw, -1.0, 1.0)
        a1 = jnp.clip(x1_raw, -1.0, 1.0)

        h0_new = network_0.apply(
            params_0, obs0, pa_of_0, h0, method=FlowActorCritic.encode)
        h1_new = network_1.apply(
            params_1, obs1, pa_of_1, h1, method=FlowActorCritic.encode)

        actions = {agents[0]: a0, agents[1]: a1}
        rng_steps = jax.random.split(rng_step, num_envs)
        next_obs, next_state, rewards, dones, info = jax.vmap(env.step)(
            rng_steps, env_state, actions)

        d0 = dones[agents[0]][:, None].astype(jnp.float32)
        d1 = dones[agents[1]][:, None].astype(jnp.float32)
        h0_next = h0_new * (1.0 - d0)
        h1_next = h1_new * (1.0 - d1)
        pa_of_0_next = a1 * (1.0 - d0)
        pa_of_1_next = a0 * (1.0 - d1)

        return ((next_state, next_obs, h0_next, h1_next,
                 pa_of_0_next, pa_of_1_next, rng),
                rewards[agents[0]])

    rng, rng_reset = jax.random.split(rng)
    reset_rngs = jax.random.split(rng_reset, num_envs)
    obs, env_state = jax.vmap(env.reset)(reset_rngs)

    h0_init = jnp.zeros((num_envs, mem_dim))
    h1_init = jnp.zeros((num_envs, mem_dim))
    pa_of_0_init = jnp.zeros((num_envs, action_dim_1))
    pa_of_1_init = jnp.zeros((num_envs, action_dim_0))

    _, rewards = jax.lax.scan(
        _step,
        (env_state, obs, h0_init, h1_init, pa_of_0_init, pa_of_1_init, rng),
        None, num_steps)
    return rewards.sum(axis=0).mean()

def rollout_trajectory(
    rng, env_raw, network_0, params_0, network_1, params_1,
    embed_0, embed_1, num_steps, cfg: Config,
    action_dim_0: int, action_dim_1: int,
):
    agents = env_raw.agents
    mem_dim = cfg.memory_dim if cfg.memory_dim > 0 else cfg.hidden_dim

    def _step(carry, unused):
        (env_state, obs, h0, h1, pa_of_0, pa_of_1, rng) = carry
        rng, rng_0, rng_1, rng_step = jax.random.split(rng, 4)

        e0 = embed_0[None]
        e1 = embed_1[None]
        obs0 = obs[agents[0]][None]
        obs1 = obs[agents[1]][None]

        x0_raw = flow_sample(
            network_0.apply, params_0, obs0, pa_of_0, h0, e0, rng_0,
            cfg.flow_steps, action_dim_0, cfg.flow_init_noise, 0.0,
            cfg.policy_output_scale,
            method=FlowActorCritic.velocity_pop,
        )
        x1_raw = flow_sample(
            network_1.apply, params_1, obs1, pa_of_1, h1, e1, rng_1,
            cfg.flow_steps, action_dim_1, cfg.flow_init_noise, 0.0,
            cfg.policy_output_scale,
            method=FlowActorCritic.velocity_pop,
        )
        a0 = jnp.clip(x0_raw[0], -1.0, 1.0)
        a1 = jnp.clip(x1_raw[0], -1.0, 1.0)

        h0_new = network_0.apply(
            params_0, obs0, pa_of_0, h0, method=FlowActorCritic.encode)
        h1_new = network_1.apply(
            params_1, obs1, pa_of_1, h1, method=FlowActorCritic.encode)

        actions = {agents[0]: a0, agents[1]: a1}
        next_obs, next_state, rewards, dones, info = env_raw.step(
            rng_step, env_state, actions)

        d0 = jnp.asarray(dones[agents[0]], dtype=jnp.float32)
        d1 = jnp.asarray(dones[agents[1]], dtype=jnp.float32)
        h0_next = h0_new * (1.0 - d0)
        h1_next = h1_new * (1.0 - d1)
        pa_of_0_next = a1[None] * (1.0 - d0)
        pa_of_1_next = a0[None] * (1.0 - d1)

        return ((next_state, next_obs, h0_next, h1_next,
                 pa_of_0_next, pa_of_1_next, rng),
                next_state.pipeline_state)

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env_raw.reset(rng_reset)
    init_pipeline = env_state.pipeline_state

    h0_init = jnp.zeros((1, mem_dim))
    h1_init = jnp.zeros((1, mem_dim))
    pa_of_0_init = jnp.zeros((1, action_dim_1))
    pa_of_1_init = jnp.zeros((1, action_dim_0))

    _, traj_pipeline_states = jax.lax.scan(
        _step,
        (env_state, obs, h0_init, h1_init, pa_of_0_init, pa_of_1_init, rng),
        None, num_steps)

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

def run_fppo_jpsro(cfg: Config):
    print("=" * 70)
    print("  FPPO-JPSRO (MAFPO flow policy) for Multi-Agent HalfCheetah")
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
            name=f"fppo-jpsro-seed{cfg.seed}-v3",
        )

    env_raw = jaxmarl.make("halfcheetah_2x3")
    env = LogWrapper(env_raw, replace_info=True)
    agents = env.agents

    obs_dims = {a: env.observation_space(a).shape[0] for a in agents}
    act_dims = {a: env.action_space(a).shape[0] for a in agents}
    print(f"  Agents: {agents}")
    print(f"  Obs dims: {obs_dims}, Act dims: {act_dims}")
    print(f"  Flow: steps={cfg.flow_steps}, t_embed_dim={cfg.t_embed_dim}, "
          f"output_scale={cfg.policy_output_scale}, "
          f"init_noise={cfg.flow_init_noise}")
    print(f"  Exploration: explore_noise(SDE/step)={cfg.flow_explore_noise}, "
          f"feather_std(final)={cfg.feather_std}")
    print(f"  MAFPO/FPO: n_samples={cfg.n_samples_per_action}, "
          f"ratio_clip={cfg.ratio_clip}, cfm_loss_clip={cfg.cfm_loss_clip}, "
          f"output_mode={cfg.output_mode}")
    print(f"  FPO++ anti-collapse: ste_clamp={cfg.cfm_diff_clamp_ste}, "
          f"cfm_loss_clamp_max={cfg.cfm_loss_clamp_max}, "
          f"neg_adv_cur_cfm_clamp_max={cfg.neg_adv_cur_cfm_clamp_max}")
    print(f"  ASPO trust region: PPO (A>=0) + SPO quadratic pull-back (A<0)")
    print(f"  Adaptive LR: enabled={cfg.use_adaptive_lr}, "
          f"desired_kl={cfg.desired_kl}, factor={cfg.kl_lr_factor}, "
          f"bounds=[{cfg.lr_min:.0e},{cfg.lr_max:.0e}], base_lr={cfg.lr:.0e}")
    print(f"  Value: vf_coef={cfg.vf_coef}, value_clip={cfg.value_clip} "
          f"({'on' if cfg.value_clip > 0 else 'off'})"
          f"  | global_adv_norm={cfg.global_adv_norm}")
    print(f"  Memory: memory_dim={cfg.memory_dim}, "
          f"use_prev_cop_action={cfg.use_prev_cop_action} "
          f"| best-EMA commit alpha={cfg.best_ema_alpha}")

    action_dims = [act_dims[agents[p]] for p in range(2)]

    cop_action_dims = [action_dims[1 - p] for p in range(2)]

    networks = [
        FlowActorCritic(
            action_dim=action_dims[p],
            cond_dim=cfg.embed_dim,
            cop_action_dim=cop_action_dims[p],
            hidden_dim=cfg.hidden_dim,
            t_embed_dim=cfg.t_embed_dim,
            memory_dim=cfg.memory_dim,
            use_prev_cop_action=cfg.use_prev_cop_action,
            br_head_layers=cfg.br_head_layers,
            pop_head_layers=cfg.pop_head_layers,
        )
        for p in range(2)
    ]

    mem_dim_eff = cfg.memory_dim if cfg.memory_dim > 0 else cfg.hidden_dim

    rng, *init_rngs = jax.random.split(rng, 3)
    dummy_obs = [jnp.zeros((1, obs_dims[agents[p]])) for p in range(2)]
    dummy_pca = [jnp.zeros((1, cop_action_dims[p])) for p in range(2)]
    dummy_h = jnp.zeros((1, mem_dim_eff))
    dummy_embed = jnp.zeros((1, cfg.embed_dim))
    dummy_xt = [jnp.zeros((1, action_dims[p])) for p in range(2)]
    dummy_t = jnp.zeros((1, 1))

    net_params = [
        networks[p].init(
            init_rngs[p], dummy_obs[p], dummy_pca[p], dummy_h,
            dummy_embed, dummy_embed, dummy_xt[p], dummy_t)
        for p in range(2)
    ]

    rng, rng_e0, rng_e1 = jax.random.split(rng, 3)
    embeds = [
        [jax.random.normal(rng_e0, (cfg.embed_dim,)) * 0.1],
        [jax.random.normal(rng_e1, (cfg.embed_dim,)) * 0.1],
    ]

    param_snapshots = [
        [jax.tree.map(jnp.copy, net_params[0])],
        [jax.tree.map(jnp.copy, net_params[1])],
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
        rng_ep, net_params[0], net_params[1], embeds[0][0], embeds[1][0]))
    print(f"  Initial payoff (strat 0 vs strat 0): {initial_payoff:.2f}")

    if cfg.wandb:
        import wandb
        wandb.log({"jpsro/initial_payoff": initial_payoff}, step=0)

    payoff_matrix = np.array([[initial_payoff]])
    sigma = cce_solver_cooperative(payoff_matrix)

    rng, rng_traj = jax.random.split(rng)
    init_ps, traj_ps = _rollout(
        rng_traj, net_params[0], net_params[1], embeds[0][0], embeds[1][0])
    traj_path = save_trajectory(cfg.save_dir, "iter_000_s0_vs_s0", init_ps, traj_ps)
    print(f"  Trajectory saved: {traj_path}")

    for t in range(1, cfg.jpsro_iters + 1):
        iter_t0 = time.time()
        br_steps_iter = cfg.br_steps + cfg.br_steps_scale * (t - 1)
        cfg_iter = replace(cfg, br_steps=br_steps_iter)

        print(f"\n{'-' * 60}")
        print(f"  JPSRO Iteration {t}/{cfg.jpsro_iters}  "
              f"(br_steps={br_steps_iter})")
        print(f"{'-' * 60}")

        ref_net_params = [jax.tree.map(jnp.copy, net_params[p]) for p in range(2)]

        ref_embeds = [list(embeds[p]) for p in range(2)]

        rng, rng_ne0, rng_ne1 = jax.random.split(rng, 3)
        embeds[0].append(jax.random.normal(rng_ne0, (cfg.embed_dim,)) * 0.1)
        embeds[1].append(jax.random.normal(rng_ne1, (cfg.embed_dim,)) * 0.1)

        for p in range(2):
            cop = 1 - p
            sigma_cop = cce_marginal(sigma, p)
            sigma_cop_jnp = jnp.array(sigma_cop, dtype=jnp.float32)

            num_cop_strats = len(ref_embeds[cop])
            if sigma_cop_jnp.shape[0] < num_cop_strats:
                sigma_cop_jnp = jnp.concatenate([
                    sigma_cop_jnp,
                    jnp.zeros(num_cop_strats - sigma_cop_jnp.shape[0])])
            elif sigma_cop_jnp.shape[0] > num_cop_strats:
                sigma_cop_jnp = sigma_cop_jnp[:num_cop_strats]
            sigma_cop_jnp = sigma_cop_jnp / (sigma_cop_jnp.sum() + 1e-8)

            cop_embeds_arr = jnp.stack(ref_embeds[cop])

            own_embed = embeds[p][-1]
            if len(ref_embeds[p]) > 0:
                old_embeds_arr = jnp.stack(ref_embeds[p])
            else:
                old_embeds_arr = jnp.zeros((0, cfg.embed_dim))
            reg_n_slots_actual = int(
                min(cfg.reg_n_slots, old_embeds_arr.shape[0]))

            print(f"\n  Player {p}: two-head flow-BR (FPO+distill+reg) "
                  f"| sigma(cce support={int((sigma_cop > 1e-6).sum())}) "
                  f"x Dir(alpha={cfg.sigma_dirichlet_alpha}) "
                  f"| old_slots={old_embeds_arr.shape[0]} K={reg_n_slots_actual}")

            br_train = jax.jit(partial(
                train_br,
                env=env,
                network=networks[p],
                cop_network=networks[cop],
                player_idx=p,
                cfg=cfg_iter,
                action_dim_p=action_dims[p],
                action_dim_c=action_dims[cop],
                reg_n_slots_actual=reg_n_slots_actual,
            ))

            init_params = {"net": net_params[p], "embed": own_embed}

            t0 = time.time()
            br_state, br_info = br_train(
                rng=rng,
                init_params=init_params,
                ref_params=ref_net_params[p],
                cop_params=ref_net_params[cop],
                cop_embeds=cop_embeds_arr,
                sigma_cce=sigma_cop_jnp,
                old_embeds=old_embeds_arr,
            )
            jax.block_until_ready(br_state.params)
            elapsed = time.time() - t0

            rng, _ = jax.random.split(rng)

            br_returns = np.array(br_info["returns"])
            br_stats = {k: np.array(v) for k, v in br_info["stats"].items()}
            best_params = br_info["best_params"]
            best_ema = float(br_info["best_ema_return"])
            print(f"    BR training: {elapsed:.1f}s, "
                  f"final return: {br_returns[-1]:.2f}, "
                  f"max return: {br_returns.max():.2f}, "
                  f"best EMA: {best_ema:.2f}  <-- committed")
            print(f"    FPO stats (last step): "
                  f"rho={br_stats['rho_mean'][-1]:.3f}+/-{br_stats['rho_std'][-1]:.3f} "
                  f"[{br_stats['rho_min'][-1]:.3f},{br_stats['rho_max'][-1]:.3f}]"
                  f"  clip_frac={br_stats['clip_frac'][-1]:.3f}"
                  f"  approx_kl={br_stats['approx_kl'][-1]:.4f}"
                  f"  lr={float(br_stats['lr'][-1]):.2e}"
                  f"  policy_loss={br_stats['policy_loss'][-1]:.4f}"
                  f"  value_loss={br_stats['value_loss'][-1]:.4f}")
            returns_history.append(br_returns)

            if cfg.wandb:
                import wandb
                for step_i, ret_val in enumerate(br_returns):
                    log_entry = {
                        f"br/p{p}_return": float(ret_val),
                        f"br/p{p}_iter": t,
                        "br/global_step": global_br_step,
                    }

                    for k, v in br_stats.items():
                        log_entry[f"br/p{p}_{k}"] = float(v[step_i])
                    wandb.log(log_entry, step=global_br_step)
                    global_br_step += 1
                wandb.log({
                    f"br/p{p}_best_ema": best_ema,
                    f"br/p{p}_final": float(br_returns[-1]),
                    f"br/p{p}_max": float(br_returns.max()),
                }, step=global_br_step)

            net_params[p] = best_params["net"]
            embeds[p][-1] = best_params["embed"]

            param_snapshots[p].append(jax.tree.map(jnp.copy, net_params[p]))

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
            cfg.save_dir, t, net_params, embeds, payoff_matrix, sigma)
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
    print(f"  FPPO-JPSRO Complete")
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
        "params": net_params,
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
        ax.set_title("Flow-BR Training Returns (MAFPO)")
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
        save_path = os.path.join(cfg.save_dir, "fppo_jpsro_cheetah.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")
        plt.close(fig)

        if cfg.wandb:
            import wandb
            wandb.log({"summary/training_plot": wandb.Image(save_path)})

    except Exception as e:
        print(f"  Plotting failed: {e}")

GPU_PROFILES = {
    "local": dict(
        num_minibatches=4,
        update_epochs=3,
        br_steps=25,
        br_steps_scale=3,
        hidden_dim=256,
        memory_dim=256,
        br_head_layers=2,
        pop_head_layers=2,
    ),
    "cluster": dict(
        num_minibatches=2,
        update_epochs=4,
        br_steps=30,
        br_steps_scale=4,
        hidden_dim=384,
        memory_dim=384,

        br_head_layers=3,
        pop_head_layers=3,
    ),
}

def main():
    parser = argparse.ArgumentParser(
        description="FPPO-JPSRO (MAFPO flow policy) for Multi-Agent HalfCheetah")
    parser.add_argument("--gpu-profile", type=str, default="local",
                        choices=["local", "cluster"],
                        help="Preset bundle for {rollout, PPO, net-size, "
                             "head-depth}.  'local'=24 GB A5000, "
                             "'cluster'=~48 GB A100/L40S.")
    parser.add_argument("--jpsro-iters", type=int, default=8)
    parser.add_argument("--br-steps", type=int, default=None,
                        help="BR PPO updates per JPSRO iter. "
                             "Default: gpu-profile-dependent.")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--update-epochs", type=int, default=None,
                        help="PPO epochs per rollout.  Default: profile-dep.")
    parser.add_argument("--num-minibatches", type=int, default=None,
                        help="PPO minibatches per epoch.  Default: profile-dep.")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="MLP hidden size in encoder / heads. "
                             "Default: gpu-profile-dependent (local 256, cluster 384).")

    parser.add_argument("--br-head-layers", type=int, default=None,
                        help="# hidden Dense layers in BR VelocityHead. "
                             "Default: 2 (local) / 3 (cluster).")
    parser.add_argument("--pop-head-layers", type=int, default=None,
                        help="# hidden Dense layers in Pop VelocityHead. "
                             "Default: 2 (local) / 3 (cluster).")

    parser.add_argument("--no-decouple-encoder", action="store_true",
                        help="Disable stop_gradient on the pop-head path of "
                             "distill and reg losses (default leaves it ON).")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor for GAE")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")

    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--t-embed-dim", type=int, default=16)
    parser.add_argument("--policy-output-scale", type=float, default=0.25)
    parser.add_argument("--flow-explore-noise", type=float, default=0.0,
                        help="Per-step SDE kick inside Euler integration "
                             "(MAFPO: sde_sigma)")
    parser.add_argument("--feather-std", type=float, default=0.0,
                        help="Gaussian noise added to the post-integration "
                             "action (MAFPO: feather_std)")
    parser.add_argument("--flow-init-noise", type=float, default=1.0)
    parser.add_argument("--no-discretize-t", action="store_true",
                        help="sample training t continuously in (0, 1] instead "
                             "of from the flow schedule grid")

    parser.add_argument("--n-samples-per-action", type=int, default=8)
    parser.add_argument("--ratio-clip", type=float, default=0.1,
                        help="ASPO/FPO per-sample ratio clip (v3 rollback: "
                             "0.05 -> 0.1; matches FPO paper default and v1 "
                             "best-EMA regime).")
    parser.add_argument("--cfm-loss-clip", type=float, default=3.0,
                        help="clip (cfm_old - cfm_new) before exp()")
    parser.add_argument("--bc-loss-scale", type=float, default=0.0)
    parser.add_argument("--output-mode", type=str,
                        default="u_but_supervise_as_eps",
                        choices=["u", "u_but_supervise_as_eps"])
    parser.add_argument("--vf-coef", type=float, default=1.0,
                        help="value loss scale (MAFPO uses 10.0)")
    parser.add_argument("--value-clip", type=float, default=0.0,
                        help="PPO-style value prediction clip (0 = off; "
                             "MAFPO default 1.0)")
    parser.add_argument("--no-global-adv-norm", action="store_true",
                        help="normalize advantages per-minibatch instead of "
                             "once globally across the full rollout memory")

    parser.add_argument("--no-cfm-diff-clamp-ste", action="store_true",
                        help="disable straight-through clamp on cfm_diff "
                             "(revert to jnp.clip which zeros gradients "
                             "at saturation -- NOT recommended)")
    parser.add_argument("--cfm-loss-clamp-max", type=float, default=0.0,
                        help="one-sided high cap on old_cfm AND current_cfm "
                             "BEFORE diff; 0 disables")
    parser.add_argument("--neg-adv-cur-cfm-clamp-max", type=float, default=0.0,
                        help="one-sided high cap on current_cfm ONLY for "
                             "samples with A<0; v3 rollback -> 0 (disabled).  "
                             "Interacted badly with tight ratio_clip=0.05 and "
                             "over-regularised BR in late iters.")

    parser.add_argument("--no-adaptive-lr", action="store_true",
                        help="disable adaptive LR schedule")
    parser.add_argument("--desired-kl", type=float, default=0.02,
                        help="target approx_kl for adaptive LR (bumped from "
                             "0.01 -> 0.02 alongside ratio_clip=0.1 so the "
                             "trust region and KL schedule stay consistent).")
    parser.add_argument("--kl-lr-factor", type=float, default=1.5,
                        help="multiplicative LR adjustment factor")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="adaptive LR floor")
    parser.add_argument("--lr-max", type=float, default=1e-2,
                        help="adaptive LR ceiling")

    parser.add_argument("--reg-beta", type=float, default=1.0,
                        help="velocity-matching regulariser weight on old slots")
    parser.add_argument("--reg-n-slots", type=int, default=4,
                        help="# old slots sampled per minibatch for reg")
    parser.add_argument("--distill-beta", type=float, default=1.0,
                        help="CFM distill weight (BR -> pop head at nu^t_p)")
    parser.add_argument("--embed-lr", type=float, default=3e-4,
                        help="Adam lr for the trainable own embedding")
    parser.add_argument("--sigma-cce-prob", type=float, default=0.5,
                        help="prob. of using CCE marginal vs Dirichlet per env")
    parser.add_argument("--sigma-dirichlet-alpha", type=float, default=1.0,
                        help="Dirichlet(alpha) concentration over V_{-p}")
    parser.add_argument("--br-steps-scale", type=int, default=None,
                        help="additional BR steps per JPSRO iter beyond 1. "
                             "Default: gpu-profile-dependent (local 3, cluster 4).")

    parser.add_argument("--best-ema-alpha", type=float, default=0.9,
                        help="EMA factor for smoothing mean_return during BR;"
                             " we commit params at argmax EMA, not final")

    parser.add_argument("--memory-dim", type=int, default=None,
                        help="GRU hidden size in encoder; 0 disables memory. "
                             "Default: gpu-profile-dependent (local 256, cluster 384).")
    parser.add_argument("--no-prev-cop-action", action="store_true",
                        help="don't feed last co-player action into encoder")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results_fppo_cheetah")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="fppo-jpsro-cheetah")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    profile = GPU_PROFILES[args.gpu_profile]

    def _prof(name):
        v = getattr(args, name)
        return v if v is not None else profile[name]

    cfg = Config(
        jpsro_iters=args.jpsro_iters,
        br_steps=_prof("br_steps"),
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        update_epochs=_prof("update_epochs"),
        num_minibatches=_prof("num_minibatches"),
        embed_dim=args.embed_dim,
        hidden_dim=_prof("hidden_dim"),
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        flow_steps=args.flow_steps,
        t_embed_dim=args.t_embed_dim,
        policy_output_scale=args.policy_output_scale,
        flow_explore_noise=args.flow_explore_noise,
        feather_std=args.feather_std,
        flow_init_noise=args.flow_init_noise,
        discretize_t=(not args.no_discretize_t),
        n_samples_per_action=args.n_samples_per_action,
        ratio_clip=args.ratio_clip,
        cfm_loss_clip=args.cfm_loss_clip,
        bc_loss_scale=args.bc_loss_scale,
        output_mode=args.output_mode,
        vf_coef=args.vf_coef,
        value_clip=args.value_clip,
        global_adv_norm=(not args.no_global_adv_norm),
        cfm_diff_clamp_ste=(not args.no_cfm_diff_clamp_ste),
        cfm_loss_clamp_max=args.cfm_loss_clamp_max,
        neg_adv_cur_cfm_clamp_max=args.neg_adv_cur_cfm_clamp_max,
        use_adaptive_lr=(not args.no_adaptive_lr),
        desired_kl=args.desired_kl,
        kl_lr_factor=args.kl_lr_factor,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        reg_beta=args.reg_beta,
        reg_n_slots=args.reg_n_slots,
        distill_beta=args.distill_beta,
        decouple_encoder=(not args.no_decouple_encoder),
        br_head_layers=_prof("br_head_layers"),
        pop_head_layers=_prof("pop_head_layers"),
        embed_lr=args.embed_lr,
        sigma_cce_prob=args.sigma_cce_prob,
        sigma_dirichlet_alpha=args.sigma_dirichlet_alpha,
        br_steps_scale=_prof("br_steps_scale"),
        best_ema_alpha=args.best_ema_alpha,
        memory_dim=_prof("memory_dim"),
        use_prev_cop_action=(not args.no_prev_cop_action),
        seed=args.seed,
        save_dir=args.save_dir,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )
    run_fppo_jpsro(cfg)
    print("\nDone!")

if __name__ == "__main__":
    main()
