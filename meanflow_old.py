#!/usr/bin/env python3

from __future__ import annotations

import sys
if sys.version_info < (3, 9):
    sys.exit(
        "meanflow_jpsro_cheetah.py requires Python 3.9+ (uses dataclasses, "
        "PEP-526 annotations, JAX). Detected Python {0}.{1}. "
        "Activate the venv first:  source JaxMARL/.venv/bin/activate  "
        "or run explicitly:  python3 meanflow_jpsro_cheetah.py ...".format(
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

@dataclass
class Config:
    jpsro_iters: int = 8
    embed_dim: int = 8
    hidden_dim: int = 128

    br_steps: int = 400
    num_envs: int = 64
    episode_len: int = 256
    update_epochs: int = 6
    num_minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 1.0
    max_grad_norm: float = 2.0
    value_clip_eps: float = 0.2

    flow_steps: int = 10
    t_embed_dim: int = 16
    policy_output_scale: float = 0.25
    gen_sigma: float = 0.1
    sde_sigma: float = 0.1
    vel_clip: float = 0.0

    ratio_clip: float = 0.2
    log_ratio_clip: float = 5.0
    kl_coef: float = 0.0
    nft_ema: float = 0.0
    entropy_coef: float = 0.0

    nft_beta: float = 0.1
    nft_adv_clip_max: float = 3.0
    nft_max_gen_delta: float = 1.0
    nft_loss_clip: float = 10.0
    nft_reg_coef: float = 0.5
    nft_kl_beta: float = 1e-3
    nft_adaptive: bool = True

    consistency_coef: float = 0.1
    consistency_clip: float = 2.0

    reg_beta: float = 1.0
    reg_n_slots: int = 4
    distill_beta: float = 1.0

    embed_lr: float = 3e-4

    sigma_cce_prob: float = 0.5
    sigma_dirichlet_alpha: float = 1.0

    br_steps_scale: int = 50

    best_ema_alpha: float = 0.9

    memory_dim: int = 128
    use_prev_cop_action: bool = True

    ep_eval_episodes: int = 32

    viz_rollout_steps: int = 300

    env_name: str = "halfcheetah_2x3"

    env_homogenisation: str = ""

    seed: int = 42
    save_dir: str = "results_meanflow_cheetah"

    wandb: bool = False
    wandb_project: str = "meanflow-jpsro-cheetah"
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

class GeneratorHead(nn.Module):

    action_dim: int
    hidden_dim: int = 128
    t_embed_dim: int = 16

    @nn.compact
    def __call__(self, h_obs, cond, z, t):
        t_emb = sinusoidal_t_embed(t, self.t_embed_dim)
        x = jnp.concatenate([h_obs, cond, z, t_emb], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.silu(x)
        g = nn.Dense(self.action_dim, kernel_init=orthogonal(0.1),
                     bias_init=constant(0.0))(x)
        return g

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

    def setup(self):
        self.encoder = ObsEncoder(
            hidden_dim=self.hidden_dim,
            memory_dim=self.memory_dim,
            use_prev_cop_action=self.use_prev_cop_action,
        )
        self.gen_br = GeneratorHead(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            t_embed_dim=self.t_embed_dim,
        )
        self.gen_pop = GeneratorHead(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            t_embed_dim=self.t_embed_dim,
        )
        self.val_br = ValueHead(hidden_dim=self.hidden_dim)
        self.val_pop = ValueHead(hidden_dim=self.hidden_dim)

    def __call__(self, obs, prev_cop_act, h_prev, cond_br, cond_pop, z, t):

        h = self.encoder(obs, prev_cop_act, h_prev)
        g_br = self.gen_br(h, cond_br, z, t)
        g_pop = self.gen_pop(h, cond_pop, z, t)
        vv_br = self.val_br(h, cond_br)
        vv_pop = self.val_pop(h, cond_pop)
        return g_br, g_pop, vv_br, vv_pop

    def encode(self, obs, prev_cop_act, h_prev):

        return self.encoder(obs, prev_cop_act, h_prev)

    def generator_br(self, obs, prev_cop_act, h_prev, cond_br, z, t):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.gen_br(h, cond_br, z, t)

    def generator_pop(self, obs, prev_cop_act, h_prev, cond_pop, z, t):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.gen_pop(h, cond_pop, z, t)

    def value_br(self, obs, prev_cop_act, h_prev, cond_br):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.val_br(h, cond_br)

    def value_pop(self, obs, prev_cop_act, h_prev, cond_pop):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.val_pop(h, cond_pop)

def gen_sample(
    apply_fn,
    params,
    obs: jnp.ndarray,
    prev_cop_act: jnp.ndarray,
    h_prev: jnp.ndarray,
    cond: jnp.ndarray,
    rng: jax.Array,
    action_dim: int,
    output_scale: float,
    sde_sigma: float,
    vel_clip: float,
    method,
):

    batch_size = obs.shape[0]
    rng_z, rng_sde = jax.random.split(rng)
    z = jax.random.normal(rng_z, (batch_size, action_dim))
    t_val = jnp.ones((batch_size, 1))

    g_raw = apply_fn(params, obs, prev_cop_act, h_prev, cond, z, t_val,
                     method=method)
    g_raw = g_raw * output_scale
    if vel_clip > 0:
        g_mean = jnp.clip(g_raw, -vel_clip, vel_clip)
    else:
        g_mean = g_raw

    raw_action = g_mean
    if sde_sigma > 0:
        raw_action = raw_action + sde_sigma * jax.random.normal(
            rng_sde, raw_action.shape)
    return raw_action, z, g_mean

def gen_log_likelihood(
    apply_fn,
    params,
    obs: jnp.ndarray,
    prev_cop_act: jnp.ndarray,
    h_prev: jnp.ndarray,
    cond: jnp.ndarray,
    z: jnp.ndarray,
    action_target: jnp.ndarray,
    gen_sigma: float,
    output_scale: float,
    vel_clip: float,
    method,
):

    B = obs.shape[0]
    d = action_target.shape[-1]
    t_val = jnp.ones((B, 1))
    g_raw = apply_fn(params, obs, prev_cop_act, h_prev, cond, z, t_val,
                     method=method)
    g_raw = g_raw * output_scale
    g_mean = jnp.clip(g_raw, -vel_clip, vel_clip) if vel_clip > 0 else g_raw
    diff = action_target - g_mean
    log_lik = (
        -0.5 * jnp.sum(diff ** 2, axis=-1) / (gen_sigma ** 2)
        - 0.5 * d * jnp.log(2.0 * jnp.pi * gen_sigma ** 2)
    )
    return log_lik, g_raw

def gen_call(apply_fn, params, obs, prev_cop_act, h_prev, cond, z, t, method):

    return apply_fn(params, obs, prev_cop_act, h_prev, cond, z, t,
                    method=method)

def flow_value(apply_fn, params, obs, prev_cop_act, h_prev, cond, method):
    return apply_fn(params, obs, prev_cop_act, h_prev, cond, method=method)

class Transition(NamedTuple):
    done: jnp.ndarray
    raw_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    prev_cop_act: jnp.ndarray
    h_prev: jnp.ndarray
    g_embed: jnp.ndarray
    rollout_noise: jnp.ndarray

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
        optax.adam(cfg.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=init_params, tx=tx)

    time_values = jnp.linspace(1.0 / cfg.flow_steps, 1.0, cfg.flow_steps)

    def _sample_sigma_batch(rng):

        rng_mix, rng_dir = jax.random.split(rng)
        use_cce = (jax.random.uniform(rng_mix, (cfg.num_envs,))
                   < cfg.sigma_cce_prob)
        sigma_dir = jax.random.dirichlet(
            rng_dir, dirichlet_alpha, (cfg.num_envs,))
        sigma_cce_b = jnp.broadcast_to(
            sigma_cce[None, :], (cfg.num_envs, num_cop_strats))
        sigma_b = jnp.where(use_cce[:, None], sigma_cce_b, sigma_dir)
        return sigma_b

    def _env_step(runner_state, unused):
        (train_state, env_state, last_obs,
         h_prev_p, h_prev_c, prev_act_cop_of_p, prev_act_cop_of_c,
         rng) = runner_state
        rng, rng_p, rng_c, rng_step, rng_sig, rng_idx = \
            jax.random.split(rng, 6)

        net_params = train_state.params["net"]

        p_obs = last_obs[player_agent]
        c_obs = last_obs[cop_agent]

        sigma_b = _sample_sigma_batch(rng_sig)
        g_batch = sigma_b @ cop_embeds

        raw_action_p, rollout_z_p, _ = gen_sample(
            network.apply, net_params, p_obs, prev_act_cop_of_p, h_prev_p,
            g_batch, rng_p,
            action_dim_p, cfg.policy_output_scale,
            cfg.sde_sigma, cfg.vel_clip,
            method=FlowActorCritic.generator_br,
        )
        p_action = jnp.tanh(raw_action_p)
        val_p = flow_value(
            network.apply, net_params, p_obs, prev_act_cop_of_p, h_prev_p,
            g_batch, method=FlowActorCritic.value_br)

        u_gumbel = jax.random.uniform(rng_idx, (cfg.num_envs, num_cop_strats),
                                      minval=1e-20, maxval=1.0)
        gumbel = -jnp.log(-jnp.log(u_gumbel))
        logits_b = jnp.log(jnp.clip(sigma_b, 1e-20, 1.0))
        strat_idx = jnp.argmax(logits_b + gumbel, axis=-1)
        c_embed = cop_embeds[strat_idx]

        raw_action_c, _, _ = gen_sample(
            cop_network.apply, cop_params, c_obs, prev_act_cop_of_c, h_prev_c,
            c_embed, rng_c,
            action_dim_c, cfg.policy_output_scale,
            0.0, cfg.vel_clip,
            method=FlowActorCritic.generator_pop,
        )
        c_action = jnp.tanh(raw_action_c)

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
            raw_action=jax.lax.stop_gradient(raw_action_p),
            value=val_p,
            reward=rewards[player_agent],
            obs=p_obs,
            prev_cop_act=prev_act_cop_of_p,
            h_prev=h_prev_p,
            g_embed=g_batch,
            rollout_noise=rollout_z_p,
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
        (rs_core, old_params, best_ema, best_prev_ema, best_params) = runner_state
        rs_core, traj_batch = jax.lax.scan(_env_step, rs_core, None, cfg.episode_len)
        (train_state, env_state, last_obs,
         h_prev_p, h_prev_c, prev_act_cop_of_p, prev_act_cop_of_c,
         rng) = rs_core

        p_obs_last = last_obs[player_agent]
        net_params = train_state.params["net"]
        rng, rng_bs = jax.random.split(rng)
        sigma_bs = _sample_sigma_batch(rng_bs)
        g_bs = sigma_bs @ cop_embeds
        last_val = flow_value(
            network.apply, net_params, p_obs_last,
            prev_act_cop_of_p, h_prev_p, g_bs,
            method=FlowActorCritic.value_br)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state, unused):
            def _update_minibatch(carry, batch_info):
                train_state, old_p = carry
                traj_batch, advantages, targets, rng_mb = batch_info

                def _loss_fn(params, traj_batch, gae, targets, old_p, rng_mb):
                    net_params = params["net"]
                    own_embed = params["embed"]
                    B = traj_batch.obs.shape[0]
                    d_emb = own_embed.shape[-1]
                    d_act = traj_batch.raw_action.shape[-1]
                    d_obs = traj_batch.obs.shape[-1]
                    d_pca = traj_batch.prev_cop_act.shape[-1]
                    d_hp = traj_batch.h_prev.shape[-1]
                    own_embed_b = jnp.broadcast_to(own_embed, (B, d_emb))

                    rng_mb, rng_nft, rng_con1, rng_con2 = jax.random.split(
                        rng_mb, 4)

                    cur_ll, g_out_cur = gen_log_likelihood(
                        network.apply, net_params,
                        traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, traj_batch.g_embed,
                        traj_batch.rollout_noise, traj_batch.raw_action,
                        cfg.gen_sigma, cfg.policy_output_scale, cfg.vel_clip,
                        method=FlowActorCritic.generator_br,
                    )
                    old_ll_val, _ = gen_log_likelihood(
                        network.apply, old_p,
                        traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, traj_batch.g_embed,
                        traj_batch.rollout_noise, traj_batch.raw_action,
                        cfg.gen_sigma, cfg.policy_output_scale, cfg.vel_clip,
                        method=FlowActorCritic.generator_br,
                    )
                    old_ll = jax.lax.stop_gradient(old_ll_val)

                    log_ratio = jnp.clip(cur_ll - old_ll,
                                         -cfg.log_ratio_clip, cfg.log_ratio_clip)
                    ratio = jnp.exp(log_ratio)
                    approx_kl = 0.5 * (log_ratio ** 2).mean()

                    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    s1 = ratio * gae_norm
                    s2 = jnp.clip(ratio, 1.0 - cfg.ratio_clip,
                                  1.0 + cfg.ratio_clip) * gae_norm
                    policy_loss = -jnp.minimum(s1, s2).mean()
                    clip_frac = (jnp.abs(ratio - 1.0) > cfg.ratio_clip).mean()

                    z_nft = jax.random.normal(rng_nft, (B, d_act))
                    t_one = jnp.ones((B, 1))
                    g_current = (gen_call(
                        network.apply, net_params,
                        traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, traj_batch.g_embed,
                        z_nft, t_one,
                        method=FlowActorCritic.generator_br)
                        * cfg.policy_output_scale)
                    g_old = jax.lax.stop_gradient(
                        gen_call(
                            network.apply, old_p,
                            traj_batch.obs, traj_batch.prev_cop_act,
                            traj_batch.h_prev, traj_batch.g_embed,
                            z_nft, t_one,
                            method=FlowActorCritic.generator_br)
                        * cfg.policy_output_scale)

                    g_delta = jnp.clip(
                        g_current - g_old,
                        -cfg.nft_max_gen_delta, cfg.nft_max_gen_delta)
                    g_clipped = g_old + g_delta

                    x0 = traj_batch.raw_action
                    x0_pos = cfg.nft_beta * g_clipped + (1.0 - cfg.nft_beta) * g_old
                    x0_neg = (1.0 + cfg.nft_beta) * g_old - cfg.nft_beta * g_clipped

                    adv_clip = jnp.clip(
                        gae_norm, -cfg.nft_adv_clip_max, cfg.nft_adv_clip_max)
                    r_nft = jnp.clip(
                        (adv_clip / cfg.nft_adv_clip_max) / 2.0 + 0.5, 0.0, 1.0)

                    if cfg.nft_adaptive:
                        wf = jax.lax.stop_gradient(jnp.clip(
                            jnp.abs(x0_pos - x0).mean(axis=-1, keepdims=True),
                            a_min=1e-5))
                        nwf = jax.lax.stop_gradient(jnp.clip(
                            jnp.abs(x0_neg - x0).mean(axis=-1, keepdims=True),
                            a_min=1e-5))
                        pos_loss = ((x0_pos - x0) ** 2 / wf).mean(axis=-1)
                        neg_loss = ((x0_neg - x0) ** 2 / nwf).mean(axis=-1)
                    else:
                        pos_loss = ((x0_pos - x0) ** 2).mean(axis=-1)
                        neg_loss = ((x0_neg - x0) ** 2).mean(axis=-1)

                    nft_loss_raw = (
                        r_nft * pos_loss / cfg.nft_beta
                        + (1.0 - r_nft) * neg_loss / cfg.nft_beta
                    )
                    nft_loss_raw = jnp.clip(nft_loss_raw, 0.0, cfg.nft_loss_clip)
                    nft_loss = (nft_loss_raw * cfg.nft_adv_clip_max).mean()
                    nft_kl = ((g_current - g_old) ** 2).mean()
                    nft_loss = nft_loss + cfg.nft_kl_beta * nft_kl

                    if cfg.consistency_coef > 0:
                        t1_idx = jax.random.randint(
                            rng_con1, (B,), 0, cfg.flow_steps)
                        t2_idx = jax.random.randint(
                            rng_con2, (B,), 0, cfg.flow_steps)
                        t1 = time_values[t1_idx][:, None]
                        t2 = time_values[t2_idx][:, None]
                        z_t1 = (1.0 - t1) * x0 + t1 * z_nft
                        z_t2 = (1.0 - t2) * x0 + t2 * z_nft
                        g_t1 = gen_call(
                            network.apply, net_params,
                            traj_batch.obs, traj_batch.prev_cop_act,
                            traj_batch.h_prev, traj_batch.g_embed,
                            z_t1, t1,
                            method=FlowActorCritic.generator_br)
                        g_t2 = jax.lax.stop_gradient(gen_call(
                            network.apply, net_params,
                            traj_batch.obs, traj_batch.prev_cop_act,
                            traj_batch.h_prev, traj_batch.g_embed,
                            z_t2, t2,
                            method=FlowActorCritic.generator_br))
                        con_loss = jnp.minimum(
                            jnp.mean((g_t1 - g_t2) ** 2), cfg.consistency_clip)
                    else:
                        con_loss = jnp.float32(0.0)

                    value_pred = flow_value(
                        network.apply, net_params,
                        traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, traj_batch.g_embed,
                        method=FlowActorCritic.value_br)
                    value_pred_clipped = traj_batch.value + (
                        value_pred - traj_batch.value
                    ).clip(-cfg.value_clip_eps, cfg.value_clip_eps)
                    vl = jnp.square(value_pred - targets)
                    vl_c = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(vl, vl_c).mean()

                    act_std = traj_batch.raw_action.std(axis=0)
                    entropy_proxy = jnp.log(act_std + 1e-8).mean()

                    rng_mb, rng_distill = jax.random.split(rng_mb)
                    z_d = jax.random.normal(rng_distill, (B, d_act))
                    t_one_d = jnp.ones((B, 1))

                    target_br = (gen_call(
                        network.apply, net_params,
                        traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, traj_batch.g_embed,
                        z_d, t_one_d,
                        method=FlowActorCritic.generator_br)
                        * cfg.policy_output_scale)
                    target_br = jax.lax.stop_gradient(target_br)

                    distill_mean = (gen_call(
                        network.apply, net_params,
                        traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, own_embed_b,
                        z_d, t_one_d,
                        method=FlowActorCritic.generator_pop)
                        * cfg.policy_output_scale)
                    distill_loss = jnp.mean((distill_mean - target_br) ** 2)

                    if has_old_slots:
                        rng_mb_re = jax.random.split(rng_mb, 4)
                        rng_slot, rng_re, rng_rt, _ = rng_mb_re
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

                        z_r = jax.random.normal(rng_re, (B, K, d_act))
                        t_r_idx = jax.random.randint(
                            rng_rt, (B, K), 0, cfg.flow_steps)
                        t_r = time_values[t_r_idx][..., None]

                        obs_flat = obs_e.reshape(-1, d_obs)
                        pca_flat = pca_e.reshape(-1, d_pca)
                        hp_flat = hp_e.reshape(-1, d_hp)
                        emb_flat = tau_e.reshape(-1, d_emb)
                        z_flat = z_r.reshape(-1, d_act)
                        t_flat = t_r.reshape(-1, 1)

                        g_cur = network.apply(
                            net_params, obs_flat, pca_flat, hp_flat,
                            emb_flat, z_flat, t_flat,
                            method=FlowActorCritic.generator_pop)
                        g_ref = network.apply(
                            ref_params, obs_flat, pca_flat, hp_flat,
                            emb_flat, z_flat, t_flat,
                            method=FlowActorCritic.generator_pop)
                        g_cur = g_cur * cfg.policy_output_scale
                        g_ref = jax.lax.stop_gradient(
                            g_ref * cfg.policy_output_scale)
                        reg_loss = jnp.mean((g_cur - g_ref) ** 2)
                    else:
                        reg_loss = jnp.asarray(0.0)

                    action_reg = 1e-3 * (g_out_cur ** 2).mean()

                    total = (
                        policy_loss
                        + cfg.nft_reg_coef * nft_loss
                        + cfg.consistency_coef * con_loss
                        + cfg.vf_coef * value_loss
                        + cfg.kl_coef * approx_kl
                        - cfg.entropy_coef * entropy_proxy
                        + action_reg
                        + cfg.distill_beta * distill_loss
                        + cfg.reg_beta * reg_loss
                    )

                    return total, (policy_loss, value_loss, nft_loss,
                                   con_loss, distill_loss, reg_loss,
                                   approx_kl, ratio.mean(), clip_frac)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss_val, aux), grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets,
                    old_p, rng_mb)
                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, old_p), (loss_val, *aux)

            (train_state, old_p, traj_batch,
             advantages, targets, rng) = update_state
            rng, rng_perm, rng_mb = jax.random.split(rng, 3)
            batch_size = cfg.episode_len * cfg.num_envs
            perm = jax.random.permutation(rng_perm, batch_size)
            batch = jax.tree.map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]),
                (traj_batch, advantages, targets))
            shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), batch)
            rng_mbs = jax.random.split(rng_mb, cfg.num_minibatches)
            minibatches = jax.tree.map(
                lambda x: x.reshape([cfg.num_minibatches, -1] + list(x.shape[1:])),
                shuffled) + (rng_mbs,)
            (train_state, old_p), losses = jax.lax.scan(
                _update_minibatch, (train_state, old_p), minibatches)
            return (train_state, old_p, traj_batch,
                    advantages, targets, rng), losses

        update_state = (train_state, old_params, traj_batch,
                        advantages, targets, rng)
        update_state, _ = jax.lax.scan(
            _update_epoch, update_state, None, cfg.update_epochs)
        train_state = update_state[0]
        old_params = update_state[1]
        rng = update_state[-1]

        new_params = train_state.params["net"]
        if cfg.nft_ema > 0:
            old_params = jax.tree.map(
                lambda o, n: cfg.nft_ema * o + (1.0 - cfg.nft_ema) * n,
                old_params, new_params)
        else:
            old_params = new_params

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
        return ((rs_core_new, old_params, new_best_ema, new_prev_ema,
                 new_best_params),
                mean_return)

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

    old_params0 = jax.tree.map(jnp.copy, init_params["net"])
    runner_state = (rs_core0, old_params0, best_ema0, best_prev_ema0,
                    best_params0)
    runner_state, returns = jax.lax.scan(
        _update_step, runner_state, None, cfg.br_steps)

    (rs_core_final, _, best_ema_final, _, best_params_final) = runner_state
    train_state_final = rs_core_final[0]

    return train_state_final, {
        "returns": returns,
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

        raw0, _, _ = gen_sample(
            network_0.apply, params_0, obs0, pa_of_0, h0, e0, rng_0,
            action_dim_0, cfg.policy_output_scale, 0.0, cfg.vel_clip,
            method=FlowActorCritic.generator_pop,
        )
        raw1, _, _ = gen_sample(
            network_1.apply, params_1, obs1, pa_of_1, h1, e1, rng_1,
            action_dim_1, cfg.policy_output_scale, 0.0, cfg.vel_clip,
            method=FlowActorCritic.generator_pop,
        )

        a0 = jnp.tanh(raw0)
        a1 = jnp.tanh(raw1)

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

        raw0, _, _ = gen_sample(
            network_0.apply, params_0, obs0, pa_of_0, h0, e0, rng_0,
            action_dim_0, cfg.policy_output_scale, 0.0, cfg.vel_clip,
            method=FlowActorCritic.generator_pop,
        )
        raw1, _, _ = gen_sample(
            network_1.apply, params_1, obs1, pa_of_1, h1, e1, rng_1,
            action_dim_1, cfg.policy_output_scale, 0.0, cfg.vel_clip,
            method=FlowActorCritic.generator_pop,
        )
        a0 = jnp.tanh(raw0[0])
        a1 = jnp.tanh(raw1[0])

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

def run_meanflow_jpsro(cfg: Config):
    print("=" * 70)
    print("  MeanFlow-NFT-JPSRO (MeanFlow + PPO-ratio + NFT) for HalfCheetah")
    print("=" * 70)

    rng = jax.random.PRNGKey(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.wandb:
        import wandb
        env_tag = "" if cfg.env_name == "halfcheetah_2x3" else f"-{cfg.env_name}"
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            config=asdict(cfg),
            mode=cfg.wandb_mode,
            name=f"meanflow-jpsro{env_tag}-seed{cfg.seed}",
        )

    env_kwargs = {}
    if cfg.env_homogenisation:
        env_kwargs["homogenisation_method"] = cfg.env_homogenisation
    elif cfg.env_name == "humanoid_9|8":
        env_kwargs["homogenisation_method"] = "max"
    env_raw = jaxmarl.make(cfg.env_name, **env_kwargs)
    env = LogWrapper(env_raw, replace_info=True)
    assert len(env.agents) == 2, (
        f"meanflow_old supports 2-agent MABrax envs only (got {len(env.agents)} "
        f"agents for env_name={cfg.env_name!r})"
    )
    agents = env.agents

    obs_dims = {a: env.observation_space(a).shape[0] for a in agents}
    act_dims = {a: env.action_space(a).shape[0] for a in agents}
    print(f"  Agents: {agents}")
    print(f"  Obs dims: {obs_dims}, Act dims: {act_dims}")
    print(f"  MeanFlow: t_embed_dim={cfg.t_embed_dim}, "
          f"output_scale={cfg.policy_output_scale}, flow_steps={cfg.flow_steps}")
    print(f"  PPO-ratio: gen_sigma={cfg.gen_sigma}, sde_sigma={cfg.sde_sigma}, "
          f"ratio_clip={cfg.ratio_clip}, vel_clip={cfg.vel_clip}, "
          f"nft_ema={cfg.nft_ema}")
    print(f"  NFT: beta={cfg.nft_beta}, adv_clip_max={cfg.nft_adv_clip_max}, "
          f"max_gen_delta={cfg.nft_max_gen_delta}, "
          f"reg_coef={cfg.nft_reg_coef}, adaptive={cfg.nft_adaptive}")
    print(f"  Consistency: coef={cfg.consistency_coef}, "
          f"clip={cfg.consistency_clip}")
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
            embed_0, embed_1, cfg.ep_eval_episodes, cfg.episode_len, cfg,
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

            print(f"\n  Player {p}: two-head MeanFlow-BR (PPO+NFT+distill+reg) "
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
            best_params = br_info["best_params"]
            best_ema = float(br_info["best_ema_return"])
            print(f"    BR training: {elapsed:.1f}s, "
                  f"final return: {br_returns[-1]:.2f}, "
                  f"max return: {br_returns.max():.2f}, "
                  f"best EMA: {best_ema:.2f}  <-- committed")
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
    print(f"  MeanFlow-NFT-JPSRO Complete")
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
        ax.set_title("MeanFlow-BR Training Returns (NFT-PN-FPO)")
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
        save_path = os.path.join(cfg.save_dir, "meanflow_jpsro_cheetah.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")
        plt.close(fig)

        if cfg.wandb:
            import wandb
            wandb.log({"summary/training_plot": wandb.Image(save_path)})

    except Exception as e:
        print(f"  Plotting failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="MeanFlow-NFT-JPSRO for Multi-Agent HalfCheetah")
    parser.add_argument("--jpsro-iters", type=int, default=8)
    parser.add_argument("--br-steps", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument(
        "--episode-len", type=int, default=256,
        help="Rollout length used for PPO training AND payoff / BR eval; "
             "keeps `br/p*_return` and `jpsro/best_payoff` on the same scale.")
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--value-clip-eps", type=float, default=0.2)

    parser.add_argument("--flow-steps", type=int, default=10,
                        help="# MeanFlow time steps used for consistency loss")
    parser.add_argument("--t-embed-dim", type=int, default=16)
    parser.add_argument("--policy-output-scale", type=float, default=0.25)
    parser.add_argument("--gen-sigma", type=float, default=0.1,
                        help="Gaussian std used for log-lik at update time")
    parser.add_argument("--sde-sigma", type=float, default=0.1,
                        help="SDE exploration noise at rollout time; MUST be"
                             " > 0 for the PPO ratio to have a gradient")
    parser.add_argument("--vel-clip", type=float, default=0.0,
                        help="generator-output clip (0 disables)")

    parser.add_argument("--ratio-clip", type=float, default=0.2)
    parser.add_argument("--log-ratio-clip", type=float, default=5.0)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--nft-ema", type=float, default=0.0,
                        help="EMA on old_params; 0 == snap pi_old to pi_new"
                             " after every BR update step")
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=1.0)

    parser.add_argument("--nft-beta", type=float, default=0.1)
    parser.add_argument("--nft-adv-clip-max", type=float, default=3.0)
    parser.add_argument("--nft-max-gen-delta", type=float, default=1.0)
    parser.add_argument("--nft-loss-clip", type=float, default=10.0)
    parser.add_argument("--nft-reg-coef", type=float, default=0.5)
    parser.add_argument("--nft-kl-beta", type=float, default=1e-3)
    parser.add_argument("--no-nft-adaptive", action="store_true",
                        help="disable NFT adaptive per-sample weighting")

    parser.add_argument("--consistency-coef", type=float, default=0.1)
    parser.add_argument("--consistency-clip", type=float, default=2.0)

    parser.add_argument("--reg-beta", type=float, default=1.0,
                        help="gen-MSE regulariser weight on old slots")
    parser.add_argument("--reg-n-slots", type=int, default=4,
                        help="# old slots sampled per minibatch for reg")
    parser.add_argument("--distill-beta", type=float, default=1.0,
                        help="gen-MSE distill weight (BR -> pop head at nu^t_p)")
    parser.add_argument("--embed-lr", type=float, default=3e-4,
                        help="Adam lr for the trainable own embedding")
    parser.add_argument("--sigma-cce-prob", type=float, default=0.5,
                        help="prob. of using CCE marginal vs Dirichlet per env")
    parser.add_argument("--sigma-dirichlet-alpha", type=float, default=1.0,
                        help="Dirichlet(alpha) concentration over V_{-p}")
    parser.add_argument("--br-steps-scale", type=int, default=50,
                        help="additional BR steps per JPSRO iter beyond 1")

    parser.add_argument("--best-ema-alpha", type=float, default=0.9,
                        help="EMA factor for smoothing mean_return during BR;"
                             " we commit params at argmax EMA, not final")

    parser.add_argument("--memory-dim", type=int, default=128,
                        help="GRU hidden size in encoder; 0 disables memory")
    parser.add_argument("--no-prev-cop-action", action="store_true",
                        help="don't feed last co-player action into encoder")

    parser.add_argument(
        "--env-name", type=str, default="halfcheetah_2x3",
        choices=["halfcheetah_2x3", "halfcheetah_6x1", "humanoid_9|8",
                 "walker2d_2x3"],
        help="MABrax env (2-agent only); humanoid_9|8 auto-homogenises.")
    parser.add_argument(
        "--env-homogenisation", type=str, default="",
        choices=["", "max", "concat"],
        help="Override MABraxEnv homogenisation_method (default: auto).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results_meanflow_cheetah")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str,
                        default="meanflow-jpsro-cheetah")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    cfg = Config(
        jpsro_iters=args.jpsro_iters,
        br_steps=args.br_steps,
        num_envs=args.num_envs,
        episode_len=args.episode_len,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        value_clip_eps=args.value_clip_eps,
        flow_steps=args.flow_steps,
        t_embed_dim=args.t_embed_dim,
        policy_output_scale=args.policy_output_scale,
        gen_sigma=args.gen_sigma,
        sde_sigma=args.sde_sigma,
        vel_clip=args.vel_clip,
        ratio_clip=args.ratio_clip,
        log_ratio_clip=args.log_ratio_clip,
        kl_coef=args.kl_coef,
        nft_ema=args.nft_ema,
        entropy_coef=args.entropy_coef,
        nft_beta=args.nft_beta,
        nft_adv_clip_max=args.nft_adv_clip_max,
        nft_max_gen_delta=args.nft_max_gen_delta,
        nft_loss_clip=args.nft_loss_clip,
        nft_reg_coef=args.nft_reg_coef,
        nft_kl_beta=args.nft_kl_beta,
        nft_adaptive=(not args.no_nft_adaptive),
        consistency_coef=args.consistency_coef,
        consistency_clip=args.consistency_clip,
        vf_coef=args.vf_coef,
        reg_beta=args.reg_beta,
        reg_n_slots=args.reg_n_slots,
        distill_beta=args.distill_beta,
        embed_lr=args.embed_lr,
        sigma_cce_prob=args.sigma_cce_prob,
        sigma_dirichlet_alpha=args.sigma_dirichlet_alpha,
        br_steps_scale=args.br_steps_scale,
        best_ema_alpha=args.best_ema_alpha,
        memory_dim=args.memory_dim,
        use_prev_cop_action=(not args.no_prev_cop_action),
        env_name=args.env_name,
        env_homogenisation=args.env_homogenisation,
        seed=args.seed,
        save_dir=args.save_dir,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )
    run_meanflow_jpsro(cfg)
    print("\nDone!")

if __name__ == "__main__":
    main()