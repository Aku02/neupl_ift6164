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
from dataclasses import dataclass, field, asdict, replace
from typing import NamedTuple, Tuple

import jax
jax.tree_map = jax.tree.map

import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import distrax
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
    hidden_dim: int = 64

    br_steps: int = 300
    num_envs: int = 64
    episode_len: int = 256
    update_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    distill_beta: float = 1.0
    reg_beta: float = 1.0
    reg_n_slots: int = 4

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
    save_dir: str = "results_neupl_cheetah"

    wandb: bool = False
    wandb_project: str = "neupl-jpsro-cheetah"
    wandb_entity: str = ""
    wandb_mode: str = "online"

class ObsEncoder(nn.Module):

    hidden_dim: int = 64
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
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        if self.memory_dim > 0:

            new_carry, _ = nn.GRUCell(features=self.memory_dim,
                                      kernel_init=orthogonal(np.sqrt(2)),
                                      bias_init=constant(0.0))(h_prev, x)
            return new_carry
        return x

class GaussianHead(nn.Module):

    action_dim: int
    hidden_dim: int = 64
    name_prefix: str = ""

    @nn.compact
    def __call__(self, h, cond):
        x = jnp.concatenate([h, cond], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        mu = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                      bias_init=constant(0.0))(x)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        return mu, log_std

class ValueHead(nn.Module):

    hidden_dim: int = 64

    @nn.compact
    def __call__(self, h, cond):
        x = jnp.concatenate([h, cond], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        v = nn.Dense(1, kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0))(x)
        return jnp.squeeze(v, axis=-1)

class ActorCritic(nn.Module):

    action_dim: int
    cop_action_dim: int
    hidden_dim: int = 64
    memory_dim: int = 128
    use_prev_cop_action: bool = True

    def setup(self):
        self.encoder = ObsEncoder(
            hidden_dim=self.hidden_dim,
            memory_dim=self.memory_dim,
            use_prev_cop_action=self.use_prev_cop_action,
        )
        self.br_head = GaussianHead(
            action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        self.pop_head = GaussianHead(
            action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        self.br_value = ValueHead(hidden_dim=self.hidden_dim)
        self.pop_value = ValueHead(hidden_dim=self.hidden_dim)

    def __call__(self, obs, prev_cop_act, h_prev, cond_br, cond_pop):

        h = self.encoder(obs, prev_cop_act, h_prev)
        mu_br, ls_br = self.br_head(h, cond_br)
        mu_p, ls_p = self.pop_head(h, cond_pop)
        v_br = self.br_value(h, cond_br)
        v_p = self.pop_value(h, cond_pop)
        return mu_br, ls_br, mu_p, ls_p, v_br, v_p

    def encode(self, obs, prev_cop_act, h_prev):

        return self.encoder(obs, prev_cop_act, h_prev)

    def policy_br(self, obs, prev_cop_act, h_prev, cond_br):
        h = self.encoder(obs, prev_cop_act, h_prev)
        mu, ls = self.br_head(h, cond_br)
        return distrax.MultivariateNormalDiag(mu, jnp.exp(ls)), (mu, ls)

    def value_br(self, obs, prev_cop_act, h_prev, cond_br):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.br_value(h, cond_br)

    def act_br(self, obs, prev_cop_act, h_prev, cond_br):
        pi, _ = self.policy_br(obs, prev_cop_act, h_prev, cond_br)
        return pi, self.value_br(obs, prev_cop_act, h_prev, cond_br)

    def policy_pop(self, obs, prev_cop_act, h_prev, cond_pop):
        h = self.encoder(obs, prev_cop_act, h_prev)
        mu, ls = self.pop_head(h, cond_pop)
        return distrax.MultivariateNormalDiag(mu, jnp.exp(ls)), (mu, ls)

    def value_pop(self, obs, prev_cop_act, h_prev, cond_pop):
        h = self.encoder(obs, prev_cop_act, h_prev)
        return self.pop_value(h, cond_pop)

    def act_pop(self, obs, prev_cop_act, h_prev, cond_pop):
        pi, _ = self.policy_pop(obs, prev_cop_act, h_prev, cond_pop)
        return pi, self.value_pop(obs, prev_cop_act, h_prev, cond_pop)

    def mu_logstd_pop(self, obs, prev_cop_act, h_prev, cond_pop):

        h = self.encoder(obs, prev_cop_act, h_prev)
        mu, ls = self.pop_head(h, cond_pop)
        return mu, ls

    def mu_logstd_br(self, obs, prev_cop_act, h_prev, cond_br):
        h = self.encoder(obs, prev_cop_act, h_prev)
        mu, ls = self.br_head(h, cond_br)
        return mu, ls

def gaussian_kl_diag(mu1, log_std1, mu2, log_std2):

    var1 = jnp.exp(2.0 * log_std1)
    var2 = jnp.exp(2.0 * log_std2)
    term = log_std2 - log_std1 + (var1 + (mu1 - mu2) ** 2) / (2.0 * var2) - 0.5
    return term.sum(axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    g_embed: jnp.ndarray
    prev_cop_act: jnp.ndarray
    h_prev: jnp.ndarray

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
    network: ActorCritic,
    init_params,
    ref_params,
    cop_params,
    cop_network: ActorCritic,
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

    mem_dim = cfg.memory_dim if cfg.memory_dim > 0 else cfg.hidden_dim

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=init_params, tx=tx)

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
         h_prev_p, h_prev_c, prev_act_cop_of_p, prev_act_cop_of_c, rng) = runner_state
        rng, rng_act, rng_step, rng_sig, rng_idx = jax.random.split(rng, 5)

        net_params = train_state.params["net"]

        p_obs = last_obs[player_agent]
        c_obs = last_obs[cop_agent]

        sigma_b = _sample_sigma_batch(rng_sig)
        g_batch = sigma_b @ cop_embeds

        pi_p, val_p = network.apply(
            net_params, p_obs, prev_act_cop_of_p, h_prev_p, g_batch,
            method=ActorCritic.act_br)
        p_action = pi_p.sample(seed=rng_act)
        p_log_prob = pi_p.log_prob(p_action)

        u_gumbel = jax.random.uniform(rng_idx, (cfg.num_envs, num_cop_strats),
                                      minval=1e-20, maxval=1.0)
        gumbel = -jnp.log(-jnp.log(u_gumbel))
        logits_b = jnp.log(jnp.clip(sigma_b, 1e-20, 1.0))
        strat_idx = jnp.argmax(logits_b + gumbel, axis=-1)
        c_embed = cop_embeds[strat_idx]
        pi_c, _ = cop_network.apply(
            cop_params, c_obs, prev_act_cop_of_c, h_prev_c, c_embed,
            method=ActorCritic.act_pop)
        c_action = pi_c.sample(seed=jax.random.fold_in(rng_act, 1))

        if player_idx == 0:
            actions = {agents[0]: p_action, agents[1]: c_action}
        else:
            actions = {agents[0]: c_action, agents[1]: p_action}

        rng_steps = jax.random.split(rng_step, cfg.num_envs)
        next_obs, next_state, rewards, dones, info = jax.vmap(env.step)(
            rng_steps, env_state, actions)

        h_new_p = network.apply(
            net_params, p_obs, prev_act_cop_of_p, h_prev_p,
            method=ActorCritic.encode)
        h_new_c = cop_network.apply(
            cop_params, c_obs, prev_act_cop_of_c, h_prev_c,
            method=ActorCritic.encode)

        d_p = dones[player_agent][:, None]
        d_c = dones[cop_agent][:, None]
        zero_h = jnp.zeros_like(h_new_p)
        h_prev_p_next = jnp.where(d_p, zero_h, h_new_p)
        h_prev_c_next = jnp.where(d_c, jnp.zeros_like(h_new_c), h_new_c)

        zero_pca_p = jnp.zeros_like(prev_act_cop_of_p)
        zero_pca_c = jnp.zeros_like(prev_act_cop_of_c)
        prev_act_cop_of_p_next = jnp.where(d_p, zero_pca_p, c_action)
        prev_act_cop_of_c_next = jnp.where(d_c, zero_pca_c, p_action)

        transition = Transition(
            done=dones[player_agent],
            action=p_action,
            value=val_p,
            reward=rewards[player_agent],
            log_prob=p_log_prob,
            obs=p_obs,
            g_embed=g_batch,
            prev_cop_act=prev_act_cop_of_p,
            h_prev=h_prev_p,
        )
        next_runner = (train_state, next_state, next_obs,
                       h_prev_p_next, h_prev_c_next,
                       prev_act_cop_of_p_next, prev_act_cop_of_c_next, rng)
        return next_runner, transition

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

    def _update_step(carry, unused):
        runner_state, best_ema, best_prev_ema, best_params = carry
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, cfg.episode_len)
        (train_state, env_state, last_obs,
         h_prev_p, h_prev_c, prev_act_cop_of_p, prev_act_cop_of_c, rng) = runner_state

        rng, rng_bs = jax.random.split(rng)
        sigma_bs = _sample_sigma_batch(rng_bs)
        g_bs = sigma_bs @ cop_embeds
        net_params = train_state.params["net"]
        last_val = network.apply(
            net_params, last_obs[player_agent], prev_act_cop_of_p, h_prev_p, g_bs,
            method=ActorCritic.value_br)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state, unused):
            def _update_minibatch(train_state, batch_info):
                traj_batch, advantages, targets, rng_mb = batch_info

                def _loss_fn(params, traj_batch, gae, targets, rng_mb):
                    net_params = params["net"]
                    own_embed = params["embed"]
                    B = traj_batch.obs.shape[0]
                    d_emb = own_embed.shape[-1]
                    d_obs = traj_batch.obs.shape[-1]
                    own_embed_b = jnp.broadcast_to(own_embed, (B, d_emb))

                    pi, value = network.apply(
                        net_params, traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, traj_batch.g_embed,
                        method=ActorCritic.act_br)
                    log_prob = pi.log_prob(traj_batch.action)
                    entropy = pi.entropy().mean()

                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-cfg.clip_eps, cfg.clip_eps)
                    value_loss = 0.5 * jnp.maximum(
                        jnp.square(value - targets),
                        jnp.square(value_pred_clipped - targets)).mean()

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_a1 = ratio * gae_norm
                    loss_a2 = jnp.clip(
                        ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * gae_norm
                    loss_actor = -jnp.minimum(loss_a1, loss_a2).mean()
                    approx_kl = jnp.mean((ratio - 1.0) - (log_prob - traj_batch.log_prob))

                    mu_pop, ls_pop = network.apply(
                        net_params, traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, own_embed_b,
                        method=ActorCritic.mu_logstd_pop)
                    mu_br, ls_br = network.apply(
                        net_params, traj_batch.obs, traj_batch.prev_cop_act,
                        traj_batch.h_prev, traj_batch.g_embed,
                        method=ActorCritic.mu_logstd_br)
                    mu_br = jax.lax.stop_gradient(mu_br)
                    ls_br = jax.lax.stop_gradient(ls_br)
                    distill_loss = gaussian_kl_diag(
                        mu_pop, ls_pop, mu_br, ls_br).mean()

                    if has_old_slots:
                        rng_slot, = jax.random.split(rng_mb, 1)
                        K = reg_n_slots_actual
                        T_old = old_embeds.shape[0]
                        d_pca = traj_batch.prev_cop_act.shape[-1]
                        d_h = traj_batch.h_prev.shape[-1]
                        idx = jax.random.randint(rng_slot, (K,), 0, T_old)
                        tau_embeds = old_embeds[idx]

                        obs_e = jnp.broadcast_to(
                            traj_batch.obs[:, None, :], (B, K, d_obs))
                        tau_e = jnp.broadcast_to(
                            tau_embeds[None, :, :], (B, K, d_emb))
                        pca_e = jnp.broadcast_to(
                            traj_batch.prev_cop_act[:, None, :], (B, K, d_pca))
                        h_e = jnp.broadcast_to(
                            traj_batch.h_prev[:, None, :], (B, K, d_h))
                        obs_flat = obs_e.reshape(-1, d_obs)
                        tau_flat = tau_e.reshape(-1, d_emb)
                        pca_flat = pca_e.reshape(-1, d_pca)
                        h_flat = h_e.reshape(-1, d_h)

                        mu_cur, ls_cur = network.apply(
                            net_params, obs_flat, pca_flat, h_flat, tau_flat,
                            method=ActorCritic.mu_logstd_pop)
                        mu_ref, ls_ref = network.apply(
                            ref_params, obs_flat, pca_flat, h_flat, tau_flat,
                            method=ActorCritic.mu_logstd_pop)
                        mu_ref = jax.lax.stop_gradient(mu_ref)
                        ls_ref = jax.lax.stop_gradient(ls_ref)
                        reg_loss = gaussian_kl_diag(
                            mu_cur, ls_cur, mu_ref, ls_ref).mean()
                    else:
                        reg_loss = jnp.asarray(0.0)

                    total = (loss_actor
                             + cfg.vf_coef * value_loss
                             - cfg.ent_coef * entropy
                             + cfg.distill_beta * distill_loss
                             + cfg.reg_beta * reg_loss)
                    return total, (value_loss, loss_actor, entropy,
                                   distill_loss, reg_loss, approx_kl)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets, rng_mb)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng = update_state
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
            train_state, losses = jax.lax.scan(
                _update_minibatch, train_state, minibatches)
            return (train_state, traj_batch, advantages, targets, rng), losses

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, _ = jax.lax.scan(
            _update_epoch, update_state, None, cfg.update_epochs)
        train_state = update_state[0]
        rng = update_state[-1]

        mean_return = traj_batch.reward.sum(axis=0).mean()

        alpha = cfg.best_ema_alpha
        new_prev_ema = alpha * best_prev_ema + (1.0 - alpha) * mean_return
        improved = new_prev_ema > best_ema
        new_best_ema = jnp.where(improved, new_prev_ema, best_ema)
        new_best_params = jax.tree.map(
            lambda old, new: jnp.where(improved, new, old),
            best_params, train_state.params)

        new_runner = (train_state, env_state, last_obs,
                      h_prev_p, h_prev_c,
                      prev_act_cop_of_p, prev_act_cop_of_c, rng)
        new_carry = (new_runner, new_best_ema, new_prev_ema, new_best_params)
        return new_carry, (mean_return, new_prev_ema)

    rng, rng_reset = jax.random.split(rng)
    reset_rngs = jax.random.split(rng_reset, cfg.num_envs)
    obs, env_state = jax.vmap(env.reset)(reset_rngs)

    rng, rng_train = jax.random.split(rng)

    h0_p = jnp.zeros((cfg.num_envs, mem_dim))
    h0_c = jnp.zeros((cfg.num_envs, mem_dim))
    pa0_p = jnp.zeros((cfg.num_envs, action_dim_c))
    pa0_c = jnp.zeros((cfg.num_envs, action_dim_p))
    runner_state = (train_state, env_state, obs,
                    h0_p, h0_c, pa0_p, pa0_c, rng_train)

    init_best_params = jax.tree.map(jnp.copy, train_state.params)
    init_best_ema = jnp.asarray(-1e9, dtype=jnp.float32)
    init_prev_ema = jnp.asarray(0.0, dtype=jnp.float32)
    init_carry = (runner_state, init_best_ema, init_prev_ema, init_best_params)
    final_carry, (returns, ema_returns) = jax.lax.scan(
        _update_step, init_carry, None, cfg.br_steps)

    final_runner, best_ema_return, _final_prev_ema, best_params = final_carry

    return final_runner[0], {
        "returns": returns,
        "ema_returns": ema_returns,
        "best_params": best_params,
        "best_ema_return": best_ema_return,
    }

def eval_joint_strategy(
    rng, env, network_0, params_0, network_1, params_1,
    embed_0, embed_1, num_envs, num_steps, mem_dim, action_dim_0, action_dim_1,
):

    agents = env.agents
    md = mem_dim

    def _step(carry, unused):
        (env_state, obs,
         h0, h1, pa_of_0, pa_of_1, rng) = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        e0 = jnp.broadcast_to(embed_0, (num_envs, embed_0.shape[-1]))
        e1 = jnp.broadcast_to(embed_1, (num_envs, embed_1.shape[-1]))

        pi_0, _ = network_0.apply(
            params_0, obs[agents[0]], pa_of_0, h0, e0,
            method=ActorCritic.act_pop)
        pi_1, _ = network_1.apply(
            params_1, obs[agents[1]], pa_of_1, h1, e1,
            method=ActorCritic.act_pop)

        a0 = pi_0.sample(seed=rng_act)
        a1 = pi_1.sample(seed=jax.random.fold_in(rng_act, 1))

        actions = {agents[0]: a0, agents[1]: a1}
        rng_steps = jax.random.split(rng_step, num_envs)
        next_obs, next_state, rewards, dones, info = jax.vmap(env.step)(
            rng_steps, env_state, actions)

        h0_new = network_0.apply(
            params_0, obs[agents[0]], pa_of_0, h0, method=ActorCritic.encode)
        h1_new = network_1.apply(
            params_1, obs[agents[1]], pa_of_1, h1, method=ActorCritic.encode)

        d0 = dones[agents[0]][:, None]
        d1 = dones[agents[1]][:, None]
        h0_next = jnp.where(d0, jnp.zeros_like(h0_new), h0_new)
        h1_next = jnp.where(d1, jnp.zeros_like(h1_new), h1_new)
        pa0_next = jnp.where(d0, jnp.zeros_like(pa_of_0), a1)
        pa1_next = jnp.where(d1, jnp.zeros_like(pa_of_1), a0)

        return (next_state, next_obs, h0_next, h1_next, pa0_next, pa1_next, rng), rewards[agents[0]]

    rng, rng_reset = jax.random.split(rng)
    reset_rngs = jax.random.split(rng_reset, num_envs)
    obs, env_state = jax.vmap(env.reset)(reset_rngs)

    h0_init = jnp.zeros((num_envs, md))
    h1_init = jnp.zeros((num_envs, md))
    pa0_init = jnp.zeros((num_envs, action_dim_1))
    pa1_init = jnp.zeros((num_envs, action_dim_0))

    _, rewards = jax.lax.scan(
        _step,
        (env_state, obs, h0_init, h1_init, pa0_init, pa1_init, rng),
        None, num_steps)
    return rewards.sum(axis=0).mean()

def rollout_trajectory(
    rng, env_raw, network_0, params_0, network_1, params_1,
    embed_0, embed_1, num_steps, mem_dim, action_dim_0, action_dim_1,
):

    agents = env_raw.agents
    md = mem_dim

    def _step(carry, unused):
        (env_state, obs,
         h0, h1, pa_of_0, pa_of_1, rng) = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        e0 = embed_0[None]
        e1 = embed_1[None]

        pi_0, _ = network_0.apply(
            params_0, obs[agents[0]][None], pa_of_0, h0, e0,
            method=ActorCritic.act_pop)
        pi_1, _ = network_1.apply(
            params_1, obs[agents[1]][None], pa_of_1, h1, e1,
            method=ActorCritic.act_pop)

        a0_b = pi_0.sample(seed=rng_act)
        a1_b = pi_1.sample(seed=jax.random.fold_in(rng_act, 1))
        a0 = a0_b[0]
        a1 = a1_b[0]

        actions = {agents[0]: a0, agents[1]: a1}
        next_obs, next_state, rewards, dones, info = env_raw.step(
            rng_step, env_state, actions)

        h0_new = network_0.apply(
            params_0, obs[agents[0]][None], pa_of_0, h0, method=ActorCritic.encode)
        h1_new = network_1.apply(
            params_1, obs[agents[1]][None], pa_of_1, h1, method=ActorCritic.encode)

        d0 = jnp.asarray(dones[agents[0]])[None, None]
        d1 = jnp.asarray(dones[agents[1]])[None, None]
        h0_next = jnp.where(d0, jnp.zeros_like(h0_new), h0_new)
        h1_next = jnp.where(d1, jnp.zeros_like(h1_new), h1_new)
        pa0_next = jnp.where(d0, jnp.zeros_like(pa_of_0), a1_b)
        pa1_next = jnp.where(d1, jnp.zeros_like(pa_of_1), a0_b)

        return (next_state, next_obs, h0_next, h1_next, pa0_next, pa1_next, rng), next_state.pipeline_state

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env_raw.reset(rng_reset)
    init_pipeline = env_state.pipeline_state

    h0_init = jnp.zeros((1, md))
    h1_init = jnp.zeros((1, md))
    pa0_init = jnp.zeros((1, action_dim_1))
    pa1_init = jnp.zeros((1, action_dim_0))

    _, traj_pipeline_states = jax.lax.scan(
        _step,
        (env_state, obs, h0_init, h1_init, pa0_init, pa1_init, rng),
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

def run_neupl_jpsro(cfg: Config):
    print("=" * 70)
    print("  NeuPL-JPSRO for Multi-Agent HalfCheetah (2 agents)")
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
            name=f"neupl-jpsro{env_tag}-seed{cfg.seed}",
        )

    env_kwargs = {}
    if cfg.env_homogenisation:
        env_kwargs["homogenisation_method"] = cfg.env_homogenisation
    elif cfg.env_name == "humanoid_9|8":
        env_kwargs["homogenisation_method"] = "max"
    env_raw = jaxmarl.make(cfg.env_name, **env_kwargs)
    env = LogWrapper(env_raw, replace_info=True)
    assert len(env.agents) == 2, (
        f"neupl_old supports 2-agent MABrax envs only (got {len(env.agents)} "
        f"agents for env_name={cfg.env_name!r})"
    )
    agents = env.agents

    obs_dims = {a: env.observation_space(a).shape[0] for a in agents}
    act_dims = {a: env.action_space(a).shape[0] for a in agents}
    print(f"  Agents: {agents}")
    print(f"  Obs dims: {obs_dims}, Act dims: {act_dims}")

    networks = [
        ActorCritic(
            action_dim=act_dims[agents[p]],
            cop_action_dim=act_dims[agents[1 - p]],
            hidden_dim=cfg.hidden_dim,
            memory_dim=cfg.memory_dim,
            use_prev_cop_action=cfg.use_prev_cop_action,
        )
        for p in range(2)
    ]

    mem_dim_eff = cfg.memory_dim if cfg.memory_dim > 0 else cfg.hidden_dim

    rng, *init_rngs = jax.random.split(rng, 3)
    dummy_obs = [jnp.zeros((obs_dims[agents[p]],)) for p in range(2)]
    dummy_pca = [jnp.zeros((act_dims[agents[1 - p]],)) for p in range(2)]
    dummy_h = jnp.zeros((mem_dim_eff,))
    dummy_embed = jnp.zeros((cfg.embed_dim,))

    net_params = [
        networks[p].init(init_rngs[p], dummy_obs[p], dummy_pca[p], dummy_h,
                         dummy_embed, dummy_embed)
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
            embed_0, embed_1, cfg.ep_eval_episodes, cfg.episode_len,
            mem_dim_eff, act_dims[agents[0]], act_dims[agents[1]])

    @jax.jit
    def _rollout(rng, params_0, params_1, embed_0, embed_1):
        return rollout_trajectory(
            rng, env_raw, networks[0], params_0, networks[1], params_1,
            embed_0, embed_1, cfg.viz_rollout_steps,
            mem_dim_eff, act_dims[agents[0]], act_dims[agents[1]])

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

        print(f"\n{'─' * 60}")
        print(f"  JPSRO Iteration {t}/{cfg.jpsro_iters}  "
              f"(br_steps={br_steps_iter})")
        print(f"{'─' * 60}")

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

            print(f"\n  Player {p}: two-head PPO-BR (PPO+KL-distill+KL-reg) "
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
                action_dim_p=act_dims[agents[p]],
                action_dim_c=act_dims[agents[cop]],
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
            ema_returns = np.array(br_info["ema_returns"])
            best_ema = float(br_info["best_ema_return"])
            print(f"    BR training: {elapsed:.1f}s, "
                  f"final return: {br_returns[-1]:.2f}, "
                  f"max return: {br_returns.max():.2f}, "
                  f"best EMA: {best_ema:.2f}")
            returns_history.append(br_returns)

            if cfg.wandb:
                import wandb
                for step_i, ret_val in enumerate(br_returns):
                    wandb.log({
                        f"br/p{p}_return": float(ret_val),
                        f"br/p{p}_ema_return": float(ema_returns[step_i]),
                        f"br/p{p}_iter": t,
                        "br/global_step": global_br_step,
                    }, step=global_br_step)
                    global_br_step += 1
                wandb.log({
                    f"br/p{p}_best_ema_return": best_ema,
                    f"br/p{p}_iter_end": t,
                }, step=global_br_step)

            net_params[p] = br_info["best_params"]["net"]
            embeds[p][-1] = br_info["best_params"]["embed"]
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

        print(f"  Payoff matrix:\n{np.round(payoff_matrix, 1)}")
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
    print(f"  NeuPL-JPSRO Complete")
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
        ax.set_title("BR Training Returns")
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
            ax.set_xlabel("P1 (front leg) strategy index j (column)")
            ax.set_ylabel("P0 (rear leg) strategy index i (row)")
            ax.set_title(
                "G[i,j] = payoff for pure joint (strat i, strat j)\n"
                "not a mixture unless averaged with σ"
            )
            plt.colorbar(im, ax=ax)
            for i in range(final_pm.shape[0]):
                for j in range(final_pm.shape[1]):
                    color = 'white' if final_pm[i, j] < final_pm.mean() else 'black'
                    ax.text(j, i, f"{final_pm[i, j]:.0f}",
                            ha='center', va='center', fontsize=7, color=color)

        plt.tight_layout()
        save_path = os.path.join(cfg.save_dir, "neupl_jpsro_cheetah.png")
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
        description="NeuPL-JPSRO for Multi-Agent HalfCheetah")
    parser.add_argument("--jpsro-iters", type=int, default=6)
    parser.add_argument("--br-steps", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument(
        "--episode-len", type=int, default=256,
        help="Rollout length used for PPO training AND payoff / BR eval; "
             "keeps `br/p*_return` and `jpsro/best_payoff` on the same scale.")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=1e-3)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)

    parser.add_argument("--distill-beta", type=float, default=1.0,
                        help="KL(pop||BR) distill weight")
    parser.add_argument("--reg-beta", type=float, default=1.0,
                        help="KL(theta||theta_hat) reg weight on old slots")
    parser.add_argument("--reg-n-slots", type=int, default=4,
                        help="# old slots sampled per minibatch for reg")
    parser.add_argument("--embed-lr", type=float, default=3e-4)
    parser.add_argument("--sigma-cce-prob", type=float, default=0.5)
    parser.add_argument("--sigma-dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--br-steps-scale", type=int, default=50)

    parser.add_argument("--best-ema-alpha", type=float, default=0.9,
                        help="EMA factor for best-params commit")
    parser.add_argument("--memory-dim", type=int, default=128,
                        help="GRU hidden size in shared encoder (0 disables)")
    parser.add_argument("--no-prev-cop-action", action="store_true",
                        help="Do not feed prev co-player action into the encoder")

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
    parser.add_argument("--save-dir", type=str, default="results_neupl_cheetah")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="neupl-jpsro-cheetah")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    cfg = Config(
        jpsro_iters=args.jpsro_iters,
        br_steps=args.br_steps,
        num_envs=args.num_envs,
        episode_len=args.episode_len,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        ent_coef=args.ent_coef,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        distill_beta=args.distill_beta,
        reg_beta=args.reg_beta,
        reg_n_slots=args.reg_n_slots,
        embed_lr=args.embed_lr,
        sigma_cce_prob=args.sigma_cce_prob,
        sigma_dirichlet_alpha=args.sigma_dirichlet_alpha,
        br_steps_scale=args.br_steps_scale,
        best_ema_alpha=args.best_ema_alpha,
        memory_dim=args.memory_dim,
        use_prev_cop_action=not args.no_prev_cop_action,
        env_name=args.env_name,
        env_homogenisation=args.env_homogenisation,
        seed=args.seed,
        save_dir=args.save_dir,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )
    run_neupl_jpsro(cfg)
    print("\nDone!")

if __name__ == "__main__":
    main()