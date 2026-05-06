

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

@dataclass
class GameSpec:

    name: str
    payoff_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    action_min: float
    action_max: float
    grid_n: int = 151

    def action_range(self) -> float:
        return self.action_max - self.action_min

    def sample_grid(self) -> np.ndarray:
        return np.linspace(self.action_min, self.action_max, self.grid_n)

def _payoff_checkerboard(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) * torch.sin(y)

def _payoff_twin_peaks(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        torch.exp(-((x - 1.0) ** 2) - (y - 1.0) ** 2)
        + torch.exp(-((x + 1.0) ** 2) - (y + 1.0) ** 2)
        - 0.5
    )

def _payoff_poly_sine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x * y) ** 2 + torch.sin(x + y)

def _payoff_stable_limit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x**2 + y**2 - 1.0) ** 2 + x * y

def _payoff_matching_pennies(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    return torch.cos(x - y)

def _payoff_double_well(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    return -((x**2 - 1.0) ** 2) + x * y

def _payoff_rastrigin_saddle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    A = 3.0
    return A * (1.0 - torch.cos(2.0 * math.pi * x)) - A * (
        1.0 - torch.cos(2.0 * math.pi * y)
    ) + 2.0 * x * y

GAMES: dict[str, GameSpec] = {
    "checkerboard": GameSpec(
        name="checkerboard",
        payoff_fn=_payoff_checkerboard,
        action_min=-math.pi,
        action_max=math.pi,
        grid_n=151,
    ),
    "twin_peaks": GameSpec(
        name="twin_peaks",
        payoff_fn=_payoff_twin_peaks,
        action_min=-3.0,
        action_max=3.0,
        grid_n=121,
    ),
    "poly_sine": GameSpec(
        name="poly_sine",
        payoff_fn=_payoff_poly_sine,
        action_min=-math.pi,
        action_max=math.pi,
        grid_n=151,
    ),
    "stable_limit": GameSpec(
        name="stable_limit",
        payoff_fn=_payoff_stable_limit,
        action_min=-2.0,
        action_max=2.0,
        grid_n=121,
    ),

    "matching_pennies": GameSpec(
        name="matching_pennies",
        payoff_fn=_payoff_matching_pennies,
        action_min=-math.pi,
        action_max=math.pi,
        grid_n=151,
    ),
    "double_well": GameSpec(
        name="double_well",
        payoff_fn=_payoff_double_well,
        action_min=-2.0,
        action_max=2.0,
        grid_n=121,
    ),
    "rastrigin_saddle": GameSpec(
        name="rastrigin_saddle",
        payoff_fn=_payoff_rastrigin_saddle,
        action_min=-2.0,
        action_max=2.0,
        grid_n=151,
    ),
}

def solve_nash_zero_sum(payoff: np.ndarray) -> np.ndarray:

    n1, _ = payoff.shape
    if n1 == 1:
        return np.array([1.0])

    c = np.zeros(n1 + 1)
    c[-1] = -1.0

    A_ub = np.zeros((payoff.shape[1], n1 + 1))
    A_ub[:, :n1] = -payoff.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(payoff.shape[1])

    A_eq = np.zeros((1, n1 + 1))
    A_eq[0, :n1] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * n1 + [(None, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if result.success:
        p = np.maximum(result.x[:n1], 0.0)
        s = p.sum()
        if s > 0:
            return p / s
    return np.ones(n1) / n1

def psro_nash_meta_graph_solver(payoff: np.ndarray) -> np.ndarray:

    n = payoff.shape[0]
    sigma = np.zeros((n, n))
    for i in range(1, n):
        sigma[i, :i] = solve_nash_zero_sum(payoff[:i, :i])
    return sigma

def discretized_nash(game: GameSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    grid = game.sample_grid()
    xs = torch.from_numpy(grid).float().view(-1, 1)
    ys = torch.from_numpy(grid).float().view(1, -1)
    with torch.no_grad():
        U = game.payoff_fn(xs, ys).numpy()

    p_row = solve_nash_zero_sum(U)
    p_col = solve_nash_zero_sum(-U.T)
    value = float(p_row @ U @ p_col)
    return grid, p_row, p_col, value

class GaussianPopulation(nn.Module):

    def __init__(
        self,
        num_slots: int,
        action_min: float,
        action_max: float,
        init_log_std: float = 0.0,
        max_std_fraction: float = 0.25,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.action_min = float(action_min)
        self.action_max = float(action_max)
        span = 0.5 * (self.action_max - self.action_min)
        centre = 0.5 * (self.action_max + self.action_min)
        self.max_std = max_std_fraction * (self.action_max - self.action_min)
        self.slot_mu = nn.ParameterList(
            [
                nn.Parameter(centre + 0.3 * span * torch.randn(1))
                for _ in range(num_slots)
            ]
        )
        self.slot_log_std = nn.ParameterList(
            [nn.Parameter(torch.tensor([init_log_std])) for _ in range(num_slots)]
        )
        self.slot_values = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_slots)]
        )

    def dist(self, idx: int) -> torch.distributions.Normal:
        mu = self.slot_mu[idx]
        std = torch.exp(self.slot_log_std[idx]).clamp(min=1e-3, max=self.max_std)
        return torch.distributions.Normal(mu, std)

    def sample(self, idx: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:

        d = self.dist(idx)
        z = d.rsample((n,)).squeeze(-1)
        log_p = d.log_prob(z).squeeze(-1)
        return z, log_p

    def sample_no_grad(self, idx: int, n: int) -> torch.Tensor:
        with torch.no_grad():
            return self.dist(idx).sample((n,)).squeeze(-1)

    def value(self, idx: int) -> torch.Tensor:
        return self.slot_values[idx].squeeze()

    def mean_np(self, idx: int) -> float:
        with torch.no_grad():
            return float(self.slot_mu[idx].item())

    def std_np(self, idx: int) -> float:
        with torch.no_grad():
            return float(torch.exp(self.slot_log_std[idx]).item())

def _embed_timestep(t: torch.Tensor, embed_dim: int) -> torch.Tensor:

    device = t.device
    freqs = 2 ** torch.arange(embed_dim // 2, device=device, dtype=t.dtype)
    scaled = t * freqs
    return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)

class FlowSlot(nn.Module):

    def __init__(
        self,
        action_dim: int,
        t_embed_dim: int,
        hidden: int = 64,
        init_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.t_embed_dim = t_embed_dim
        self.net = nn.Sequential(
            nn.Linear(action_dim + t_embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            if init_bias is None:
                self.net[-1].bias.zero_()
            else:
                self.net[-1].bias.copy_(init_bias.view(-1))

    def forward(self, x_t: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, t_embed], dim=-1))

class FlowPopulation(nn.Module):

    def __init__(
        self,
        num_slots: int,
        action_min: float,
        action_max: float,
        flow_steps: int = 5,
        t_embed_dim: int = 8,
        hidden: int = 64,
        action_dim: int = 1,
        sample_noise: float = 1.5,
        explore_noise: float = 0.3,
    ):
        super().__init__()
        assert t_embed_dim % 2 == 0, "t_embed_dim must be even"
        self.num_slots = num_slots
        self.action_dim = action_dim
        self.flow_steps = flow_steps
        self.t_embed_dim = t_embed_dim
        self.action_min = float(action_min)
        self.action_max = float(action_max)
        self.action_centre = 0.5 * (self.action_max + self.action_min)
        self.action_halfrange = 0.5 * (self.action_max - self.action_min)
        self.sample_noise = sample_noise
        self.explore_noise = explore_noise

        if num_slots > 1:
            slot_centres = torch.linspace(
                self.action_min + 0.1 * (self.action_max - self.action_min),
                self.action_max - 0.1 * (self.action_max - self.action_min),
                num_slots,
            )
        else:
            slot_centres = torch.tensor(
                [0.5 * (self.action_min + self.action_max)]
            )
        self.slots = nn.ModuleList(
            [
                FlowSlot(
                    action_dim,
                    t_embed_dim,
                    hidden=hidden,
                    init_bias=-torch.ones(action_dim) * slot_centres[i],
                )
                for i in range(num_slots)
            ]
        )
        self.slot_values = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_slots)]
        )

    def _euler_sample(self, idx: int, n: int, explore: bool = True) -> torch.Tensor:

        device = self.slot_values[idx].device
        x = torch.randn(n, self.action_dim, device=device) * self.sample_noise
        t_grid = torch.linspace(1.0, 0.0, self.flow_steps + 1, device=device)
        for step in range(self.flow_steps):
            t_cur = t_grid[step].view(1, 1).expand(n, 1)
            t_next = t_grid[step + 1].view(1, 1).expand(n, 1)
            dt = t_next - t_cur
            t_embed = _embed_timestep(t_cur, self.t_embed_dim)
            v = self.slots[idx](x, t_embed)
            x = x + dt * v
            if explore and self.explore_noise > 0.0:
                kick = torch.randn_like(x) * self.explore_noise * (-dt).sqrt()
                x = x + kick
        return x

    def _to_action(self, x_raw: torch.Tensor) -> torch.Tensor:

        a = x_raw.clamp(self.action_min, self.action_max)
        return a.squeeze(-1) if (self.action_dim == 1 and a.dim() > 1) else a

    def sample_actions_with_raw(
        self, idx: int, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x_raw = self._euler_sample(idx, n)
        action = self._to_action(x_raw)
        return x_raw, action

    def sample_actions(self, idx: int, n: int, deterministic: bool = False) -> torch.Tensor:

        with torch.no_grad():
            return self._to_action(self._euler_sample(idx, n))

    def sample_no_grad(self, idx: int, n: int) -> torch.Tensor:
        with torch.no_grad():
            return self._to_action(self._euler_sample(idx, n))

    def value(self, idx: int) -> torch.Tensor:
        return self.slot_values[idx].squeeze()

    def velocity(
        self,
        idx: int,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_embed = _embed_timestep(t, self.t_embed_dim)
        return self.slots[idx](x_t, t_embed)

def _sample_slot(
    net: nn.Module, idx: int, n: int, no_grad: bool = True
) -> torch.Tensor:

    if no_grad:
        if isinstance(net, GaussianPopulation):
            return net.sample_no_grad(idx, n)
        elif isinstance(net, FlowPopulation):
            return net.sample_no_grad(idx, n)
    if isinstance(net, GaussianPopulation):
        a, _ = net.sample(idx, n)
        return a
    return net.sample_actions(idx, n)

def eval_payoffs_mc(
    net: nn.Module,
    game: GameSpec,
    num_samples: int = 2048,
) -> np.ndarray:

    n = net.num_slots
    U = np.zeros((n, n))
    with torch.no_grad():
        for i in range(n):
            a_i = _sample_slot(net, i, num_samples)
            for j in range(n):
                a_j = _sample_slot(net, j, num_samples)
                r = game.payoff_fn(a_i, a_j)
                U[i, j] = float(r.mean().item())
    return U

def _opponent_sample(
    net: nn.Module,
    sigma_row: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:

    w = sigma_row.clamp(min=0.0)
    s = w.sum().item()
    if s < 1e-8:
        raise ValueError("empty sigma row")
    probs = (w / s).detach()
    idx_draws = torch.distributions.Categorical(probs=probs).sample((batch_size,))
    out = torch.empty(batch_size)
    for j in torch.unique(idx_draws):
        mask = idx_draws == j
        k = int(mask.sum().item())
        out[mask] = _sample_slot(net, int(j.item()), k)
    return out

def gaussian_abr_step(
    net: GaussianPopulation,
    sigma: torch.Tensor,
    slot_i: int,
    batch_size: int,
    opt: optim.Optimizer,
    game: GameSpec,
    entropy_coef: float = 0.001,
    value_coef: float = 0.5,
) -> None:

    sigma_i = sigma[slot_i]
    if sigma_i.sum().item() < 1e-8:
        return

    a_j = _opponent_sample(net, sigma_i, batch_size).detach()

    a_i, logp_i = net.sample(slot_i, batch_size)
    a_i_clipped = a_i.clamp(net.action_min, net.action_max)

    reward = game.payoff_fn(a_i_clipped, a_j)

    v = net.value(slot_i)
    advantage = (reward - v).detach()

    policy_loss = -(logp_i * advantage).mean()
    value_loss = (reward.detach() - v).pow(2).mean()
    entropy = net.dist(slot_i).entropy().squeeze()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
    opt.step()

def flow_abr_step(
    net: FlowPopulation,
    old_net: FlowPopulation,
    sigma: torch.Tensor,
    slot_i: int,
    batch_size: int,
    opt: optim.Optimizer,
    game: GameSpec,
    beta: float = 1.0,
    adv_clip_max: float = 3.0,
    value_coef: float = 0.5,
    adaptive_weighting: bool = True,
    awr_temperature: float = 1.0,
    loss_mode: str = "awr",
) -> None:

    sigma_i = sigma[slot_i]
    if sigma_i.sum().item() < 1e-8:
        return

    action_dim = net.action_dim
    device = net.slot_values[slot_i].device

    with torch.no_grad():
        x_raw, a_i = net.sample_actions_with_raw(slot_i, batch_size)
    a_j = _opponent_sample(net, sigma_i, batch_size).detach()
    with torch.no_grad():
        reward = game.payoff_fn(a_i, a_j)

    x0 = x_raw

    v = net.value(slot_i)
    advantage = (reward - v).detach()
    value_loss = (reward.detach() - v).pow(2).mean()

    adv_std = advantage.std().clamp(min=1e-4)
    adv_norm = (advantage - advantage.mean()) / adv_std

    noise = torch.randn(batch_size, action_dim, device=device)
    t = torch.rand(batch_size, 1, device=device).clamp(min=1e-3)
    x_t = (1.0 - t) * x0 + t * noise

    if loss_mode == "awr":

        w = torch.softmax(adv_norm.clamp(-adv_clip_max, adv_clip_max) / awr_temperature, dim=0)
        w = w * batch_size
        v_cur = net.velocity(slot_i, x_t, t)
        x0_pred = x_t - t * v_cur
        pos_loss = ((x0_pred - x0) ** 2).mean(dim=-1)
        policy_loss = (w.view(-1) * pos_loss).mean()
    else:
        adv_c = adv_norm.clamp(-adv_clip_max, adv_clip_max)
        r = ((adv_c / adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0).view(-1, 1)
        with torch.no_grad():
            v_old = old_net.velocity(slot_i, x_t, t).detach()
        v_cur = net.velocity(slot_i, x_t, t)
        v_pos = beta * v_cur + (1.0 - beta) * v_old
        v_neg = (1.0 + beta) * v_old - beta * v_cur
        x0_pos = x_t - t * v_pos
        x0_neg = x_t - t * v_neg
        pos_loss = ((x0_pos - x0) ** 2).mean(dim=-1)
        neg_loss = ((x0_neg - x0) ** 2).mean(dim=-1)
        r_flat = r.squeeze(-1)
        policy_loss = ((r_flat * pos_loss + (1.0 - r_flat) * neg_loss) / beta).mean()

    loss = policy_loss + value_coef * value_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
    opt.step()

def sigma_self_play(n: int) -> torch.Tensor:
    return torch.full((n, n), 1.0 / n)

def sigma_strategic_cycle(n: int = 3) -> torch.Tensor:
    sigma = torch.zeros(n, n)
    for i in range(n):
        sigma[i, (i + 1) % n] = 1.0
    return sigma

def sigma_fictitious_play(n: int) -> torch.Tensor:
    sigma = torch.zeros(n, n)
    for i in range(1, n):
        sigma[i, :i] = 1.0 / float(i)
    return sigma

def unique_rows(sigma: torch.Tensor) -> list[int]:
    rows: list[np.ndarray] = []
    reps: list[int] = []
    for i in range(sigma.shape[0]):
        r = sigma[i].numpy().round(6)
        if not any(np.allclose(r, rr) for rr in rows):
            rows.append(r)
            reps.append(i)
    return reps

@dataclass
class TrainConfig:
    outer_iters: int = 200
    abr_steps_per_iter: int = 4
    episodes_per_step: int = 256
    lr: float = 3e-3
    entropy_coef: float = 0.01
    seed: int = 0

    flow_steps: int = 5
    t_embed_dim: int = 8
    hidden: int = 64
    nft_beta: float = 1.0
    nft_adv_clip_max: float = 3.0
    old_policy_ema: float = 0.0
    flow_sample_noise: float = 1.5
    flow_explore_noise: float = 0.3
    flow_loss_mode: str = "awr"
    flow_awr_temperature: float = 1.0
    gaussian_max_std_fraction: float = 0.25

    mc_eval_samples: int = 1024

PolicyType = str

def make_population(
    policy_type: PolicyType,
    num_slots: int,
    game: GameSpec,
    cfg: TrainConfig,
) -> nn.Module:
    if policy_type == "gaussian":
        return GaussianPopulation(
            num_slots=num_slots,
            action_min=game.action_min,
            action_max=game.action_max,
            init_log_std=math.log(0.5 * (cfg.gaussian_max_std_fraction * game.action_range())),
            max_std_fraction=cfg.gaussian_max_std_fraction,
        )
    elif policy_type == "flow":
        return FlowPopulation(
            num_slots=num_slots,
            action_min=game.action_min,
            action_max=game.action_max,
            flow_steps=cfg.flow_steps,
            t_embed_dim=cfg.t_embed_dim,
            hidden=cfg.hidden,
            action_dim=1,
            sample_noise=cfg.flow_sample_noise,
            explore_noise=cfg.flow_explore_noise,
        )
    raise ValueError(f"Unknown policy_type: {policy_type}")

def _update_old(old_net: FlowPopulation, net: FlowPopulation, ema: float) -> None:
    with torch.no_grad():
        for p_old, p_new in zip(old_net.parameters(), net.parameters()):
            if ema > 0.0:
                p_old.data.mul_(ema).add_(p_new.data, alpha=1.0 - ema)
            else:
                p_old.data.copy_(p_new.data)

def abr_step(
    policy_type: PolicyType,
    net: nn.Module,
    old_net: Optional[nn.Module],
    sigma: torch.Tensor,
    slot_i: int,
    batch_size: int,
    opt: optim.Optimizer,
    game: GameSpec,
    cfg: TrainConfig,
) -> None:
    if policy_type == "gaussian":
        gaussian_abr_step(
            net,
            sigma,
            slot_i,
            batch_size,
            opt,
            game,
            entropy_coef=cfg.entropy_coef,
        )
    else:
        flow_abr_step(
            net,
            old_net,
            sigma,
            slot_i,
            batch_size,
            opt,
            game,
            beta=cfg.nft_beta,
            adv_clip_max=cfg.nft_adv_clip_max,
            loss_mode=cfg.flow_loss_mode,
            awr_temperature=cfg.flow_awr_temperature,
        )

def _record_history(
    net: nn.Module,
    history: dict[int, list[np.ndarray]],
    mc_samples: int,
) -> None:

    n = net.num_slots
    for i in range(n):
        samp = _sample_slot(net, i, mc_samples).detach().cpu().numpy().reshape(-1)
        history[i].append(samp)

def train_neupl_static(
    sigma: torch.Tensor,
    cfg: TrainConfig,
    game: GameSpec,
    policy_type: PolicyType,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    n = sigma.shape[0]
    net = make_population(policy_type, n, game, cfg)
    old_net = None
    if policy_type == "flow":
        old_net = make_population(policy_type, n, game, cfg)
        old_net.load_state_dict(net.state_dict())
        for p in old_net.parameters():
            p.requires_grad_(False)

    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    history: dict[int, list[np.ndarray]] = {i: [] for i in range(n)}

    for it in range(cfg.outer_iters):
        reps = unique_rows(sigma)
        for i in reps:
            if sigma[i].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps_per_iter):
                abr_step(
                    policy_type,
                    net,
                    old_net,
                    sigma,
                    i,
                    cfg.episodes_per_step,
                    opt,
                    game,
                    cfg,
                )

        if policy_type == "flow":
            _update_old(old_net, net, cfg.old_policy_ema)

        _record_history(net, history, mc_samples=min(256, cfg.mc_eval_samples))

        if (it + 1) % 50 == 0:
            print(f"  [static {policy_type} iter {it+1}/{cfg.outer_iters}]")

    history = {i: np.stack(v) for i, v in history.items()}
    return net, history

def train_neupl_adaptive(
    n: int,
    cfg: TrainConfig,
    game: GameSpec,
    policy_type: PolicyType,
    epoch_len: int = 10,
) -> tuple[nn.Module, dict, list[np.ndarray], list[np.ndarray]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    net = make_population(policy_type, n, game, cfg)
    old_net = None
    if policy_type == "flow":
        old_net = make_population(policy_type, n, game, cfg)
        old_net.load_state_dict(net.state_dict())
        for p in old_net.parameters():
            p.requires_grad_(False)

    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    sigma = torch.full((n, n), 1.0 / n)
    history: dict[int, list[np.ndarray]] = {i: [] for i in range(n)}
    sigma_history: list[np.ndarray] = []
    payoff_history: list[np.ndarray] = []

    for it in range(cfg.outer_iters):
        if it % epoch_len == 0:
            U = eval_payoffs_mc(net, game, num_samples=cfg.mc_eval_samples)
            sigma_np = psro_nash_meta_graph_solver(U)
            sigma = torch.from_numpy(sigma_np).float()
            sigma_history.append(sigma_np.copy())
            payoff_history.append(U.copy())

        reps = unique_rows(sigma)
        for i in reps:
            if sigma[i].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps_per_iter):
                abr_step(
                    policy_type,
                    net,
                    old_net,
                    sigma,
                    i,
                    cfg.episodes_per_step,
                    opt,
                    game,
                    cfg,
                )

        if policy_type == "flow":
            _update_old(old_net, net, cfg.old_policy_ema)

        _record_history(net, history, mc_samples=min(256, cfg.mc_eval_samples))

        if (it + 1) % 50 == 0:
            print(f"  [adaptive {policy_type} iter {it+1}/{cfg.outer_iters}]")

    history = {i: np.stack(v) for i, v in history.items()}
    return net, history, sigma_history, payoff_history

def train_psro(
    n_iters: int,
    cfg: TrainConfig,
    game: GameSpec,
    policy_type: PolicyType,
    abr_steps: int = 800,
) -> tuple[list[nn.Module], dict, list[np.ndarray], list[np.ndarray]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    pi0 = make_population(policy_type, 1, game, cfg)
    population: list[nn.Module] = [pi0]
    history: dict[int, list[np.ndarray]] = {0: [_sample_slot(pi0, 0, 256).numpy()]}
    sigma_history: list[np.ndarray] = []
    payoff_history: list[np.ndarray] = []

    for iteration in range(n_iters):

        n_pop = len(population)
        U = np.zeros((n_pop, n_pop))
        with torch.no_grad():
            for i in range(n_pop):
                a_i = _sample_slot(population[i], 0, cfg.mc_eval_samples)
                for j in range(n_pop):
                    a_j = _sample_slot(population[j], 0, cfg.mc_eval_samples)
                    U[i, j] = float(game.payoff_fn(a_i, a_j).mean().item())
        payoff_history.append(U.copy())

        nash_mix = solve_nash_zero_sum(U)
        sigma_history.append(nash_mix.copy())

        new_pi = make_population(policy_type, 1, game, cfg)
        opt_new = optim.Adam(new_pi.parameters(), lr=cfg.lr)
        old_new = None
        if policy_type == "flow":
            old_new = make_population(policy_type, 1, game, cfg)
            old_new.load_state_dict(new_pi.state_dict())
            for p in old_new.parameters():
                p.requires_grad_(False)

        for step in range(abr_steps):

            j = int(
                torch.distributions.Categorical(
                    probs=torch.from_numpy(nash_mix).float()
                ).sample().item()
            )
            a_j = _sample_slot(population[j], 0, cfg.episodes_per_step).detach()

            if policy_type == "gaussian":
                a_i, logp = new_pi.sample(0, cfg.episodes_per_step)
                a_i_c = a_i.clamp(new_pi.action_min, new_pi.action_max)
                r = game.payoff_fn(a_i_c, a_j)
                v = new_pi.value(0)
                adv = (r - v).detach()
                policy_loss = -(logp * adv).mean()
                value_loss = (r.detach() - v).pow(2).mean()
                entropy = new_pi.dist(0).entropy().squeeze()
                loss = policy_loss + 0.5 * value_loss - cfg.entropy_coef * entropy
                opt_new.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(new_pi.parameters(), max_norm=2.0)
                opt_new.step()
            else:

                sigma_one = torch.ones((1, 1))

                flow_abr_step_psro_helper(
                    new_pi,
                    old_new,
                    a_j,
                    opt_new,
                    game,
                    cfg,
                )

        if policy_type == "flow":
            _update_old(old_new, new_pi, cfg.old_policy_ema)

        population.append(new_pi)
        history[iteration + 1] = [_sample_slot(new_pi, 0, 256).numpy()]

        print(
            f"  [PSRO {policy_type} iter {iteration+1}/{n_iters}] pop={len(population)}"
        )

    n_pop = len(population)
    U_final = np.zeros((n_pop, n_pop))
    with torch.no_grad():
        for i in range(n_pop):
            a_i = _sample_slot(population[i], 0, cfg.mc_eval_samples)
            for j in range(n_pop):
                a_j = _sample_slot(population[j], 0, cfg.mc_eval_samples)
                U_final[i, j] = float(game.payoff_fn(a_i, a_j).mean().item())
    payoff_history.append(U_final)
    sigma_history.append(solve_nash_zero_sum(U_final))

    return population, history, sigma_history, payoff_history

def flow_abr_step_psro_helper(
    net: FlowPopulation,
    old_net: FlowPopulation,
    a_j_batch: torch.Tensor,
    opt: optim.Optimizer,
    game: GameSpec,
    cfg: TrainConfig,
) -> None:

    action_dim = net.action_dim
    device = net.slot_values[0].device
    beta = cfg.nft_beta
    adv_clip_max = cfg.nft_adv_clip_max
    loss_mode = cfg.flow_loss_mode
    awr_temperature = cfg.flow_awr_temperature
    batch_size = a_j_batch.shape[0]

    with torch.no_grad():
        x_raw, a_i = net.sample_actions_with_raw(0, batch_size)
        reward = game.payoff_fn(a_i, a_j_batch)

    x0 = x_raw

    v = net.value(0)
    advantage = (reward - v).detach()
    value_loss = (reward.detach() - v).pow(2).mean()
    adv_std = advantage.std().clamp(min=1e-4)
    adv_norm = (advantage - advantage.mean()) / adv_std
    adv_c = adv_norm.clamp(-adv_clip_max, adv_clip_max)
    r = (adv_c / adv_clip_max) / 2.0 + 0.5
    r = r.clamp(0.0, 1.0).view(-1, 1)

    noise = torch.randn(batch_size, action_dim, device=device)
    t = torch.rand(batch_size, 1, device=device).clamp(min=1e-3)
    x_t = (1.0 - t) * x0 + t * noise

    if loss_mode == "awr":
        w = torch.softmax(adv_norm.clamp(-adv_clip_max, adv_clip_max) / awr_temperature, dim=0)
        w = w * batch_size
        v_cur = net.velocity(0, x_t, t)
        x0_pred = x_t - t * v_cur
        pos_loss = ((x0_pred - x0) ** 2).mean(dim=-1)
        policy_loss = (w.view(-1) * pos_loss).mean()
    else:
        with torch.no_grad():
            v_old = old_net.velocity(0, x_t, t).detach()
        v_cur = net.velocity(0, x_t, t)
        v_pos = beta * v_cur + (1.0 - beta) * v_old
        v_neg = (1.0 + beta) * v_old - beta * v_cur
        x0_pos = x_t - t * v_pos
        x0_neg = x_t - t * v_neg
        pos_loss = ((x0_pos - x0) ** 2).mean(dim=-1)
        neg_loss = ((x0_neg - x0) ** 2).mean(dim=-1)
        r_flat = r.squeeze(-1)
        policy_loss = ((r_flat * pos_loss + (1.0 - r_flat) * neg_loss) / beta).mean()

    loss = policy_loss + 0.5 * value_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
    opt.step()

def train_self_play(
    cfg: TrainConfig,
    game: GameSpec,
    policy_type: PolicyType,
) -> tuple[nn.Module, dict]:
    sigma = torch.ones((1, 1))
    return train_neupl_static(sigma, cfg, game, policy_type)

def exploitability(
    net: nn.Module,
    game: GameSpec,
    sigma_row: Optional[np.ndarray] = None,
    num_samples: int = 2048,
) -> float:

    n = net.num_slots

    if sigma_row is None:
        U = eval_payoffs_mc(net, game, num_samples=num_samples)
        sigma_row = solve_nash_zero_sum(U)

    _, _, _, v_ref = discretized_nash(game)

    grid = torch.from_numpy(game.sample_grid()).float()

    ns = max(256, num_samples // max(1, n))
    mixture_actions = []
    with torch.no_grad():
        for i in range(n):
            w = float(sigma_row[i])
            if w < 1e-8:
                continue
            k = max(1, int(round(w * num_samples)))
            mixture_actions.append(_sample_slot(net, i, k).view(-1))
    if not mixture_actions:
        return float("inf")
    x_samples = torch.cat(mixture_actions)

    xs = x_samples.view(-1, 1)
    ys = grid.view(1, -1)
    with torch.no_grad():
        pay_matrix = game.payoff_fn(xs, ys)
    mean_payoff_per_y = pay_matrix.mean(dim=0).cpu().numpy()

    br_up = float(mean_payoff_per_y.max())
    br_down = float(mean_payoff_per_y.min())
    return (br_up - v_ref) + (v_ref - br_down)

def plot_game_landscape(
    game: GameSpec,
    nash_row: np.ndarray,
    nash_col: np.ndarray,
    grid: np.ndarray,
    save: str,
) -> None:
    xx, yy = np.meshgrid(grid, grid)
    xt = torch.from_numpy(xx).float()
    yt = torch.from_numpy(yy).float()
    with torch.no_grad():
        zz = game.payoff_fn(xt, yt).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im = axes[0].imshow(
        zz,
        extent=[grid[0], grid[-1], grid[0], grid[-1]],
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
    )
    axes[0].set_title(f"{game.name}: J_1(x, y)")
    axes[0].set_xlabel("x (player 1)")
    axes[0].set_ylabel("y (player 2)")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    axes[1].plot(grid, nash_row, "b-", lw=2)
    axes[1].fill_between(grid, 0, nash_row, alpha=0.3)
    axes[1].set_title("Nash mixture (player 1)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("p(x)")

    axes[2].plot(grid, nash_col, "r-", lw=2)
    axes[2].fill_between(grid, 0, nash_col, alpha=0.3, color="r")
    axes[2].set_title("Nash mixture (player 2)")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("p(y)")

    plt.tight_layout()
    fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save}")

def plot_population_actions(
    net: nn.Module,
    game: GameSpec,
    title: str,
    save: str,
    nash_row: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    num_samples: int = 2048,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, net.num_slots)))
    for i in range(net.num_slots):
        samp = _sample_slot(net, i, num_samples).numpy().reshape(-1)
        ax.hist(
            samp,
            bins=60,
            alpha=0.45,
            color=colors[i],
            density=True,
            label=f"slot {i}",
        )
    if nash_row is not None and grid is not None:
        ax.plot(grid, nash_row / max(nash_row.max(), 1e-8), "k--", lw=1.5, label="Nash (normalized)")
    ax.set_xlim(game.action_min, game.action_max)
    ax.set_title(title)
    ax.set_xlabel("action")
    ax.set_ylabel("density")
    if net.num_slots <= 6:
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save}")

def plot_exploitability_curves(
    curves: dict[str, list[float]],
    title: str,
    save: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, vals in curves.items():
        ax.plot(vals, lw=1.8, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("exploitability gap")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save}")

def _track_exploitability(
    net: nn.Module,
    game: GameSpec,
    samples: int = 1024,
) -> float:
    return exploitability(net, game, num_samples=samples)

def run_experiment_suite(
    game: GameSpec,
    cfg: TrainConfig,
    save_dir: str,
    experiments: list[str],
) -> dict:

    results: dict = {}
    exploitability_curves: dict[str, list[float]] = {}

    grid, nash_row, nash_col, v_ref = discretized_nash(game)
    plot_game_landscape(
        game,
        nash_row,
        nash_col,
        grid,
        save=os.path.join(save_dir, f"{game.name}_nash.png"),
    )
    print(
        f"  Reference: game value v* = {v_ref:.4f}, Nash support width = "
        f"{(nash_row > 1e-3).sum()} / {grid.size} grid cells"
    )
    results["nash_row"] = nash_row
    results["nash_col"] = nash_col
    results["value"] = v_ref

    for pt in ["gaussian", "flow"]:
        key = f"self_play_{pt}"
        if key in experiments:
            print(f"\n[{game.name}] Self-Play ({pt})")
            net, _ = train_self_play(cfg, game, pt)
            expl = _track_exploitability(net, game, cfg.mc_eval_samples)
            print(f"  Exploitability: {expl:.4f}")
            results[key] = {"net": net, "expl": expl}
            plot_population_actions(
                net,
                game,
                title=f"{game.name} | Self-Play {pt} | expl={expl:.3f}",
                save=os.path.join(save_dir, f"{game.name}_{key}.png"),
                nash_row=nash_row,
                grid=grid,
            )

    for (key_suffix, sigma_cons, slots) in [
        ("cycle", sigma_strategic_cycle, 3),
        ("fp", sigma_fictitious_play, 5),
    ]:
        for pt in ["gaussian", "flow"]:
            key = f"{key_suffix}_{pt}"
            if key not in experiments:
                continue
            print(f"\n[{game.name}] {key_suffix} ({pt}, N={slots})")
            sigma = sigma_cons(slots)
            net, hist = train_neupl_static(sigma, cfg, game, pt)
            expl = _track_exploitability(net, game, cfg.mc_eval_samples)
            print(f"  Exploitability: {expl:.4f}")
            results[key] = {"net": net, "hist": hist, "expl": expl}
            plot_population_actions(
                net,
                game,
                title=f"{game.name} | {key_suffix} {pt} | expl={expl:.3f}",
                save=os.path.join(save_dir, f"{game.name}_{key}.png"),
                nash_row=nash_row,
                grid=grid,
            )

    for pt in ["gaussian", "flow"]:
        key = f"adaptive_{pt}"
        if key not in experiments:
            continue
        print(f"\n[{game.name}] Adaptive PSRO-Nash ({pt}, N=4)")
        cfg_ad = TrainConfig(**{**cfg.__dict__, "outer_iters": cfg.outer_iters})
        net, hist, sigma_hist, pay_hist = train_neupl_adaptive(
            n=4, cfg=cfg_ad, game=game, policy_type=pt, epoch_len=10
        )
        expl = _track_exploitability(net, game, cfg.mc_eval_samples)
        print(f"  Exploitability: {expl:.4f}")
        results[key] = {"net": net, "hist": hist, "expl": expl}
        plot_population_actions(
            net,
            game,
            title=f"{game.name} | Adaptive PSRO-Nash {pt} | expl={expl:.3f}",
            save=os.path.join(save_dir, f"{game.name}_{key}.png"),
            nash_row=nash_row,
            grid=grid,
        )

    for pt in ["gaussian", "flow"]:
        key = f"psro_{pt}"
        if key not in experiments:
            continue
        print(f"\n[{game.name}] Classical PSRO ({pt})")
        pop, hist_psro, sigma_hist, pay_hist = train_psro(
            n_iters=4,
            cfg=cfg,
            game=game,
            policy_type=pt,
            abr_steps=max(200, cfg.outer_iters),
        )

        mix = sigma_hist[-1][: len(pop) - 1] if len(pop) > 1 else np.array([1.0])

        fig, ax = plt.subplots(figsize=(6, 4))
        for i, p in enumerate(pop):
            samp = _sample_slot(p, 0, 1024).numpy().reshape(-1)
            ax.hist(samp, bins=60, alpha=0.4, density=True, label=f"pi_{i}")
        ax.set_xlim(game.action_min, game.action_max)
        ax.set_title(f"{game.name} | PSRO {pt} | pop={len(pop)}")
        ax.set_xlabel("action")
        if nash_row is not None:
            ax.plot(grid, nash_row / max(nash_row.max(), 1e-8), "k--", lw=1.2)
        ax.legend(fontsize=8)
        plt.tight_layout()
        psro_fig_path = os.path.join(save_dir, f"{game.name}_{key}.png")
        fig.savefig(psro_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {psro_fig_path}")
        results[key] = {"pop": pop, "sigma": sigma_hist[-1]}

    expl_items = [
        (k, v.get("expl"))
        for k, v in results.items()
        if isinstance(v, dict) and "expl" in v
    ]
    if expl_items:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [k for k, _ in expl_items]
        values = [v for _, v in expl_items]
        colors = [
            "tab:blue" if "gaussian" in l else "tab:orange" for l in labels
        ]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("exploitability gap")
        ax.set_title(f"{game.name}: exploitability by method")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{game.name}_exploitability.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    return results

ALL_EXPERIMENTS = [
    "self_play_gaussian",
    "self_play_flow",
    "cycle_gaussian",
    "cycle_flow",
    "fp_gaussian",
    "fp_flow",
    "adaptive_gaussian",
    "adaptive_flow",
    "psro_gaussian",
    "psro_flow",
]

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowPL vs NeuPL on continuous Cartesian games"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="checkerboard",
        choices=list(GAMES.keys()) + ["all"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-iters", type=int, default=200)
    parser.add_argument("--abr-steps-per-iter", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--nft-beta", type=float, default=1.0)
    parser.add_argument("--save-dir", type=str, default="results_cartesian")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "self_play_gaussian",
            "self_play_flow",
            "adaptive_gaussian",
            "adaptive_flow",
        ],
        choices=ALL_EXPERIMENTS,
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    cfg = TrainConfig(
        outer_iters=args.outer_iters,
        abr_steps_per_iter=args.abr_steps_per_iter,
        episodes_per_step=args.episodes,
        lr=args.lr,
        flow_steps=args.flow_steps,
        hidden=args.hidden,
        nft_beta=args.nft_beta,
        seed=args.seed,
    )

    games_to_run = list(GAMES.keys()) if args.game == "all" else [args.game]

    all_results: dict[str, dict] = {}
    for g_name in games_to_run:
        game = GAMES[g_name]
        sub_dir = os.path.join(args.save_dir, g_name)
        os.makedirs(sub_dir, exist_ok=True)
        print("\n" + "=" * 70)
        print(f"  Game: {g_name}   [{game.action_min:.2f}, {game.action_max:.2f}]")
        print("=" * 70)
        all_results[g_name] = run_experiment_suite(
            game, cfg, sub_dir, args.experiments
        )

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for g_name, res in all_results.items():
        print(f"\n[{g_name}]  v* = {res.get('value', float('nan')):.4f}")
        for k, v in res.items():
            if isinstance(v, dict) and "expl" in v:
                print(f"  {k:30s}  expl = {v['expl']:.4f}")

    print(f"\nDone. All plots saved to: {args.save_dir}/")

if __name__ == "__main__":
    main()
