

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

TWOPI = 2.0 * np.pi

MULTIMODAL_ALPHA = 0.5
MULTIMODAL_K = 3

def payoff_multimodal(theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:

    d = theta1 - theta2
    return torch.sin(d) + MULTIMODAL_ALPHA * torch.sin(MULTIMODAL_K * d)

def payoff_matrix_multimodal(pop_theta: torch.Tensor) -> torch.Tensor:
    t1 = pop_theta.unsqueeze(1)
    t2 = pop_theta.unsqueeze(0)
    return payoff_multimodal(t1, t2)

def sigma_strategic_cycle(n: int) -> torch.Tensor:
    sigma = torch.zeros(n, n)
    for i in range(n):
        sigma[i, (i + 1) % n] = 1.0
    return sigma

def solve_nash_zero_sum(payoff: np.ndarray) -> np.ndarray:
    n = payoff.shape[0]
    if n == 1:
        return np.array([1.0])
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.zeros((n, n + 1))
    A_ub[:, :n] = -payoff.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * n + [(None, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if result.success:
        p = np.maximum(result.x[:n], 0.0)
        p /= p.sum()
        return p
    return np.ones(n) / n

def psro_nash_meta_graph_solver(payoff: np.ndarray) -> np.ndarray:
    n = payoff.shape[0]
    sigma = np.zeros((n, n))
    for i in range(1, n):
        sigma[i, :i] = solve_nash_zero_sum(payoff[:i, :i])
    return sigma

def unique_rows(sigma: torch.Tensor) -> list[int]:
    rows, reps = [], []
    for i in range(sigma.shape[0]):
        r = sigma[i].numpy().round(6)
        if not any(np.allclose(r, rr) for rr in rows):
            rows.append(r)
            reps.append(i)
    return reps

class CircleNet(nn.Module):
    def __init__(self, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        init = torch.linspace(0, TWOPI - TWOPI / num_slots, num_slots) + 0.2 * torch.randn(num_slots)
        self.slot_angles = nn.ParameterList(
            [nn.Parameter(init[i].unsqueeze(0)) for i in range(num_slots)]
        )

    def angle(self, idx: int) -> torch.Tensor:
        return self.slot_angles[idx].squeeze(0)

    def get_angle_np(self, idx: int) -> float:
        with torch.no_grad():
            return float(self.angle(idx).item() % TWOPI)

class CircleNetStochastic(nn.Module):

    def __init__(self, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        init = torch.linspace(0, TWOPI - TWOPI / num_slots, num_slots) + 0.2 * torch.randn(num_slots)
        self.slot_means = nn.ParameterList(
            [nn.Parameter(init[i].unsqueeze(0)) for i in range(num_slots)]
        )
        self.slot_log_std = nn.ParameterList(
            [nn.Parameter(torch.tensor(-1.0)) for _ in range(num_slots)]
        )

    def mean_angle(self, idx: int) -> torch.Tensor:
        return self.slot_means[idx].squeeze(0)

    def std_angle(self, idx: int) -> torch.Tensor:
        return torch.exp(self.slot_log_std[idx]).clamp(min=1e-3)

    def sample_angle(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.mean_angle(idx)
        s = self.std_angle(idx)
        dist = torch.distributions.Normal(m, s)
        theta = dist.rsample()
        log_p = dist.log_prob(theta).sum()
        return theta, log_p

    def get_angle_np(self, idx: int) -> float:
        with torch.no_grad():
            return float(self.mean_angle(idx).item() % TWOPI)

def loss_exact_multimodal(net: CircleNet, sigma: torch.Tensor, slot_i: int) -> torch.Tensor:
    sigma_i = sigma[slot_i]
    theta_i = net.angle(slot_i)
    payoff = torch.tensor(0.0, device=theta_i.device)
    for j in range(sigma.shape[0]):
        w = sigma_i[j].item()
        if w > 1e-8:
            theta_j = net.angle(j).detach()
            payoff = payoff + w * payoff_multimodal(theta_i, theta_j)
    return -payoff

def eval_payoffs(net: CircleNet) -> np.ndarray:
    n = net.num_slots
    with torch.no_grad():
        angles = torch.stack([net.angle(i) for i in range(n)])
        return payoff_matrix_multimodal(angles).numpy()

def rl_abr_step_multimodal(
    net: CircleNetStochastic,
    sigma: torch.Tensor,
    slot_i: int,
    batch_size: int,
    opt: optim.Optimizer,
) -> None:

    sigma_i = sigma[slot_i]
    if sigma_i.sum().item() < 1e-8:
        return
    opp_dist = torch.distributions.Categorical(probs=sigma_i / sigma_i.sum())
    log_probs, rewards = [], []
    for _ in range(batch_size):
        j = opp_dist.sample().item()
        theta_i, log_p_i = net.sample_angle(slot_i)
        with torch.no_grad():
            theta_j, _ = net.sample_angle(j)
        r = payoff_multimodal(theta_i, theta_j)
        log_probs.append(log_p_i)
        rewards.append(r)
    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    advantage = (rewards - rewards.detach().mean()).detach()
    loss = -(log_probs * advantage).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

@dataclass
class Config:
    outer_iters: int = 350
    abr_steps: int = 15
    lr: float = 0.08
    seed: int = 0

@dataclass
class ConfigRL:
    outer_iters: int = 350
    abr_steps_per_iter: int = 5
    batch_size: int = 128
    lr: float = 1e-2
    seed: int = 0

def train_multimodal(sigma: torch.Tensor, cfg: Config) -> tuple[CircleNet, dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    n = sigma.shape[0]
    net = CircleNet(num_slots=n)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    history = {i: [] for i in range(n)}
    for it in range(cfg.outer_iters):
        reps = unique_rows(sigma)
        for i in reps:
            if sigma[i].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps):
                loss = loss_exact_multimodal(net, sigma, i)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        for i in range(n):
            history[i].append(net.get_angle_np(i))
        if (it + 1) % 70 == 0:
            angs = [f"{np.rad2deg(history[i][-1]):.0f}°" for i in range(n)]
            print(f"  [iter {it+1}/{cfg.outer_iters}] angles: {angs}")
    return net, history

def train_multimodal_rl(sigma: torch.Tensor, cfg: ConfigRL) -> tuple[CircleNetStochastic, dict]:

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    n = sigma.shape[0]
    net = CircleNetStochastic(num_slots=n)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    history = {i: [] for i in range(n)}
    for it in range(cfg.outer_iters):
        reps = unique_rows(sigma)
        for i in reps:
            if sigma[i].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps_per_iter):
                rl_abr_step_multimodal(net, sigma, i, cfg.batch_size, opt)
        for i in range(n):
            history[i].append(net.get_angle_np(i))
        if (it + 1) % 70 == 0:
            angs = [f"{np.rad2deg(history[i][-1]):.0f}°" for i in range(n)]
            print(f"  [RL iter {it+1}/{cfg.outer_iters}] angles: {angs}")
    return net, history

def spacing_deg(angles_deg: list[float]) -> list[float]:
    s = sorted(angles_deg)
    n = len(s)
    return [float((s[(i+1) % n] - s[i]) % 360) for i in range(n)]

def draw_circle(ax: plt.Axes, title: str):
    th = np.linspace(0, TWOPI, 100)
    ax.plot(np.cos(th), np.sin(th), "k-", linewidth=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(title, fontsize=13)

def scatter_traj(ax: plt.Axes, angles_hist: list[float], s: int = 45):
    xs = [np.cos(a) for a in angles_hist]
    ys = [np.sin(a) for a in angles_hist]
    t = len(angles_hist)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, t))
    ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=2, zorder=0)
    ax.scatter(xs, ys, c=colors, s=s, alpha=0.85, edgecolors="k", linewidths=0.8, zorder=1)

def main():
    parser = argparse.ArgumentParser(description="NeuPL hard game: Multimodal Circle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-iters", type=int, default=350)
    parser.add_argument("--save-dir", type=str, default="results_hard")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = Config(outer_iters=args.outer_iters, seed=args.seed)
    cfg_rl = ConfigRL(outer_iters=args.outer_iters, seed=args.seed + 1)

    sigma = sigma_strategic_cycle(3)

    print("\n" + "=" * 60)
    print("  HARD GAME: Multimodal Circle (N=3 strategic cycle)")
    print("  J(θ1,θ2) = sin(θ1-θ2) + 0.5*sin(3*(θ1-θ2))  → 3 local BRs")
    print("=" * 60)

    print("\n  --- Exact ABR ---")
    net_exact, history_exact = train_multimodal(sigma, cfg)
    final_exact = [float(np.rad2deg(history_exact[i][-1])) for i in range(3)]
    spacings_exact = spacing_deg(final_exact)
    print(f"  Final angles (deg): {[round(f, 1) for f in final_exact]}")
    print(f"  Spacings (deg):     {[round(s, 1) for s in spacings_exact]}")

    print("\n  --- RL ABR (REINFORCE) ---")
    net_rl, history_rl = train_multimodal_rl(sigma, cfg_rl)
    final_rl = [float(np.rad2deg(history_rl[i][-1])) for i in range(3)]
    spacings_rl = spacing_deg(final_rl)
    print(f"  Final angles (deg): {[round(f, 1) for f in final_rl]}")
    print(f"  Spacings (deg):     {[round(s, 1) for s in spacings_rl]}")

    print("\n  (Clean cycle would have ~120° or ~90° uniform spacing;")
    print("   multimodal BRs cause uneven or collapsed spacing → NeuPL 'fails'.)")

    for label, history in [("exact", history_exact), ("rl", history_rl)]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        draw_circle(ax, f"Multimodal Circle — {label.upper()} ABR\nred=start → green=end")
        for i in range(3):
            scatter_traj(ax, history[i], s=38)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7, label="Time")
        plt.tight_layout()
        path = f"{args.save_dir}/hard_multimodal_circle_{label}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

if __name__ == "__main__":
    main()
