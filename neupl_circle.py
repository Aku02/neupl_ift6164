

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

def circle_payoff(theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:

    return torch.sin(theta1 - theta2)

def circle_payoff_matrix(pop_theta: torch.Tensor) -> torch.Tensor:

    t1 = pop_theta.unsqueeze(1)
    t2 = pop_theta.unsqueeze(0)
    return torch.sin(t1 - t2)

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
    rows: list[np.ndarray] = []
    reps: list[int] = []
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

        init_angles = torch.linspace(0, TWOPI - TWOPI / num_slots, num_slots) + 0.1 * torch.randn(num_slots)
        self.slot_angles = nn.ParameterList(
            [nn.Parameter(init_angles[i].unsqueeze(0)) for i in range(num_slots)]
        )

    def angle(self, idx: int) -> torch.Tensor:
        return self.slot_angles[idx].squeeze(0)

    def angles(self) -> torch.Tensor:
        return torch.stack([self.slot_angles[i].squeeze(0) for i in range(self.num_slots)])

    def get_angle_np(self, idx: int) -> float:
        with torch.no_grad():
            return float(self.angle(idx).item() % TWOPI)

class CircleNetStochastic(nn.Module):

    def __init__(self, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        init = torch.linspace(0, TWOPI - TWOPI / num_slots, num_slots) + 0.1 * torch.randn(num_slots)
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

def circle_loss_exact(
    net: CircleNet,
    sigma: torch.Tensor,
    slot_i: int,
) -> torch.Tensor:

    sigma_i = sigma[slot_i]
    theta_i = net.angle(slot_i)
    payoff = torch.tensor(0.0, device=theta_i.device)
    for j in range(sigma.shape[0]):
        w = sigma_i[j].item()
        if w > 1e-8:
            theta_j = net.angle(j).detach()
            payoff = payoff + w * circle_payoff(theta_i, theta_j)
    return -payoff

def circle_rl_abr_step(
    net: CircleNetStochastic,
    sigma: torch.Tensor,
    slot_i: int,
    batch_size: int,
    opt: optim.Optimizer,
):

    sigma_i = sigma[slot_i]
    if sigma_i.sum().item() < 1e-8:
        return
    opp_dist = torch.distributions.Categorical(probs=sigma_i / sigma_i.sum())

    log_probs = []
    rewards = []

    for _ in range(batch_size):
        j = opp_dist.sample().item()
        theta_i, log_p_i = net.sample_angle(slot_i)
        with torch.no_grad():
            theta_j, _ = net.sample_angle(j)
        r = circle_payoff(theta_i, theta_j)
        log_probs.append(log_p_i)
        rewards.append(r)

    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    advantage = (rewards - rewards.detach().mean()).detach()
    loss = -(log_probs * advantage).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

def eval_payoffs_circle(net, use_means: bool = True) -> np.ndarray:

    n = net.num_slots
    with torch.no_grad():
        if use_means and hasattr(net, "slot_means"):
            angles = torch.stack([net.mean_angle(i) for i in range(n)])
        else:
            angles = net.angles()
        M = circle_payoff_matrix(angles)
    return M.numpy()

@dataclass
class TrainConfig:
    outer_iters: int = 400
    abr_steps: int = 10
    lr: float = 0.1
    seed: int = 0

@dataclass
class TrainConfigRL:
    outer_iters: int = 400
    abr_steps_per_iter: int = 5
    batch_size: int = 128
    lr: float = 1e-2
    seed: int = 0

def train_circle_static_exact(
    sigma: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[CircleNet, dict]:

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
                loss = circle_loss_exact(net, sigma, i)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        for i in range(n):
            history[i].append(net.get_angle_np(i))

        if (it + 1) % 80 == 0:
            angs = [f"{np.rad2deg(history[i][-1]):.0f}°" for i in range(n)]
            print(f"  [Exact iter {it+1}/{cfg.outer_iters}] angles: {angs}")

    return net, history

def train_circle_adaptive_exact(
    n: int,
    cfg: TrainConfig,
    epoch_len: int = 10,
) -> tuple[CircleNet, dict, list[np.ndarray], list[np.ndarray]]:

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    net = CircleNet(num_slots=n)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    sigma = torch.full((n, n), 1.0 / n)
    history = {i: [] for i in range(n)}
    sigma_hist: list[np.ndarray] = []
    payoff_hist: list[np.ndarray] = []

    for it in range(cfg.outer_iters):
        if it % epoch_len == 0:
            U = eval_payoffs_circle(net, use_means=False)
            sigma_np = psro_nash_meta_graph_solver(U)
            sigma = torch.from_numpy(sigma_np).float()
            sigma_hist.append(sigma_np.copy())
            payoff_hist.append(U.copy())

        reps = unique_rows(sigma)
        for i in reps:
            if sigma[i].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps):
                loss = circle_loss_exact(net, sigma, i)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        for i in range(n):
            history[i].append(net.get_angle_np(i))

        if (it + 1) % 80 == 0:
            print(f"  [Exact Adaptive iter {it+1}/{cfg.outer_iters}]")

    return net, history, sigma_hist, payoff_hist

def train_circle_static_rl(
    sigma: torch.Tensor,
    cfg: TrainConfigRL,
) -> tuple[CircleNetStochastic, dict]:

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
                circle_rl_abr_step(net, sigma, i, cfg.batch_size, opt)

        for i in range(n):
            history[i].append(net.get_angle_np(i))

        if (it + 1) % 80 == 0:
            angs = [f"{np.rad2deg(history[i][-1]):.0f}°" for i in range(n)]
            print(f"  [RL iter {it+1}/{cfg.outer_iters}] angles: {angs}")

    return net, history

def train_circle_adaptive_rl(
    n: int,
    cfg: TrainConfigRL,
    epoch_len: int = 10,
) -> tuple[CircleNetStochastic, dict, list[np.ndarray], list[np.ndarray]]:

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    net = CircleNetStochastic(num_slots=n)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    sigma = torch.full((n, n), 1.0 / n)
    history = {i: [] for i in range(n)}
    sigma_hist = []
    payoff_hist = []

    for it in range(cfg.outer_iters):
        if it % epoch_len == 0:
            U = eval_payoffs_circle(net, use_means=True)
            sigma_np = psro_nash_meta_graph_solver(U)
            sigma = torch.from_numpy(sigma_np).float()
            sigma_hist.append(sigma_np.copy())
            payoff_hist.append(U.copy())

        reps = unique_rows(sigma)
        for i in reps:
            if sigma[i].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps_per_iter):
                circle_rl_abr_step(net, sigma, i, cfg.batch_size, opt)

        for i in range(n):
            history[i].append(net.get_angle_np(i))

        if (it + 1) % 80 == 0:
            print(f"  [RL Adaptive iter {it+1}/{cfg.outer_iters}]")

    return net, history, sigma_hist, payoff_hist

def angle_to_xy(theta: float) -> tuple[float, float]:

    return np.cos(theta), np.sin(theta)

def draw_circle(ax: plt.Axes, title: Optional[str] = None, show_time_legend: bool = False):
    th = np.linspace(0, TWOPI, 100)
    ax.plot(np.cos(th), np.sin(th), "k-", linewidth=1.5)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=13)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    if show_time_legend:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Time (red=start → green=end)", fontsize=10)

def scatter_trajectory_circle(
    ax: plt.Axes,
    angles_hist: list[float],
    alpha: float = 0.85,
    s: int = 45,
    line_alpha: float = 0.35,
    linewidth: float = 2.5,
):

    xs = np.array([np.cos(a) for a in angles_hist])
    ys = np.array([np.sin(a) for a in angles_hist])
    t = len(angles_hist)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, t))

    ax.plot(xs, ys, color="gray", alpha=line_alpha, linewidth=linewidth, zorder=0)
    ax.scatter(xs, ys, c=colors, s=s, alpha=alpha, edgecolors="k", linewidths=0.8, zorder=1)

def plot_circle_single(history: dict, slot: int = 0, title: str = "", save: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    draw_circle(ax, title, show_time_legend=True)
    scatter_trajectory_circle(ax, history[slot], s=50, linewidth=3)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)

def plot_circle_population(
    history: dict,
    indices: Optional[list[int]] = None,
    title: str = "",
    save: Optional[str] = None,
):
    if indices is None:
        indices = sorted(history.keys())
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    draw_circle(ax, title, show_time_legend=True)
    for i in indices:
        scatter_trajectory_circle(ax, history[i], alpha=0.7, s=32, line_alpha=0.25, linewidth=2)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)

def plot_circle_combined(
    hist_self: dict,
    hist_cycle: dict,
    hist_fp: dict,
    hist_adaptive: dict,
    save: Optional[str] = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    draw_circle(axes[0, 0], "Self-Play (2 agents)", show_time_legend=True)
    scatter_trajectory_circle(axes[0, 0], hist_self[0], alpha=0.85, s=40)
    scatter_trajectory_circle(axes[0, 0], hist_self[1], alpha=0.85, s=40)
    draw_circle(axes[0, 1], "Strategic Cycle (3)", show_time_legend=True)
    for i in range(3):
        scatter_trajectory_circle(axes[0, 1], hist_cycle[i], alpha=0.7, s=28, line_alpha=0.3)
    draw_circle(axes[1, 0], "Fictitious Play (6)", show_time_legend=True)
    for i in range(6):
        scatter_trajectory_circle(axes[1, 0], hist_fp[i], alpha=0.65, s=22, line_alpha=0.25)
    draw_circle(axes[1, 1], "Adaptive PSRO-Nash (4)", show_time_legend=True)
    for i in range(4):
        scatter_trajectory_circle(axes[1, 1], hist_adaptive[i], alpha=0.7, s=28, line_alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)

def analyze_trajectory(history: dict, name: str) -> dict:

    out = {"name": name, "final_angles_deg": [], "spacing_deg": None, "notes": []}
    n = len(history)
    final = [float(np.rad2deg(history[i][-1])) for i in range(n)]
    final_sorted = sorted(final)
    out["final_angles_deg"] = [round(f, 1) for f in final]
    if n >= 2:
        spacings = []
        for i in range(n):
            a, b = final_sorted[i], final_sorted[(i + 1) % n]
            d = b - a if b >= a else (360 - a + b)
            spacings.append(float(d))
        out["spacing_deg"] = [round(s, 1) for s in spacings]

        if n == 3:
            out["notes"].append("Cycle: expect ~120° or 90° spacing for BR cycle")
    return out

def print_analysis(hist_exact: dict, hist_rl: dict, variant: str):
    print(f"\n  --- {variant} ---")
    a_ex = analyze_trajectory(hist_exact, f"{variant} (exact)")
    a_rl = analyze_trajectory(hist_rl, f"{variant} (RL)")
    print(f"  Exact final angles (deg): {a_ex['final_angles_deg']}  spacing: {a_ex['spacing_deg']}")
    print(f"  RL   final angles (deg): {a_rl['final_angles_deg']}  spacing: {a_rl['spacing_deg']}")
    for n in a_ex.get("notes", []):
        print(f"    {n}")

def main():
    parser = argparse.ArgumentParser(description="NeuPL Continuous Circle Game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-iters", type=int, default=400)
    parser.add_argument("--save-dir", type=str, default="results_circle")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["self_play", "cycle", "fp", "adaptive"],
        choices=["self_play", "cycle", "fp", "adaptive"],
    )
    parser.add_argument("--abr", type=str, default="both", choices=["exact", "rl", "both"])
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    cfg = TrainConfig(outer_iters=args.outer_iters, seed=args.seed)
    cfg_rl = TrainConfigRL(outer_iters=args.outer_iters, seed=args.seed)
    results_exact = {}
    results_rl = {}

    if "self_play" in args.experiments:
        print("\n" + "=" * 60)
        print("  Circle: Self-Play (N=2)")
        print("=" * 60)
        sigma = sigma_self_play(2)
        if args.abr in ("exact", "both"):
            _, hist_e = train_circle_static_exact(sigma, cfg)
            results_exact["self_play"] = hist_e
            plot_circle_population(hist_e, title="Circle Self-Play (Exact)", save=f"{args.save_dir}/circle_self_play_exact.png")
        if args.abr in ("rl", "both"):
            _, hist_r = train_circle_static_rl(sigma, cfg_rl)
            results_rl["self_play"] = hist_r
            plot_circle_population(hist_r, title="Circle Self-Play (RL)", save=f"{args.save_dir}/circle_self_play_rl.png")
        if args.abr == "both":
            print_analysis(results_exact["self_play"], results_rl["self_play"], "Self-Play")

    if "cycle" in args.experiments:
        print("\n" + "=" * 60)
        print("  Circle: Strategic Cycle (N=3)")
        print("=" * 60)
        sigma = sigma_strategic_cycle(3)
        if args.abr in ("exact", "both"):
            _, hist_e = train_circle_static_exact(sigma, cfg)
            results_exact["cycle"] = hist_e
            plot_circle_population(hist_e, title="Circle Strategic Cycle (Exact)", save=f"{args.save_dir}/circle_cycle_exact.png")
        if args.abr in ("rl", "both"):
            _, hist_r = train_circle_static_rl(sigma, cfg_rl)
            results_rl["cycle"] = hist_r
            plot_circle_population(hist_r, title="Circle Strategic Cycle (RL)", save=f"{args.save_dir}/circle_cycle_rl.png")
        if args.abr == "both":
            print_analysis(results_exact["cycle"], results_rl["cycle"], "Strategic Cycle")

    if "fp" in args.experiments:
        print("\n" + "=" * 60)
        print("  Circle: Fictitious Play (N=6)")
        print("=" * 60)
        sigma = sigma_fictitious_play(6)
        if args.abr in ("exact", "both"):
            _, hist_e = train_circle_static_exact(sigma, cfg)
            results_exact["fp"] = hist_e
            plot_circle_population(hist_e, title="Circle Fictitious Play (Exact)", save=f"{args.save_dir}/circle_fp_exact.png")
        if args.abr in ("rl", "both"):
            _, hist_r = train_circle_static_rl(sigma, cfg_rl)
            results_rl["fp"] = hist_r
            plot_circle_population(hist_r, title="Circle Fictitious Play (RL)", save=f"{args.save_dir}/circle_fp_rl.png")
        if args.abr == "both":
            print_analysis(results_exact["fp"], results_rl["fp"], "Fictitious Play")

    if "adaptive" in args.experiments:
        print("\n" + "=" * 60)
        print("  Circle: Adaptive PSRO-Nash (N=4)")
        print("=" * 60)
        if args.abr in ("exact", "both"):
            _, hist_e, _, _ = train_circle_adaptive_exact(4, cfg, epoch_len=10)
            results_exact["adaptive"] = hist_e
            plot_circle_population(hist_e, title="Circle Adaptive (Exact)", save=f"{args.save_dir}/circle_adaptive_exact.png")
        if args.abr in ("rl", "both"):
            _, hist_r, _, _ = train_circle_adaptive_rl(4, cfg_rl, epoch_len=10)
            results_rl["adaptive"] = hist_r
            plot_circle_population(hist_r, title="Circle Adaptive (RL)", save=f"{args.save_dir}/circle_adaptive_rl.png")
        if args.abr == "both":
            print_analysis(results_exact["adaptive"], results_rl["adaptive"], "Adaptive")

    if args.abr == "both" and all(k in results_exact and k in results_rl for k in ["self_play", "cycle", "fp", "adaptive"]):
        plot_circle_combined(
            results_exact["self_play"],
            results_exact["cycle"],
            results_exact["fp"],
            results_exact["adaptive"],
            save=f"{args.save_dir}/circle_combined_exact.png",
        )
        plot_circle_combined(
            results_rl["self_play"],
            results_rl["cycle"],
            results_rl["fp"],
            results_rl["adaptive"],
            save=f"{args.save_dir}/circle_combined_rl.png",
        )

    print("\n" + "=" * 60)
    print("  CONVERGENCE ANALYSIS (Continuous Circle Game)")
    print("=" * 60)
    print("  Payoff: J(θ₁,θ₂)=sin(θ₁-θ₂). BR to θ is θ+π/2 (90°).")
    print("  - Self-Play (N=2): Agents chase each other (θ₀→θ₁+90°, θ₁→θ₀+90°).")
    print("    Expect: persistent rotation / cycling, no fixed point.")
    print("  - Strategic Cycle (N=3): Each BR to next → stable 90° spacing (or 120°).")
    print("  - Fictitious Play: Chain of BRs; later agents mix toward 'center'.")
    print("  - Adaptive: Graph from Nash over payoffs; mix of cycle + equilibrium.")
    if results_exact and results_rl:
        print("\n  Exact ABR typically converges faster; RL ABR may show more variance.")
    print(f"\n  Plots saved to: {args.save_dir}/")

if __name__ == "__main__":
    main()
