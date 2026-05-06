"""
Neural Population Learning (NeuPL) for Rock-Paper-Scissors — RL-based ABR.

This implements NeuPL using the RL-style Approximate Best Response from
Section 3.4 of "NeuPL: Neural Population Learning" (Liu et al., ICLR 2022).

Instead of computing exact analytical gradients (pi_i^T A pi_j), this version
samples actions from both players and uses REINFORCE with a learned value
baseline — matching the paper's use of RL (MPO) as the ABR operator.

    J_sigma_i = E_{j ~ P(sigma_i)} [
        E_{a ~ Pi(.|sigma_i), a' ~ Pi(.|sigma_j)} [ sum_t r_t gamma^t ]
    ]

Compare with neupl_rps.py which uses exact analytical gradients.
"""

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

# ── RPS Payoff Matrix ───────────────────────────────────────────────────────────

A = torch.tensor(
    [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=torch.float32
)
NUM_ACTIONS = 3


# ── Interaction Graphs ──────────────────────────────────────────────────────────


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


# ── Nash Solver ─────────────────────────────────────────────────────────────────


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


# ── Neural Population Network ──────────────────────────────────────────────────


class NeuPLNet(nn.Module):
    """
    Conditional policy Pi_theta(.|s, sigma_i) with per-slot parameters.

    Each slot has:
      - A logit vector (policy head) parameterizing the action distribution
      - A scalar value baseline for variance reduction in REINFORCE

    In stochastic games (Running-with-Scissors, MuJoCo Football), these would
    be shared encoder+memory networks. For normal-form RPS the per-slot
    parameterization isolates the algorithm from architecture challenges.
    """

    def __init__(self, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        self.slot_logits = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(NUM_ACTIONS)) for _ in range(num_slots)]
        )
        self.slot_values = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_slots)]
        )

    def policy_dist(self, idx: int) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.slot_logits[idx])

    def policy(self, idx: int) -> torch.Tensor:
        return torch.softmax(self.slot_logits[idx], dim=-1)

    def value(self, idx: int) -> torch.Tensor:
        return self.slot_values[idx].squeeze()

    def get_policy_np(self, idx: int) -> np.ndarray:
        with torch.no_grad():
            return self.policy(idx).numpy()


class PolicyNet(nn.Module):
    """Single-policy network for the PSRO baseline."""

    def __init__(self):
        super().__init__()
        self.logits_param = nn.Parameter(torch.zeros(NUM_ACTIONS))
        self.value_param = nn.Parameter(torch.zeros(1))

    def policy_dist(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits_param)

    def policy(self) -> torch.Tensor:
        return torch.softmax(self.logits_param, dim=-1)

    def value(self) -> torch.Tensor:
        return self.value_param.squeeze()

    def get_policy_np(self) -> np.ndarray:
        with torch.no_grad():
            return self.policy().numpy()


# ── RL-Based ABR (REINFORCE with value baseline) ───────────────────────────────


def rl_abr_step_neupl(
    net: NeuPLNet,
    sigma: torch.Tensor,
    slot_i: int,
    batch_size: int,
    opt: optim.Optimizer,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
):
    """
    One RL-based ABR gradient step for NeuPL slot i.

    Implements the paper's Eq. 1 via REINFORCE:
      1. Sample opponent j ~ P(sigma_i)
      2. Sample actions a_i ~ Pi(.|sigma_i),  a_j ~ Pi(.|sigma_j)
      3. Get reward r = A[a_i, a_j]
      4. REINFORCE update with value baseline for variance reduction

    This matches the paper's use of RL (MPO) as the ABR operator,
    where actions are sampled and rewards are stochastic.
    """
    sigma_i = sigma[slot_i]
    if sigma_i.sum().item() < 1e-8:
        return

    opp_dist = torch.distributions.Categorical(probs=sigma_i / sigma_i.sum())

    # Collect a batch of episodes
    log_probs = []
    rewards = []
    entropies = []

    dist_i = net.policy_dist(slot_i)
    v_i = net.value(slot_i)

    for _ in range(batch_size):
        # Sample opponent slot
        j = opp_dist.sample().item()

        # Player i samples action
        a_i = dist_i.sample()
        logp_i = dist_i.log_prob(a_i)

        # Opponent j samples action (no gradient — opponent is "the environment")
        with torch.no_grad():
            dist_j = net.policy_dist(j)
            a_j = dist_j.sample()

        # Reward from the RPS payoff matrix
        r = A[a_i.item(), a_j.item()]

        log_probs.append(logp_i)
        rewards.append(r)

    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    entropy = dist_i.entropy()

    # Advantage: reward - value baseline
    advantage = (rewards - v_i).detach()

    # REINFORCE policy gradient
    policy_loss = -(log_probs * advantage).mean()

    # Value loss (MSE between baseline and actual returns)
    value_loss = (rewards.detach() - v_i).pow(2).mean()

    # Entropy bonus for exploration
    entropy_loss = -entropy

    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def rl_abr_step_psro(
    policy_net: PolicyNet,
    opponent_nets: list[PolicyNet],
    sigma: np.ndarray,
    batch_size: int,
    opt: optim.Optimizer,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
):
    """RL-based ABR for classical PSRO (separate networks)."""
    if sigma.sum() < 1e-8:
        return

    opp_dist = torch.distributions.Categorical(
        probs=torch.from_numpy(sigma / sigma.sum()).float()
    )

    dist_i = policy_net.policy_dist()
    v_i = policy_net.value()

    log_probs = []
    rewards = []

    for _ in range(batch_size):
        j = opp_dist.sample().item()
        a_i = dist_i.sample()
        logp_i = dist_i.log_prob(a_i)

        with torch.no_grad():
            dist_j = opponent_nets[j].policy_dist()
            a_j = dist_j.sample()

        r = A[a_i.item(), a_j.item()]
        log_probs.append(logp_i)
        rewards.append(r)

    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    entropy = dist_i.entropy()

    advantage = (rewards - v_i).detach()
    policy_loss = -(log_probs * advantage).mean()
    value_loss = (rewards.detach() - v_i).pow(2).mean()
    entropy_loss = -entropy

    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


# ── Payoff evaluation (by sampling, not analytical) ─────────────────────────────


def eval_payoffs_neupl(net: NeuPLNet, num_games: int = 2048) -> np.ndarray:
    """Estimate payoff matrix by playing games (RL-style evaluation)."""
    n = net.num_slots
    U = np.zeros((n, n))
    counts = np.zeros((n, n))
    with torch.no_grad():
        for _ in range(num_games):
            for i in range(n):
                for j in range(n):
                    a_i = net.policy_dist(i).sample().item()
                    a_j = net.policy_dist(j).sample().item()
                    U[i, j] += A[a_i, a_j].item()
                    counts[i, j] += 1
    return U / np.maximum(counts, 1)


def eval_payoffs_psro(nets: list[PolicyNet], num_games: int = 2048) -> np.ndarray:
    n = len(nets)
    U = np.zeros((n, n))
    with torch.no_grad():
        for _ in range(num_games):
            for i in range(n):
                for j in range(n):
                    a_i = nets[i].policy_dist().sample().item()
                    a_j = nets[j].policy_dist().sample().item()
                    U[i, j] += A[a_i, a_j].item()
    return U / num_games


def unique_rows(sigma: torch.Tensor) -> list[int]:
    rows: list[np.ndarray] = []
    reps: list[int] = []
    for i in range(sigma.shape[0]):
        r = sigma[i].numpy().round(6)
        if not any(np.allclose(r, rr) for rr in rows):
            rows.append(r)
            reps.append(i)
    return reps


# ── Training Configuration ─────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    outer_iters: int = 300
    abr_steps_per_iter: int = 5
    episodes_per_step: int = 256
    lr: float = 3e-3
    entropy_coef: float = 0.01
    seed: int = 0


# ── Sink Initialization ────────────────────────────────────────────────────────


def init_sink_as_rock(net: NeuPLNet):
    """Bias slot 0 toward pure Rock by setting logits directly."""
    with torch.no_grad():
        net.slot_logits[0].copy_(torch.tensor([3.0, -1.0, -1.0]))


# ── NeuPL Training (RL-based ABR) ──────────────────────────────────────────────


def train_neupl_static(
    sigma: torch.Tensor,
    cfg: TrainConfig,
    bias_sink_rock: bool = False,
) -> tuple[NeuPLNet, dict]:
    """
    NeuPL with a static interaction graph (Algorithm 5) using RL-based ABR.

    Each ABR step samples `episodes_per_step` games and uses REINFORCE
    with a value baseline to update the policy.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    n = sigma.shape[0]
    net = NeuPLNet(num_slots=n)

    if bias_sink_rock:
        init_sink_as_rock(net)
        print(f"  Sink (slot 0) biased to: {net.get_policy_np(0).round(3)}")

    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    history = {i: [] for i in range(n)}

    for it in range(cfg.outer_iters):
        reps = unique_rows(sigma)
        for i in reps:
            if sigma[i].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps_per_iter):
                rl_abr_step_neupl(
                    net, sigma, i, cfg.episodes_per_step, opt, cfg.entropy_coef
                )

        for i in range(n):
            history[i].append(net.get_policy_np(i))

        if (it + 1) % 50 == 0:
            policies_str = "  ".join(
                f"slot{i}={net.get_policy_np(i).round(2)}" for i in range(n)
            )
            print(f"  [iter {it+1}/{cfg.outer_iters}] {policies_str}")

    for i in range(n):
        history[i] = np.stack(history[i])
    return net, history


def train_neupl_adaptive(
    n: int,
    cfg: TrainConfig,
    epoch_len: int = 10,
) -> tuple[NeuPLNet, dict, list[np.ndarray], list[np.ndarray]]:
    """NeuPL with adaptive PSRO-Nash meta-graph solver (Algorithm 6), RL-based ABR."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    net = NeuPLNet(num_slots=n)
    init_sink_as_rock(net)
    print(f"  Sink (slot 0) biased to: {net.get_policy_np(0).round(3)}")

    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    sigma = torch.full((n, n), 1.0 / n)
    history = {i: [] for i in range(n)}
    sigma_history: list[np.ndarray] = []
    payoff_history: list[np.ndarray] = []

    for it in range(cfg.outer_iters):
        if it % epoch_len == 0:
            U = eval_payoffs_neupl(net)
            sigma_np = psro_nash_meta_graph_solver(U)
            sigma = torch.from_numpy(sigma_np).float()
            sigma_history.append(sigma_np.copy())
            payoff_history.append(U.copy())

        reps = unique_rows(sigma)
        for i in reps:
            if sigma[i].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps_per_iter):
                rl_abr_step_neupl(
                    net, sigma, i, cfg.episodes_per_step, opt, cfg.entropy_coef
                )

        for i in range(n):
            history[i].append(net.get_policy_np(i))

        if (it + 1) % 50 == 0:
            print(f"  [iter {it+1}/{cfg.outer_iters}]")

    for i in range(n):
        history[i] = np.stack(history[i])
    return net, history, sigma_history, payoff_history


def train_psro(
    n_iters: int,
    abr_steps: int = 1500,
    episodes_per_step: int = 256,
    lr: float = 3e-3,
    entropy_coef: float = 0.01,
    seed: int = 0,
) -> tuple[list[PolicyNet], dict, list[np.ndarray], list[np.ndarray]]:
    """Classical PSRO (Algorithm 2) with RL-based ABR."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    pi0 = PolicyNet()
    with torch.no_grad():
        pi0.logits_param.copy_(torch.tensor([3.0, -1.0, -1.0]))
    print(f"  Initial policy (pi_0): {pi0.get_policy_np().round(3)}")

    population: list[PolicyNet] = [pi0]
    history: dict = {0: [pi0.get_policy_np()]}
    sigma_history: list[np.ndarray] = []
    payoff_history: list[np.ndarray] = []

    for iteration in range(n_iters):
        U = eval_payoffs_psro(population)
        payoff_history.append(U.copy())

        nash_mix = solve_nash_zero_sum(U)
        sigma = np.zeros(len(population))
        sigma[: len(nash_mix)] = nash_mix
        sigma_history.append(sigma.copy())

        new_pi = PolicyNet()
        opt_new = optim.Adam(new_pi.parameters(), lr=lr)

        for _ in range(abr_steps):
            rl_abr_step_psro(
                new_pi, population, sigma, episodes_per_step, opt_new, entropy_coef
            )

        population.append(new_pi)
        history[iteration + 1] = [new_pi.get_policy_np()]

        print(
            f"  [PSRO iter {iteration+1}/{n_iters}] pop={len(population)}  "
            f"new_policy={new_pi.get_policy_np().round(3)}"
        )

    U_final = eval_payoffs_psro(population)
    payoff_history.append(U_final)
    sigma_history.append(solve_nash_zero_sum(U_final))
    return population, history, sigma_history, payoff_history


# ── Visualization (shared with neupl_rps.py) ────────────────────────────────────

S_VERT = np.array([0.0, 0.0])
P_VERT = np.array([1.0, 0.0])
R_VERT = np.array([0.5, np.sqrt(3) / 2])


def rps_to_xy(p: np.ndarray) -> tuple[float, float]:
    xy = float(p[2]) * S_VERT + float(p[1]) * P_VERT + float(p[0]) * R_VERT
    return float(xy[0]), float(xy[1])


def draw_simplex(ax: plt.Axes, title: Optional[str] = None):
    tri = np.vstack([S_VERT, P_VERT, R_VERT, S_VERT])
    ax.plot(tri[:, 0], tri[:, 1], "k-", linewidth=2)
    off = 0.04
    ax.text(S_VERT[0] - off, S_VERT[1] - off, "S", fontsize=12, fontweight="bold")
    ax.text(P_VERT[0] + 0.01, P_VERT[1] - off, "P", fontsize=12, fontweight="bold")
    ax.text(R_VERT[0] - 0.01, R_VERT[1] + 0.02, "R", fontsize=12, fontweight="bold")
    cx, cy = rps_to_xy(np.array([1 / 3, 1 / 3, 1 / 3]))
    ax.plot(cx, cy, "k+", markersize=10, markeredgewidth=2, zorder=10)
    if title:
        ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, R_VERT[1] + 0.08)


def scatter_trajectory(ax: plt.Axes, traj: np.ndarray, alpha: float = 0.8, s: int = 10):
    t = traj.shape[0]
    xs, ys = zip(*[rps_to_xy(traj[k]) for k in range(t)])
    colors = plt.cm.RdYlGn(np.linspace(0, 1, t))
    ax.scatter(xs, ys, c=colors, s=s, alpha=alpha, edgecolors="none")


def plot_single(history: dict, slot: int = 0, title: str = "", save: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    draw_simplex(ax, title)
    scatter_trajectory(ax, history[slot], alpha=0.9, s=12)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)


def plot_population(history: dict, indices=None, title: str = "", save: Optional[str] = None):
    if indices is None:
        indices = sorted(history.keys())
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    draw_simplex(ax, title)
    for i in indices:
        scatter_trajectory(ax, history[i], alpha=0.35, s=8)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)


def plot_combined(hists: dict[str, dict], save: Optional[str] = None):
    n = len(hists)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (title, hist) in zip(axes, hists.items()):
        draw_simplex(ax, title)
        for i in sorted(hist.keys()):
            scatter_trajectory(ax, hist[i], alpha=0.4, s=8)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)


def verify_results(history: dict, label: str):
    print(f"\n  Final policies for [{label}]:")
    nash = np.array([1 / 3, 1 / 3, 1 / 3])
    for i in sorted(history.keys()):
        final = history[i][-1] if history[i].ndim == 2 else history[i]
        d = np.linalg.norm(final - nash)
        print(f"    slot {i}: {np.round(final, 4)}  (dist to Nash: {d:.4f})")


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NeuPL (RL-based ABR) for RPS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-iters", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="results_rl")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["self_play", "cycle", "fp", "adaptive", "psro"],
        choices=["self_play", "cycle", "fp", "adaptive", "psro"],
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    cfg = TrainConfig(
        outer_iters=args.outer_iters,
        episodes_per_step=args.episodes,
        seed=args.seed,
    )
    results = {}

    # ── 1. Self-Play ────────────────────────────────────────────────────────
    if "self_play" in args.experiments:
        print("\n" + "=" * 60)
        print("  NeuPL Self-Play (RL-ABR, N=1)")
        print("=" * 60)
        sigma = sigma_self_play(1)
        _, hist = train_neupl_static(sigma, cfg)
        verify_results(hist, "Self-Play (RL)")
        plot_single(hist, 0, "Self-Play (RL-ABR)", f"{args.save_dir}/neupl_self_play_rl.png")
        results["Self-Play\n(RL-ABR)"] = hist

    # ── 2. Strategic Cycle ──────────────────────────────────────────────────
    if "cycle" in args.experiments:
        print("\n" + "=" * 60)
        print("  NeuPL Strategic Cycle (RL-ABR, N=3)")
        print("=" * 60)
        sigma = sigma_strategic_cycle(3)
        _, hist = train_neupl_static(sigma, cfg)
        verify_results(hist, "Strategic Cycle (RL)")
        plot_population(
            hist, title="Strategic Cycle (RL-ABR)",
            save=f"{args.save_dir}/neupl_cycle_rl.png",
        )
        results["Strategic Cycle\n(RL-ABR)"] = hist

    # ── 3. Fictitious Play ──────────────────────────────────────────────────
    if "fp" in args.experiments:
        print("\n" + "=" * 60)
        print("  NeuPL Fictitious Play (RL-ABR, N=6)")
        print("=" * 60)
        cfg_fp = TrainConfig(
            outer_iters=400, episodes_per_step=cfg.episodes_per_step,
            seed=cfg.seed + 1, abr_steps_per_iter=5, entropy_coef=0.01,
        )
        sigma = sigma_fictitious_play(6)
        _, hist = train_neupl_static(sigma, cfg_fp, bias_sink_rock=True)
        verify_results(hist, "Fictitious Play (RL)")
        plot_population(
            hist, indices=list(range(1, 6)),
            title="Fictitious Play (RL-ABR)",
            save=f"{args.save_dir}/neupl_fp_rl.png",
        )
        results["Fictitious Play\n(RL-ABR)"] = {i: hist[i] for i in range(1, 6)}

    # ── 4. Adaptive PSRO-Nash ───────────────────────────────────────────────
    if "adaptive" in args.experiments:
        print("\n" + "=" * 60)
        print("  NeuPL Adaptive PSRO-Nash (RL-ABR, N=4)")
        print("=" * 60)
        cfg_ad = TrainConfig(
            outer_iters=500, episodes_per_step=cfg.episodes_per_step,
            seed=cfg.seed + 2, abr_steps_per_iter=5, entropy_coef=0.01,
        )
        _, hist, _, _ = train_neupl_adaptive(n=4, cfg=cfg_ad, epoch_len=10)
        verify_results(hist, "Adaptive PSRO-Nash (RL)")
        plot_population(
            hist, title="PSRO-Nash Adaptive (RL-ABR)",
            save=f"{args.save_dir}/neupl_adaptive_rl.png",
        )
        results["PSRO-Nash\n(RL-ABR)"] = hist

    # ── 5. PSRO Baseline ───────────────────────────────────────────────────
    if "psro" in args.experiments:
        print("\n" + "=" * 60)
        print("  Classical PSRO Baseline (RL-ABR, 5 iterations)")
        print("=" * 60)
        pop, hist_psro, _, pay_hist = train_psro(
            n_iters=5, abr_steps=1500, episodes_per_step=256,
            lr=3e-3, seed=cfg.seed + 3,
        )
        print("\n  Final PSRO policies:")
        for i, net in enumerate(pop):
            print(f"    pi_{i}: {np.round(net.get_policy_np(), 4)}")
        print(f"  Final payoff matrix:\n{np.round(pay_hist[-1], 3)}")

    # ── Combined plot ───────────────────────────────────────────────────────
    if results:
        plot_combined(results, save=f"{args.save_dir}/neupl_rl_combined.png")

    print(f"\nDone! All plots saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
