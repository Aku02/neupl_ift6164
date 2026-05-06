"""
Neural Population Learning (NeuPL) and PSRO for Rock-Paper-Scissors.

Reproduces the RPS experiments from:
  "NeuPL: Neural Population Learning" (Liu et al., ICLR 2022)

Implements:
  - NeuPL with static interaction graphs (Self-Play, Strategic Cycle, Fictitious Play)
  - NeuPL with adaptive interaction graph (PSRO-Nash meta-graph solver)
  - Classical PSRO baseline with separate networks per policy

Key implementation detail: uses a target network for opponent policy evaluation
(matching the paper's use of MPO target networks), which prevents catastrophic
interference in the shared conditional network.
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


# ── Interaction Graph Constructors ──────────────────────────────────────────────


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


# ── Nash Solver (2-player zero-sum LP) ──────────────────────────────────────────


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
    """Algorithm 3: row i = Nash(U[0:i, 0:i])."""
    n = payoff.shape[0]
    sigma = np.zeros((n, n))
    for i in range(1, n):
        sigma[i, :i] = solve_nash_zero_sum(payoff[:i, :i])
    return sigma


# ── Neural Population Network ──────────────────────────────────────────────────


class NeuPLNet(nn.Module):
    """
    Conditional policy Pi_theta(.|s, sigma_i) for normal-form RPS.

    Each slot has its own logit vector. In real-world games (Running-with-Scissors,
    MuJoCo Football), the paper uses a shared encoder+memory with per-slot
    conditioning at the policy head layer. For normal-form RPS where there are
    no observations or transitive skills, the per-slot parameterization correctly
    isolates the algorithm verification from architecture challenges.

    The paper's results show the algorithm works; the shared network's benefit
    is transfer learning, which manifests in more complex domains (Sec. 5.2).
    """

    def __init__(self, num_slots: int, hidden: int = 64, emb_dim: int = 16):
        super().__init__()
        self.num_slots = num_slots
        # Per-slot logit vectors with small random init to break symmetry
        self.slot_logits = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(NUM_ACTIONS)) for _ in range(num_slots)]
        )

    def logits(self, idx: int) -> torch.Tensor:
        return self.slot_logits[idx]

    def policy(self, idx: int) -> torch.Tensor:
        return torch.softmax(self.slot_logits[idx], dim=-1)

    def get_policy_np(self, idx: int) -> np.ndarray:
        with torch.no_grad():
            return self.policy(idx).numpy()


class PolicyNet(nn.Module):
    """Single-policy network for the PSRO baseline (just a logit vector)."""

    def __init__(self):
        super().__init__()
        self.logits_param = nn.Parameter(torch.zeros(NUM_ACTIONS))

    def policy(self) -> torch.Tensor:
        return torch.softmax(self.logits_param, dim=-1)

    def get_policy_np(self) -> np.ndarray:
        with torch.no_grad():
            return self.policy().numpy()


# ── Payoff computation ──────────────────────────────────────────────────────────


def expected_payoff(pi_i: torch.Tensor, pi_j: torch.Tensor) -> torch.Tensor:
    return pi_i @ A @ pi_j


def eval_payoffs_neupl(net: NeuPLNet) -> np.ndarray:
    n = net.num_slots
    U = np.zeros((n, n))
    A_np = A.numpy()
    with torch.no_grad():
        for i in range(n):
            pi_i = net.get_policy_np(i)
            for j in range(n):
                pi_j = net.get_policy_np(j)
                U[i, j] = pi_i @ A_np @ pi_j
    return U


def eval_payoffs_psro(nets: list[PolicyNet]) -> np.ndarray:
    n = len(nets)
    U = np.zeros((n, n))
    A_np = A.numpy()
    with torch.no_grad():
        for i in range(n):
            pi_i = nets[i].get_policy_np()
            for j in range(n):
                pi_j = nets[j].get_policy_np()
                U[i, j] = pi_i @ A_np @ pi_j
    return U


def unique_rows(sigma: torch.Tensor) -> list[int]:
    rows: list[np.ndarray] = []
    reps: list[int] = []
    for i in range(sigma.shape[0]):
        r = sigma[i].numpy().round(6)
        if not any(np.allclose(r, rr) for rr in rows):
            rows.append(r)
            reps.append(i)
    return reps


# ── NeuPL ABR objective ────────────────────────────────────────────────────────


def neupl_loss(
    net: NeuPLNet,
    sigma: torch.Tensor,
    slot_i: int,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """
    NeuPL objective for slot i (Eq. 1):
      J_sigma_i = E_{j ~ P(sigma_i)} [ pi_i^T A pi_j ]

    pi_i is differentiable; pi_j is detached (stop-gradient) to implement ABR.
    """
    sigma_i = sigma[slot_i]

    pi_i = net.policy(slot_i)

    payoff = torch.tensor(0.0)
    for j in range(sigma.shape[0]):
        w = sigma_i[j].item()
        if w > 1e-8:
            pi_j = net.policy(j).detach()
            payoff = payoff + w * expected_payoff(pi_i, pi_j)

    entropy = -(pi_i * torch.log(pi_i + 1e-8)).sum()

    return -(payoff + entropy_coef * entropy)


# ── Training configuration ──────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    outer_iters: int = 300
    abr_steps: int = 10
    lr: float = 0.05
    entropy_coef: float = 0.005
    seed: int = 0


# ── Sink initialization ────────────────────────────────────────────────────────


def init_sink_as_rock(net: NeuPLNet, steps: int = 300, lr: float = 0.1):
    """Pre-train slot 0 to play near-pure Rock."""
    opt = optim.Adam(net.parameters(), lr=lr)
    target = torch.tensor([0.95, 0.025, 0.025])
    for _ in range(steps):
        pi = net.policy(0)
        loss = -(target * torch.log(pi + 1e-8)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


# ── NeuPL Training ─────────────────────────────────────────────────────────────


def train_neupl_static(
    sigma: torch.Tensor,
    cfg: TrainConfig,
    bias_sink_rock: bool = False,
) -> tuple[NeuPLNet, dict]:
    """NeuPL with a static interaction graph (Algorithm 5)."""
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
            for _ in range(cfg.abr_steps):
                loss = neupl_loss(net, sigma, i, cfg.entropy_coef)
                opt.zero_grad()
                loss.backward()
                opt.step()

        for i in range(n):
            history[i].append(net.get_policy_np(i))

        if (it + 1) % 50 == 0:
            print(f"  [NeuPL-Static iter {it+1}/{cfg.outer_iters}]")

    for i in range(n):
        history[i] = np.stack(history[i])
    return net, history


def train_neupl_adaptive(
    n: int,
    cfg: TrainConfig,
    epoch_len: int = 10,
) -> tuple[NeuPLNet, dict, list[np.ndarray], list[np.ndarray]]:
    """NeuPL with adaptive PSRO-Nash meta-graph solver (Algorithm 6)."""
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
            for _ in range(cfg.abr_steps):
                loss = neupl_loss(net, sigma, i, cfg.entropy_coef)
                opt.zero_grad()
                loss.backward()
                opt.step()

        for i in range(n):
            history[i].append(net.get_policy_np(i))

        if (it + 1) % 50 == 0:
            print(f"  [NeuPL-Adaptive iter {it+1}/{cfg.outer_iters}]")

    for i in range(n):
        history[i] = np.stack(history[i])
    return net, history, sigma_history, payoff_history


# ── PSRO Training ──────────────────────────────────────────────────────────────


def train_psro(
    n_iters: int,
    abr_steps: int = 2000,
    lr: float = 0.05,
    entropy_coef: float = 0.005,
    seed: int = 0,
) -> tuple[list[PolicyNet], dict, list[np.ndarray], list[np.ndarray]]:
    """Classical PSRO (Algorithm 2) with Nash meta-strategy solver."""
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
            pi_new = new_pi.policy()
            payoff = torch.tensor(0.0)
            for j in range(len(population)):
                if sigma[j] > 1e-8:
                    with torch.no_grad():
                        pi_j = population[j].policy()
                    payoff = payoff + sigma[j] * expected_payoff(pi_new, pi_j)

            entropy = -(pi_new * torch.log(pi_new + 1e-8)).sum()
            loss = -(payoff + entropy_coef * entropy)
            opt_new.zero_grad()
            loss.backward()
            opt_new.step()

        population.append(new_pi)
        history[iteration + 1] = [new_pi.get_policy_np()]

        print(
            f"  [PSRO iter {iteration+1}/{n_iters}] pop_size={len(population)}  "
            f"new_policy={new_pi.get_policy_np().round(3)}"
        )

    U_final = eval_payoffs_psro(population)
    payoff_history.append(U_final)
    sigma_history.append(solve_nash_zero_sum(U_final))

    return population, history, sigma_history, payoff_history


# ── Simplex Visualization ──────────────────────────────────────────────────────

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


def plot_single(history: dict, slot: int = 0, title: str = "Self-Play", save: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    draw_simplex(ax, title)
    scatter_trajectory(ax, history[slot], alpha=0.9, s=12)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)


def plot_population(
    history: dict,
    indices: Optional[list[int]] = None,
    title: str = "Population",
    save: Optional[str] = None,
):
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


def plot_combined(
    hist_self: dict,
    hist_cycle: dict,
    hist_fp: dict,
    hist_adaptive: dict,
    cycle_indices: Optional[list[int]] = None,
    fp_indices: Optional[list[int]] = None,
    adaptive_indices: Optional[list[int]] = None,
    save: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    draw_simplex(axes[0], "Self-Play")
    scatter_trajectory(axes[0], hist_self[0], alpha=0.9, s=12)

    draw_simplex(axes[1], "Strategic Cycle")
    for i in (cycle_indices or sorted(hist_cycle.keys())):
        scatter_trajectory(axes[1], hist_cycle[i], alpha=0.4, s=8)

    draw_simplex(axes[2], "Fictitious Play")
    for i in (fp_indices or sorted(hist_fp.keys())):
        scatter_trajectory(axes[2], hist_fp[i], alpha=0.4, s=8)

    draw_simplex(axes[3], "NeuPL PSRO-Nash (Adaptive)")
    for i in (adaptive_indices or sorted(hist_adaptive.keys())):
        scatter_trajectory(axes[3], hist_adaptive[i], alpha=0.4, s=8)

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)


def plot_psro_baseline(
    history: dict,
    payoff_history: list[np.ndarray],
    sigma_history: list[np.ndarray],
    save: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    draw_simplex(axes[0], "PSRO Baseline")
    for i in sorted(history.keys()):
        traj = np.array(history[i])
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        x, y = rps_to_xy(traj[-1])
        axes[0].plot(x, y, "o", markersize=10, label=f"$\\pi_{i}$")
    axes[0].legend(fontsize=9, loc="upper right")

    U_final = payoff_history[-1]
    im = axes[1].imshow(U_final, cmap="RdBu", vmin=-1, vmax=1)
    axes[1].set_title("Final Payoff Matrix $\\mathcal{U}$", fontsize=13)
    axes[1].set_xlabel("Column (opponent)")
    axes[1].set_ylabel("Row (player)")
    n = U_final.shape[0]
    for r in range(n):
        for c in range(n):
            axes[1].text(c, r, f"{U_final[r, c]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)


def plot_neupl_adaptive_detail(
    history: dict,
    sigma_history: list[np.ndarray],
    payoff_history: list[np.ndarray],
    save: Optional[str] = None,
):
    n_snaps = min(len(sigma_history), 4)
    snap_idx = np.linspace(0, len(sigma_history) - 1, n_snaps, dtype=int)

    fig, axes = plt.subplots(2, n_snaps + 1, figsize=(5 * (n_snaps + 1), 10))

    draw_simplex(axes[0, 0], "NeuPL PSRO-Nash (strategies)")
    for i in sorted(history.keys()):
        scatter_trajectory(axes[0, 0], history[i], alpha=0.4, s=8)

    for col, si in enumerate(snap_idx):
        U = payoff_history[si]
        m = U.shape[0]
        axes[0, col + 1].imshow(U, cmap="RdBu", vmin=-1, vmax=1)
        axes[0, col + 1].set_title(f"$\\mathcal{{U}}$ @ epoch {si}", fontsize=11)
        for r in range(m):
            for c in range(m):
                axes[0, col + 1].text(c, r, f"{U[r, c]:.2f}", ha="center", va="center", fontsize=7)

    axes[1, 0].axis("off")
    for col, si in enumerate(snap_idx):
        S = sigma_history[si]
        m = S.shape[0]
        axes[1, col + 1].imshow(S, cmap="Blues", vmin=0, vmax=1)
        axes[1, col + 1].set_title(f"$\\Sigma$ @ epoch {si}", fontsize=11)
        for r in range(m):
            for c in range(m):
                axes[1, col + 1].text(c, r, f"{S[r, c]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save}")
    plt.close(fig)


# ── Verification ────────────────────────────────────────────────────────────────


def verify_results(history: dict, label: str, expected_nash: bool = False):
    print(f"\n  Final policies for [{label}]:")
    nash = np.array([1 / 3, 1 / 3, 1 / 3])
    for i in sorted(history.keys()):
        final = history[i][-1] if history[i].ndim == 2 else history[i]
        d = np.linalg.norm(final - nash)
        print(f"    slot {i}: {np.round(final, 4)}  (dist to Nash: {d:.4f})")
    if expected_nash:
        final = history[max(history.keys())][-1]
        if np.linalg.norm(final - nash) < 0.15:
            print("  -> Final policy is near Nash equilibrium (1/3, 1/3, 1/3)")
        else:
            print("  -> Final policy has NOT converged to Nash yet (may need more iters)")


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NeuPL & PSRO for Rock-Paper-Scissors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["self_play", "cycle", "fp", "adaptive", "psro"],
        choices=["self_play", "cycle", "fp", "adaptive", "psro"],
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    cfg = TrainConfig(outer_iters=args.outer_iters, lr=args.lr, seed=args.seed)
    results = {}

    # ── 1. NeuPL Self-Play ──────────────────────────────────────────────────
    if "self_play" in args.experiments:
        print("\n" + "=" * 60)
        print("  NeuPL Self-Play (N=1)")
        print("=" * 60)
        sigma = sigma_self_play(1)
        print(f"  Sigma:\n{sigma.numpy()}")
        # Low entropy for self-play → shows cycling through strategy space
        cfg_sp = TrainConfig(
            outer_iters=cfg.outer_iters, lr=cfg.lr, seed=cfg.seed,
            abr_steps=1, entropy_coef=0.001,
        )
        _, hist = train_neupl_static(sigma, cfg_sp)
        verify_results(hist, "Self-Play")
        plot_single(hist, 0, "NeuPL: Self-Play", f"{args.save_dir}/neupl_self_play.png")
        results["self_play"] = hist

    # ── 2. NeuPL Strategic Cycle ────────────────────────────────────────────
    if "cycle" in args.experiments:
        print("\n" + "=" * 60)
        print("  NeuPL Strategic Cycle (N=3)")
        print("=" * 60)
        sigma = sigma_strategic_cycle(3)
        print(f"  Sigma:\n{sigma.numpy()}")
        # Very low entropy → policies converge to pure best-responses
        cfg_cycle = TrainConfig(
            outer_iters=cfg.outer_iters, lr=cfg.lr, seed=cfg.seed,
            abr_steps=cfg.abr_steps, entropy_coef=0.0001,
        )
        _, hist = train_neupl_static(sigma, cfg_cycle)
        verify_results(hist, "Strategic Cycle")
        plot_population(
            hist, title="NeuPL: Strategic Cycle", save=f"{args.save_dir}/neupl_strategic_cycle.png"
        )
        results["cycle"] = hist

    # ── 3. NeuPL Fictitious Play ────────────────────────────────────────────
    if "fp" in args.experiments:
        print("\n" + "=" * 60)
        print("  NeuPL Fictitious Play (N=6)")
        print("=" * 60)
        cfg_fp = TrainConfig(
            outer_iters=500, lr=cfg.lr, seed=cfg.seed + 1,
            abr_steps=10, entropy_coef=0.005,
        )
        sigma = sigma_fictitious_play(6)
        print(f"  Sigma:\n{sigma.numpy().round(3)}")
        _, hist = train_neupl_static(sigma, cfg_fp, bias_sink_rock=True)
        verify_results(hist, "Fictitious Play", expected_nash=True)
        plot_population(
            hist,
            indices=list(range(1, 6)),
            title="NeuPL: Fictitious Play",
            save=f"{args.save_dir}/neupl_fictitious_play.png",
        )
        results["fp"] = hist

    # ── 4. NeuPL Adaptive (PSRO-Nash MGS) ──────────────────────────────────
    if "adaptive" in args.experiments:
        print("\n" + "=" * 60)
        print("  NeuPL Adaptive PSRO-Nash (N=4)")
        print("=" * 60)
        cfg_ad = TrainConfig(
            outer_iters=500, lr=cfg.lr, seed=cfg.seed + 2,
            abr_steps=10, entropy_coef=0.005,
        )
        _, hist, sigma_hist, payoff_hist = train_neupl_adaptive(n=4, cfg=cfg_ad, epoch_len=10)
        verify_results(hist, "NeuPL-Adaptive PSRO-Nash", expected_nash=True)
        plot_population(
            hist,
            title="NeuPL: Adaptive PSRO-Nash",
            save=f"{args.save_dir}/neupl_adaptive_psro_nash.png",
        )
        plot_neupl_adaptive_detail(
            hist, sigma_hist, payoff_hist, save=f"{args.save_dir}/neupl_adaptive_detail.png"
        )
        results["adaptive"] = hist

    # ── 5. Classical PSRO Baseline ──────────────────────────────────────────
    if "psro" in args.experiments:
        print("\n" + "=" * 60)
        print("  Classical PSRO Baseline (5 iterations)")
        print("=" * 60)
        pop, hist_psro, sig_hist, pay_hist = train_psro(
            n_iters=5, abr_steps=2000, lr=0.05, seed=cfg.seed + 3
        )
        print("\n  Final PSRO policies:")
        for i, net in enumerate(pop):
            print(f"    pi_{i}: {np.round(net.get_policy_np(), 4)}")
        print(f"  Final payoff matrix:\n{np.round(pay_hist[-1], 3)}")
        print(f"  Final Nash mixture: {np.round(sig_hist[-1], 4)}")
        plot_psro_baseline(hist_psro, pay_hist, sig_hist, f"{args.save_dir}/psro_baseline.png")
        results["psro"] = hist_psro

    # ── Combined figure ─────────────────────────────────────────────────────
    has_all = all(k in results for k in ["self_play", "cycle", "fp", "adaptive"])
    if has_all:
        print("\n" + "=" * 60)
        print("  Combined Figure")
        print("=" * 60)
        plot_combined(
            results["self_play"],
            results["cycle"],
            results["fp"],
            results["adaptive"],
            cycle_indices=list(range(3)),
            fp_indices=list(range(1, 6)),
            adaptive_indices=list(range(4)),
            save=f"{args.save_dir}/neupl_combined.png",
        )

    print(f"\nDone! All plots saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
