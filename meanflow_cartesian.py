

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import trapezoid as _trapz1d
from scipy.optimize import linprog
from scipy.signal import find_peaks

matplotlib.use("Agg")

@dataclass
class GameConfig:
    xmin: float = -3.0
    xmax: float = 3.0
    ymin: float = -3.0
    ymax: float = 3.0
    num_bins: int = 41

@dataclass
class TrainConfig:
    outer_iters: int = 300
    abr_steps: int = 15
    lr: float = 0.03
    seed: int = 0
    epoch_len: int = 10
    flow_steps: int = 10
    support_size: int = 7
    hidden: int = 64
    t_embed_dim: int = 8
    noise_scale: float = 1.0
    policy_output_scale: float = 1.0
    consistency_coef: float = 0.1

def cartesian_payoff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sin(3.0 * x) * torch.cos(3.0 * y)

def build_payoff_grid(game_cfg: GameConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_grid = torch.linspace(game_cfg.xmin, game_cfg.xmax, game_cfg.num_bins)
    y_grid = torch.linspace(game_cfg.ymin, game_cfg.ymax, game_cfg.num_bins)
    payoff_grid = cartesian_payoff(x_grid[:, None], y_grid[None, :])
    return x_grid, y_grid, payoff_grid

def bounded_scalar(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)

def solve_row_nash_zero_sum(payoff: np.ndarray) -> np.ndarray:
    m, n = payoff.shape
    if m == 1:
        return np.array([1.0])

    c = np.zeros(m + 1, dtype=np.float64)
    c[-1] = -1.0

    a_ub = np.zeros((n, m + 1), dtype=np.float64)
    a_ub[:, :m] = -payoff.T
    a_ub[:, -1] = 1.0
    b_ub = np.zeros(n, dtype=np.float64)

    a_eq = np.zeros((1, m + 1), dtype=np.float64)
    a_eq[0, :m] = 1.0
    b_eq = np.array([1.0], dtype=np.float64)

    bounds = [(0.0, None)] * m + [(None, None)]
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if result.success:
        mix = np.maximum(result.x[:m], 0.0)
        total = mix.sum()
        if total > 0:
            return mix / total
    return np.ones(m) / m

def solve_col_nash_zero_sum(payoff: np.ndarray) -> np.ndarray:
    return solve_row_nash_zero_sum(-payoff.T)

def discretized_nash(
    game_cfg: GameConfig,
    resolution: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_grid = np.linspace(game_cfg.xmin, game_cfg.xmax, resolution)
    y_grid = np.linspace(game_cfg.ymin, game_cfg.ymax, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    payoff = np.sin(3.0 * xx) * np.cos(3.0 * yy)
    row_mix = solve_row_nash_zero_sum(payoff)
    col_mix = solve_col_nash_zero_sum(payoff)
    value = float(row_mix @ payoff @ col_mix)
    return x_grid, row_mix, col_mix, value

def psro_nash_meta_graph_solver(payoff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(payoff.shape[0], payoff.shape[1])
    row_sigma = np.zeros((payoff.shape[0], payoff.shape[1]))
    col_sigma = np.zeros((payoff.shape[1], payoff.shape[0]))
    for i in range(1, n):
        subgame = payoff[:i, :i]
        row_sigma[i, :i] = solve_row_nash_zero_sum(subgame)
        col_sigma[i, :i] = solve_col_nash_zero_sum(subgame)
    return row_sigma, col_sigma

def sigma_self_play(n: int) -> np.ndarray:
    return np.full((n, n), 1.0 / n, dtype=np.float32)

def sigma_strategic_cycle(n: int) -> np.ndarray:
    sigma = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sigma[i, (i + 1) % n] = 1.0
    return sigma

def sigma_fictitious_play(n: int) -> np.ndarray:
    sigma = np.zeros((n, n), dtype=np.float32)
    for i in range(1, n):
        sigma[i, :i] = 1.0 / float(i)
    return sigma

def unique_rows(sigma: torch.Tensor) -> list[int]:
    rows: list[np.ndarray] = []
    reps: list[int] = []
    for i in range(sigma.shape[0]):
        row = sigma[i].detach().cpu().numpy().round(6)
        if not any(np.allclose(row, prev) for prev in rows):
            rows.append(row)
            reps.append(i)
    return reps

def gaussian_mixture_density(
    support: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    density = np.zeros_like(grid, dtype=np.float64)
    bandwidth = max(bandwidth, 1e-6)
    for point, weight in zip(support, weights):
        density += weight * np.exp(-0.5 * ((grid - point) / bandwidth) ** 2)
    density *= 1.0 / (bandwidth * np.sqrt(2.0 * np.pi))
    total = _trapz1d(density, grid)
    if total > 0:
        density /= total
    return density

def sinusoidal_t_embed(t: torch.Tensor, embed_dim: int) -> torch.Tensor:

    half = embed_dim // 2
    freqs = torch.pow(2.0, torch.arange(half, dtype=torch.float32, device=t.device))
    scaled = t * freqs[None, :]
    return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)

class ScalarGenerator(nn.Module):

    def __init__(self, hidden_dim: int, t_embed_dim: int = 8):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.net = nn.Sequential(
            nn.Linear(1 + t_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        t_emb = sinusoidal_t_embed(t, self.t_embed_dim)
        inp = torch.cat([z, t_emb], dim=-1)
        return self.net(inp)

class BasePopulation(nn.Module):
    def __init__(self, num_slots: int, game_cfg: GameConfig):
        super().__init__()
        self.num_slots = num_slots
        self.game_cfg = game_cfg
        x_grid, y_grid, payoff_grid = build_payoff_grid(game_cfg)
        self.register_buffer("x_grid", x_grid)
        self.register_buffer("y_grid", y_grid)
        self.register_buffer("payoff_grid", payoff_grid)

    def evaluate_payoff_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def row_loss(self, sigma: torch.Tensor, slot_i: int) -> torch.Tensor:
        raise NotImplementedError

    def col_loss(self, sigma: torch.Tensor, slot_j: int) -> torch.Tensor:
        raise NotImplementedError

    def init_anchor(self, x0: float, y0: float):
        raise NotImplementedError

    def expected_row_action(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    def expected_col_action(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    def meta_density(
        self,
        row_weights: np.ndarray,
        col_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def summarize_slot(self, idx: int) -> tuple[float, float, float]:
        raise NotImplementedError

class MeanFlowPopulation(BasePopulation):

    def __init__(self, num_slots: int, game_cfg: GameConfig, cfg: TrainConfig):
        super().__init__(num_slots, game_cfg)
        self.cfg = cfg
        k = cfg.support_size

        self.row_logits = nn.ParameterList(
            [nn.Parameter(0.05 * torch.randn(k)) for _ in range(num_slots)]
        )
        self.col_logits = nn.ParameterList(
            [nn.Parameter(0.05 * torch.randn(k)) for _ in range(num_slots)]
        )
        self.row_generators = nn.ModuleList(
            [ScalarGenerator(cfg.hidden, cfg.t_embed_dim) for _ in range(num_slots)]
        )
        self.col_generators = nn.ModuleList(
            [ScalarGenerator(cfg.hidden, cfg.t_embed_dim) for _ in range(num_slots)]
        )

        row_noise = cfg.noise_scale * torch.randn(num_slots, k)
        col_noise = cfg.noise_scale * torch.randn(num_slots, k)
        self.register_buffer("row_noise_anchors", row_noise)
        self.register_buffer("col_noise_anchors", col_noise)

        flow_t = torch.linspace(1.0 / cfg.flow_steps, 1.0, cfg.flow_steps)
        self.register_buffer("flow_time_grid", flow_t)

    def _generate_atoms(self, idx: int, player: str) -> torch.Tensor:

        if player == "row":
            gen = self.row_generators[idx]
            z = self.row_noise_anchors[idx].clone()
        else:
            gen = self.col_generators[idx]
            z = self.col_noise_anchors[idx].clone()

        z = z.unsqueeze(-1)
        t = torch.ones(z.shape[0], 1, device=z.device)
        raw = gen(z, t) * self.cfg.policy_output_scale
        return raw.squeeze(-1)

    def _slot_support(self, idx: int, player: str) -> tuple[torch.Tensor, torch.Tensor]:
        if player == "row":
            weights = torch.softmax(self.row_logits[idx], dim=-1)
            atoms = bounded_scalar(
                self._generate_atoms(idx, "row"),
                self.game_cfg.xmin, self.game_cfg.xmax,
            )
        else:
            weights = torch.softmax(self.col_logits[idx], dim=-1)
            atoms = bounded_scalar(
                self._generate_atoms(idx, "col"),
                self.game_cfg.ymin, self.game_cfg.ymax,
            )
        return weights, atoms

    def _consistency_loss(self, idx: int, player: str) -> torch.Tensor:

        if player == "row":
            gen = self.row_generators[idx]
            z = self.row_noise_anchors[idx].clone().unsqueeze(-1)
        else:
            gen = self.col_generators[idx]
            z = self.col_noise_anchors[idx].clone().unsqueeze(-1)

        k = z.shape[0]
        perm = torch.randperm(len(self.flow_time_grid), device=z.device)
        t1_val = self.flow_time_grid[perm[0]]
        t2_val = self.flow_time_grid[perm[1 % len(perm)]]

        t1 = torch.full((k, 1), float(t1_val.item()), device=z.device)
        t2 = torch.full((k, 1), float(t2_val.item()), device=z.device)

        z_t1 = t1 * z
        z_t2 = t2 * z

        g1 = gen(z_t1, t1) * self.cfg.policy_output_scale
        g2 = gen(z_t2, t2) * self.cfg.policy_output_scale

        return ((g1 - g2.detach()) ** 2).mean()

    def init_anchor(self, x0: float, y0: float):
        def inv_sigmoid(v: float) -> float:
            v = np.clip(v, 1e-4, 1.0 - 1e-4)
            return float(np.log(v / (1.0 - v)))

        x_scaled = (x0 - self.game_cfg.xmin) / (self.game_cfg.xmax - self.game_cfg.xmin)
        y_scaled = (y0 - self.game_cfg.ymin) / (self.game_cfg.ymax - self.game_cfg.ymin)
        row_center = inv_sigmoid(x_scaled)
        col_center = inv_sigmoid(y_scaled)
        k = self.cfg.support_size
        with torch.no_grad():
            self.row_logits[0].copy_(torch.linspace(-0.2, 0.2, k))
            self.col_logits[0].copy_(torch.linspace(-0.2, 0.2, k))

            row_gen = self.row_generators[0]
            col_gen = self.col_generators[0]
            with torch.no_grad():
                z_row = self.row_noise_anchors[0].unsqueeze(-1)
                t_ones = torch.ones(k, 1)
                raw_row = row_gen(z_row, t_ones).squeeze(-1)
                row_bias_shift = row_center / max(self.cfg.policy_output_scale, 1e-6) - raw_row.mean()
                row_gen.net[-1].bias.add_(row_bias_shift)

                z_col = self.col_noise_anchors[0].unsqueeze(-1)
                raw_col = col_gen(z_col, t_ones).squeeze(-1)
                col_bias_shift = col_center / max(self.cfg.policy_output_scale, 1e-6) - raw_col.mean()
                col_gen.net[-1].bias.add_(col_bias_shift)

    def expected_row_action(self, idx: int) -> torch.Tensor:
        w, support = self._slot_support(idx, "row")
        return (w * support).sum()

    def expected_col_action(self, idx: int) -> torch.Tensor:
        w, support = self._slot_support(idx, "col")
        return (w * support).sum()

    def evaluate_payoff_matrix(self) -> np.ndarray:
        payoff = np.zeros((self.num_slots, self.num_slots), dtype=np.float32)
        with torch.no_grad():
            for i in range(self.num_slots):
                row_w, row_s = self._slot_support(i, "row")
                for j in range(self.num_slots):
                    col_w, col_s = self._slot_support(j, "col")
                    pairwise = cartesian_payoff(row_s[:, None], col_s[None, :])
                    payoff[i, j] = float((row_w @ pairwise @ col_w).item())
        return payoff

    def row_loss(self, sigma: torch.Tensor, slot_i: int) -> torch.Tensor:
        row_w, row_s = self._slot_support(slot_i, "row")
        value = torch.tensor(0.0, device=row_s.device)
        for j, weight in enumerate(sigma[slot_i]):
            w = float(weight.item())
            if w <= 1e-8:
                continue
            col_w, col_s = self._slot_support(j, "col")
            pairwise = cartesian_payoff(row_s[:, None], col_s.detach()[None, :])
            value = value + w * (row_w @ pairwise @ col_w.detach())
        return -value + self.cfg.consistency_coef * self._consistency_loss(slot_i, "row")

    def col_loss(self, sigma: torch.Tensor, slot_j: int) -> torch.Tensor:
        col_w, col_s = self._slot_support(slot_j, "col")
        value = torch.tensor(0.0, device=col_s.device)
        for i, weight in enumerate(sigma[slot_j]):
            w = float(weight.item())
            if w <= 1e-8:
                continue
            row_w, row_s = self._slot_support(i, "row")
            pairwise = cartesian_payoff(row_s.detach()[:, None], col_s[None, :])
            value = value + w * (row_w.detach() @ pairwise @ col_w)
        return value + self.cfg.consistency_coef * self._consistency_loss(slot_j, "col")

    def summarize_slot(self, idx: int) -> tuple[float, float, float]:
        with torch.no_grad():
            row_w, row_s = self._slot_support(idx, "row")
            col_w, col_s = self._slot_support(idx, "col")
            x = float((row_w * row_s).sum().item())
            y = float((col_w * col_s).sum().item())
            value = float((row_w @ cartesian_payoff(row_s[:, None], col_s[None, :]) @ col_w).item())
        return x, y, value

    def meta_density(
        self,
        row_weights: np.ndarray,
        col_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_grid = np.linspace(self.game_cfg.xmin, self.game_cfg.xmax, 400)
        y_grid = np.linspace(self.game_cfg.ymin, self.game_cfg.ymax, 400)
        row_density = np.zeros_like(x_grid, dtype=np.float64)
        col_density = np.zeros_like(y_grid, dtype=np.float64)
        bandwidth_x = 0.06 * (self.game_cfg.xmax - self.game_cfg.xmin)
        bandwidth_y = 0.06 * (self.game_cfg.ymax - self.game_cfg.ymin)
        with torch.no_grad():
            for idx, slot_weight in enumerate(row_weights):
                if slot_weight <= 1e-10:
                    continue
                atom_w, atom_s = self._slot_support(idx, "row")
                row_density += slot_weight * gaussian_mixture_density(
                    atom_s.cpu().numpy(),
                    atom_w.cpu().numpy(),
                    x_grid,
                    bandwidth_x,
                )
            for idx, slot_weight in enumerate(col_weights):
                if slot_weight <= 1e-10:
                    continue
                atom_w, atom_s = self._slot_support(idx, "col")
                col_density += slot_weight * gaussian_mixture_density(
                    atom_s.cpu().numpy(),
                    atom_w.cpu().numpy(),
                    y_grid,
                    bandwidth_y,
                )
        return x_grid, y_grid, row_density, col_density

def compute_exploitability(
    population: BasePopulation,
    row_mix: np.ndarray,
    col_mix: np.ndarray,
    game_cfg: GameConfig,
    resolution: int = 181,
) -> dict[str, float]:
    x_grid = np.linspace(game_cfg.xmin, game_cfg.xmax, resolution)
    y_grid = np.linspace(game_cfg.ymin, game_cfg.ymax, resolution)
    payoff_slots = population.evaluate_payoff_matrix()
    profile_value = float(row_mix @ payoff_slots @ col_mix)

    with torch.no_grad():
        row_vs_y = []
        for x in x_grid:
            x_t = torch.tensor(float(x), dtype=torch.float32)
            value = 0.0
            for j, weight in enumerate(col_mix):
                if weight <= 1e-10:
                    continue
                col_w, col_s = population._slot_support(j, "col")
                value += float(weight * (cartesian_payoff(x_t, col_s) @ col_w).item())
            row_vs_y.append(value)

        col_vs_x = []
        for y in y_grid:
            y_t = torch.tensor(float(y), dtype=torch.float32)
            value = 0.0
            for i, weight in enumerate(row_mix):
                if weight <= 1e-10:
                    continue
                row_w, row_s = population._slot_support(i, "row")
                value += float(weight * (row_w @ cartesian_payoff(row_s, y_t)).item())
            col_vs_x.append(value)

    row_br_value = float(np.max(row_vs_y))
    col_br_value = float(np.min(col_vs_x))
    row_gap = row_br_value - profile_value
    col_gap = profile_value - col_br_value
    exploitability = max(row_gap, col_gap)
    return {
        "profile_value": profile_value,
        "row_br_value": row_br_value,
        "col_br_value": col_br_value,
        "row_gap": float(row_gap),
        "col_gap": float(col_gap),
        "exploitability": float(exploitability),
    }

def train_neupl(
    population: BasePopulation,
    cfg: TrainConfig,
    graph_mode: str,
) -> tuple[dict, dict, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    opt = optim.Adam(population.parameters(), lr=cfg.lr)
    n = population.num_slots
    if graph_mode in {"adaptive", "psro_nash"}:
        row_sigma = torch.full((n, n), 1.0 / n)
        col_sigma = torch.full((n, n), 1.0 / n)
    elif graph_mode == "self_play":
        sigma = torch.from_numpy(sigma_self_play(n))
        row_sigma = sigma.clone()
        col_sigma = sigma.clone()
    elif graph_mode == "cycle":
        sigma = torch.from_numpy(sigma_strategic_cycle(n))
        row_sigma = sigma.clone()
        col_sigma = sigma.clone()
    elif graph_mode == "fp":
        sigma = torch.from_numpy(sigma_fictitious_play(n))
        row_sigma = sigma.clone()
        col_sigma = sigma.clone()
    else:
        raise ValueError(f"Unknown graph mode: {graph_mode}")

    row_history = {i: [] for i in range(n)}
    col_history = {i: [] for i in range(n)}
    row_sigma_history: list[np.ndarray] = []
    col_sigma_history: list[np.ndarray] = []
    payoff_history: list[np.ndarray] = []

    for it in range(cfg.outer_iters):
        if graph_mode in {"adaptive", "psro_nash"} and it % cfg.epoch_len == 0:
            payoff = population.evaluate_payoff_matrix()
            row_sigma_np, col_sigma_np = psro_nash_meta_graph_solver(payoff)
            row_sigma = torch.from_numpy(row_sigma_np).float()
            col_sigma = torch.from_numpy(col_sigma_np).float()
            row_sigma_history.append(row_sigma_np.copy())
            col_sigma_history.append(col_sigma_np.copy())
            payoff_history.append(payoff.copy())
        elif graph_mode not in {"adaptive", "psro_nash"} and it == 0:
            payoff = population.evaluate_payoff_matrix()
            row_sigma_history.append(row_sigma.detach().cpu().numpy().copy())
            col_sigma_history.append(col_sigma.detach().cpu().numpy().copy())
            payoff_history.append(payoff.copy())

        for i in unique_rows(row_sigma):
            if row_sigma[i].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps):
                loss = population.row_loss(row_sigma, i)
                opt.zero_grad()
                loss.backward()
                opt.step()

        for j in unique_rows(col_sigma):
            if col_sigma[j].sum().item() < 1e-8:
                continue
            for _ in range(cfg.abr_steps):
                loss = population.col_loss(col_sigma, j)
                opt.zero_grad()
                loss.backward()
                opt.step()

        for idx in range(n):
            row_history[idx].append(float(population.expected_row_action(idx).item()))
            col_history[idx].append(float(population.expected_col_action(idx).item()))

        if (it + 1) % 50 == 0:
            print(f"  [{graph_mode} iter {it + 1}/{cfg.outer_iters}]")

    for idx in range(n):
        row_history[idx] = np.asarray(row_history[idx], dtype=np.float32)
        col_history[idx] = np.asarray(col_history[idx], dtype=np.float32)

    return row_history, col_history, row_sigma_history, col_sigma_history, payoff_history

def make_payoff_grid(game_cfg: GameConfig, resolution: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(game_cfg.xmin, game_cfg.xmax, resolution)
    ys = np.linspace(game_cfg.ymin, game_cfg.ymax, resolution)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.sin(3.0 * xx) * np.cos(3.0 * yy)
    return xx, yy, zz

def draw_payoff_landscape(ax: plt.Axes, game_cfg: GameConfig, title: str):
    xx, yy, zz = make_payoff_grid(game_cfg)
    contour = ax.contourf(xx, yy, zz, levels=40, cmap="coolwarm", alpha=0.42)
    ax.contour(xx, yy, zz, levels=10, colors="white", linewidths=0.35, alpha=0.35)
    plt.colorbar(contour, ax=ax, shrink=0.8)
    ax.set_xlabel("x (row player)")
    ax.set_ylabel("y (column player)")
    ax.set_title(title)

def plot_discretized_nash(
    game_cfg: GameConfig,
    x_grid: np.ndarray,
    row_mix: np.ndarray,
    col_mix: np.ndarray,
    value: float,
    save_path: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    draw_payoff_landscape(axes[0], game_cfg, f"Payoff landscape (v*={value:.4f})")
    axes[1].plot(x_grid, row_mix, color="#155e75", linewidth=2)
    axes[1].fill_between(x_grid, 0.0, row_mix, color="#67e8f9", alpha=0.35)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("probability")
    axes[1].set_title("Discretized Nash: row player")
    axes[2].plot(x_grid, col_mix, color="#9a3412", linewidth=2)
    axes[2].fill_between(x_grid, 0.0, col_mix, color="#fdba74", alpha=0.35)
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("probability")
    axes[2].set_title("Discretized Nash: column player")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)

def plot_trajectories(
    row_history: dict,
    col_history: dict,
    game_cfg: GameConfig,
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    draw_payoff_landscape(ax, game_cfg, title)
    for idx in sorted(row_history.keys()):
        xs = row_history[idx]
        ys = col_history[idx]
        colors = plt.cm.viridis(np.linspace(0, 1, len(xs)))
        ax.scatter(xs, ys, c=colors, s=12, alpha=0.8, edgecolors="none")
        ax.plot(xs[-1], ys[-1], "ko", markersize=4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)

def plot_adaptive_detail(
    row_history: dict,
    col_history: dict,
    row_sigma_history: list[np.ndarray],
    col_sigma_history: list[np.ndarray],
    payoff_history: list[np.ndarray],
    game_cfg: GameConfig,
    save_path: str,
):
    n_snaps = min(len(payoff_history), 4)
    snap_idx = np.linspace(0, len(payoff_history) - 1, n_snaps, dtype=int)
    fig, axes = plt.subplots(3, n_snaps + 1, figsize=(5 * (n_snaps + 1), 12))
    draw_payoff_landscape(axes[0, 0], game_cfg, "Adaptive trajectories")
    for idx in sorted(row_history.keys()):
        axes[0, 0].plot(row_history[idx], col_history[idx], alpha=0.4)
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    for col, snap in enumerate(snap_idx):
        im0 = axes[0, col + 1].imshow(payoff_history[snap], cmap="coolwarm")
        axes[0, col + 1].set_title(f"Payoff @ epoch {snap}")
        plt.colorbar(im0, ax=axes[0, col + 1], shrink=0.7)
        im1 = axes[1, col + 1].imshow(row_sigma_history[snap], cmap="Blues", vmin=0, vmax=1)
        axes[1, col + 1].set_title(f"Row Sigma @ epoch {snap}")
        plt.colorbar(im1, ax=axes[1, col + 1], shrink=0.7)
        im2 = axes[2, col + 1].imshow(col_sigma_history[snap], cmap="Greens", vmin=0, vmax=1)
        axes[2, col + 1].set_title(f"Col Sigma @ epoch {snap}")
        plt.colorbar(im2, ax=axes[2, col + 1], shrink=0.7)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)

def plot_density(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    row_density: np.ndarray,
    col_density: np.ndarray,
    title: str,
    save_path: str,
):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    axes[0].plot(x_grid, row_density, color="#155e75", linewidth=2)
    axes[0].fill_between(x_grid, 0.0, row_density, color="#67e8f9", alpha=0.35)
    axes[0].set_ylabel("density / mass")
    axes[0].set_title(f"{title}: row-player support")
    axes[1].plot(y_grid, col_density, color="#9a3412", linewidth=2)
    axes[1].fill_between(y_grid, 0.0, col_density, color="#fdba74", alpha=0.35)
    axes[1].set_xlabel("action")
    axes[1].set_ylabel("density / mass")
    axes[1].set_title(f"{title}: column-player support")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)

def print_final_summary(population: BasePopulation, label: str):
    print(f"\n  Final slot summaries for [{label}]:")
    for idx in range(population.num_slots):
        x, y, value = population.summarize_slot(idx)
        print(f"    slot {idx}: x={x:.4f}, y={y:.4f}, value={value:.4f}")

def compute_diversity_metrics(
    population: BasePopulation,
    row_mix: np.ndarray,
    col_mix: np.ndarray,
) -> dict[str, float]:
    n = population.num_slots
    positions = [population.summarize_slot(i) for i in range(n)]
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])

    n_eff_row = int((row_mix > 0.01).sum())
    n_eff_col = int((col_mix > 0.01).sum())

    def entropy(p: np.ndarray) -> float:
        p = p[p > 1e-10]
        return float(-np.sum(p * np.log(p)))

    def weighted_pairwise_dist(vals: np.ndarray, w: np.ndarray) -> float:
        total = 0.0
        norm = 0.0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                ww = w[i] * w[j]
                total += ww * abs(vals[i] - vals[j])
                norm += ww
        return float(total / max(norm, 1e-10))

    row_mean = float(np.sum(row_mix * xs))
    col_mean = float(np.sum(col_mix * ys))
    row_std = float(np.sqrt(np.sum(row_mix * (xs - row_mean) ** 2)))
    col_std = float(np.sqrt(np.sum(col_mix * (ys - col_mean) ** 2)))

    return {
        "n_effective_row": n_eff_row,
        "n_effective_col": n_eff_col,
        "row_entropy": entropy(row_mix),
        "col_entropy": entropy(col_mix),
        "row_pairwise_dist": weighted_pairwise_dist(xs, row_mix),
        "col_pairwise_dist": weighted_pairwise_dist(ys, col_mix),
        "row_weighted_std": row_std,
        "col_weighted_std": col_std,
    }

def compute_nash_closeness(
    population: BasePopulation,
    row_mix: np.ndarray,
    col_mix: np.ndarray,
    game_cfg: GameConfig,
    nash_grid: np.ndarray,
    nash_row: np.ndarray,
    nash_col: np.ndarray,
    nash_value: float,
    exploit: dict[str, float],
) -> dict[str, float]:
    value_gap = abs(exploit["profile_value"] - nash_value)

    x_grid, y_grid, row_density, col_density = population.meta_density(row_mix, col_mix)

    nash_row_interp = np.interp(x_grid, nash_grid, nash_row)
    nash_col_interp = np.interp(y_grid, nash_grid, nash_col)

    def normalise_density(d: np.ndarray, g: np.ndarray) -> np.ndarray:
        t = _trapz1d(d, g)
        return d / t if t > 0 else d

    row_dn = normalise_density(row_density, x_grid)
    col_dn = normalise_density(col_density, y_grid)
    nash_row_n = normalise_density(nash_row_interp, x_grid)
    nash_col_n = normalise_density(nash_col_interp, y_grid)

    row_l1 = float(_trapz1d(np.abs(row_dn - nash_row_n), x_grid))
    col_l1 = float(_trapz1d(np.abs(col_dn - nash_col_n), y_grid))

    return {
        "nash_value_gap": value_gap,
        "row_density_l1": row_l1,
        "col_density_l1": col_l1,
        "exploitability": exploit["exploitability"],
    }

def compute_mode_metrics(
    learned_density: np.ndarray,
    grid: np.ndarray,
    nash_density: np.ndarray,
    mode_tol: float = 0.15,
    peak_prominence: float = 0.05,
) -> dict[str, float]:

    nash_peaks, _ = find_peaks(nash_density, prominence=peak_prominence)
    learned_peaks, _ = find_peaks(learned_density, prominence=peak_prominence)
    n_true = len(nash_peaks)
    n_learned = len(learned_peaks)
    nash_locs = grid[nash_peaks]
    learned_locs = grid[learned_peaks]

    matched_nash: set[int] = set()
    n_optimal = 0
    for lp in learned_locs:
        dists = np.abs(nash_locs - lp)
        if len(dists) > 0 and dists.min() < mode_tol:
            n_optimal += 1
            matched_nash.add(int(dists.argmin()))

    n_suboptimal = n_learned - n_optimal
    precision = n_optimal / max(n_learned, 1)
    recall = len(matched_nash) / max(n_true, 1)
    return {
        "n_true_modes": float(n_true),
        "n_learned_modes": float(n_learned),
        "n_optimal_modes": float(n_optimal),
        "n_suboptimal_modes": float(n_suboptimal),
        "mode_precision": precision,
        "mode_recall": recall,
    }

def main():
    parser = argparse.ArgumentParser(description="MeanFlow NeuPL for a Cartesian game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-iters", type=int, default=300)
    parser.add_argument("--abr-steps", type=int, default=15)
    parser.add_argument("--epoch-len", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--save-dir", type=str, default="cartesian_meanflow_results")
    parser.add_argument("--xmin", type=float, default=-3.0)
    parser.add_argument("--xmax", type=float, default=3.0)
    parser.add_argument("--ymin", type=float, default=-3.0)
    parser.add_argument("--ymax", type=float, default=3.0)
    parser.add_argument("--num-bins", type=int, default=41)
    parser.add_argument("--adaptive-n", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--support-size", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--t-embed-dim", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--policy-output-scale", type=float, default=1.0)
    parser.add_argument("--consistency-coef", type=float, default=0.1)
    parser.add_argument(
        "--graph",
        type=str,
        default="psro_nash",
        choices=["adaptive", "psro_nash", "self_play", "cycle", "fp"],
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="neupl-cartesian")
    parser.add_argument("--wandb-entity", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    game_cfg = GameConfig(
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        num_bins=args.num_bins,
    )
    cfg = TrainConfig(
        outer_iters=args.outer_iters,
        abr_steps=args.abr_steps,
        lr=args.lr,
        seed=args.seed,
        epoch_len=args.epoch_len,
        flow_steps=args.flow_steps,
        support_size=args.support_size,
        hidden=args.hidden,
        t_embed_dim=args.t_embed_dim,
        noise_scale=args.noise_scale,
        policy_output_scale=args.policy_output_scale,
        consistency_coef=args.consistency_coef,
    )
    anchor = (0.7 * args.xmin + 0.3 * args.xmax, 0.25 * args.ymin + 0.75 * args.ymax)
    graph_label = "psro_nash" if args.graph == "adaptive" else args.graph
    model_tag = "meanflow"

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=f"{model_tag}_{graph_label}_s{args.seed}",
            config={
                "model": model_tag,
                "graph": graph_label,
                "seed": args.seed,
                "outer_iters": args.outer_iters,
                "abr_steps": args.abr_steps,
                "epoch_len": args.epoch_len,
                "lr": args.lr,
                "adaptive_n": args.adaptive_n,
                "flow_steps": args.flow_steps,
                "support_size": args.support_size,
                "hidden": args.hidden,
                "consistency_coef": args.consistency_coef,
            },
        )

    print("\n" + "=" * 60)
    print(f"  NeuPL ({graph_label}, N={args.adaptive_n}, model=meanflow)")
    print("=" * 60)

    nash_grid, nash_row, nash_col, nash_value = discretized_nash(game_cfg)
    print(
        f"  Discretized Nash value: {nash_value:.4f} | "
        f"row support={(nash_row > 1e-3).sum()} bins | "
        f"col support={(nash_col > 1e-3).sum()} bins"
    )
    plot_discretized_nash(
        game_cfg,
        nash_grid,
        nash_row,
        nash_col,
        nash_value,
        os.path.join(args.save_dir, "cartesian_discretized_nash.png"),
    )

    population = MeanFlowPopulation(args.adaptive_n, game_cfg, cfg)
    population.init_anchor(*anchor)
    row0, col0, _ = population.summarize_slot(0)
    print(f"  Anchor slot 0: x={row0:.3f}, y={col0:.3f}")

    row_history, col_history, row_sigma_history, col_sigma_history, payoff_history = train_neupl(
        population, cfg, graph_label
    )
    print_final_summary(population, f"NeuPL-{graph_label}-meanflow")

    plot_trajectories(
        row_history,
        col_history,
        game_cfg,
        f"NeuPL {graph_label} (meanflow)",
        os.path.join(args.save_dir, "cartesian_adaptive_landscape.png"),
    )
    plot_adaptive_detail(
        row_history,
        col_history,
        row_sigma_history,
        col_sigma_history,
        payoff_history,
        game_cfg,
        os.path.join(args.save_dir, "cartesian_adaptive_detail.png"),
    )

    final_row_mix = row_sigma_history[-1][-1]
    final_col_mix = col_sigma_history[-1][-1]
    x_grid, y_grid, row_density, col_density = population.meta_density(final_row_mix, final_col_mix)
    plot_density(
        x_grid,
        y_grid,
        row_density,
        col_density,
        f"NeuPL {graph_label} (meanflow)",
        os.path.join(args.save_dir, "cartesian_adaptive_density.png"),
    )

    exploit = compute_exploitability(population, final_row_mix, final_col_mix, game_cfg)
    print("\n  Final meta-mixture evaluation:")
    print(f"    profile value      = {exploit['profile_value']:.4f}")
    print(f"    row BR value       = {exploit['row_br_value']:.4f}")
    print(f"    col BR value       = {exploit['col_br_value']:.4f}")
    print(f"    row exploitability = {exploit['row_gap']:.4f}")
    print(f"    col exploitability = {exploit['col_gap']:.4f}")
    print(f"    exploitability     = {exploit['exploitability']:.4f}")

    div = compute_diversity_metrics(population, final_row_mix, final_col_mix)
    print("\n  Diversity metrics:")
    print(f"    effective slots    = row {div['n_effective_row']}, col {div['n_effective_col']}")
    print(f"    meta-mix entropy   = row {div['row_entropy']:.4f}, col {div['col_entropy']:.4f}")
    print(f"    pairwise dist      = row {div['row_pairwise_dist']:.4f}, col {div['col_pairwise_dist']:.4f}")
    print(f"    weighted std       = row {div['row_weighted_std']:.4f}, col {div['col_weighted_std']:.4f}")

    nash_close = compute_nash_closeness(
        population, final_row_mix, final_col_mix, game_cfg,
        nash_grid, nash_row, nash_col, nash_value, exploit,
    )
    print("\n  Nash-closeness metrics:")
    print(f"    value gap |v - v*| = {nash_close['nash_value_gap']:.4f}")
    print(f"    row density L1     = {nash_close['row_density_l1']:.4f}")
    print(f"    col density L1     = {nash_close['col_density_l1']:.4f}")
    print(f"    exploitability     = {nash_close['exploitability']:.4f}")

    nash_row_interp = np.interp(x_grid, nash_grid, nash_row)
    nash_col_interp = np.interp(y_grid, nash_grid, nash_col)
    row_modes = compute_mode_metrics(row_density, x_grid, nash_row_interp)
    col_modes = compute_mode_metrics(col_density, y_grid, nash_col_interp)
    print("\n  Mode discovery (row):")
    print(f"    true Nash modes    = {int(row_modes['n_true_modes'])}")
    print(f"    learned modes      = {int(row_modes['n_learned_modes'])}")
    print(f"    optimal modes      = {int(row_modes['n_optimal_modes'])}")
    print(f"    suboptimal modes   = {int(row_modes['n_suboptimal_modes'])}")
    print(f"    precision          = {row_modes['mode_precision']:.4f}")
    print(f"    recall             = {row_modes['mode_recall']:.4f}")
    print("  Mode discovery (col):")
    print(f"    true Nash modes    = {int(col_modes['n_true_modes'])}")
    print(f"    learned modes      = {int(col_modes['n_learned_modes'])}")
    print(f"    optimal modes      = {int(col_modes['n_optimal_modes'])}")
    print(f"    suboptimal modes   = {int(col_modes['n_suboptimal_modes'])}")
    print(f"    precision          = {col_modes['mode_precision']:.4f}")
    print(f"    recall             = {col_modes['mode_recall']:.4f}")

    summary = {
        "model": model_tag,
        "graph": graph_label,
        **{f"exploit/{k}": v for k, v in exploit.items()},
        **{f"diversity/{k}": v for k, v in div.items()},
        **{f"nash/{k}": v for k, v in nash_close.items()},
        **{f"modes_row/{k}": v for k, v in row_modes.items()},
        **{f"modes_col/{k}": v for k, v in col_modes.items()},
    }
    summary_path = os.path.join(args.save_dir, "run_summary.npz")
    np.savez_compressed(
        summary_path,
        x_grid=x_grid,
        y_grid=y_grid,
        row_density=row_density,
        col_density=col_density,
        **{k: v for k, v in summary.items() if isinstance(v, (int, float))},
        model_kind=np.array(model_tag),
    )
    print(f"  Saved: {summary_path}")

    if args.wandb:
        import wandb
        wandb.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nDone! All plots saved to: {args.save_dir}/")

if __name__ == "__main__":
    main()
