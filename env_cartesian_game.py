"""
Cartesian game environment: two agents control x and y; payoff is a highly
nonlinear function with multiple (local) optima.

- Agent 1 chooses x ∈ [x_min, x_max]
- Agent 2 chooses y ∈ [y_min, y_max]
- Payoff for agent 1: J1(x, y) = f(x, y)  (e.g. nonlinear in both)
- Payoff for agent 2: J2(x, y) = -f(x, y) for zero-sum, or g(x,y) for general-sum

We use a nonlinear f with multiple peaks/valleys so there are multiple
Nash-like or locally optimal (x*, y*) pairs.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CartesianGameConfig:
    """Bounds and nonlinearity for the Cartesian game."""
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    # Nonlinear function parameters (multiple modes)
    n_modes: int = 3
    scale: float = 1.0


def payoff_agent1_cartesian(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_modes: int = 3,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Highly nonlinear payoff for agent 1 (row player): J1(x, y).
    Multiple local optima via sum of Gaussians / sinusoidal terms.

    f(x,y) = scale * sum over modes of sin(k*x) * cos(k*y) type terms
    so that there are several saddle/peak structures.
    """
    out = np.zeros_like(x, dtype=float)
    for k in range(1, n_modes + 1):
        out += np.sin(k * x) * np.cos(k * y) + 0.3 * np.sin(2 * k * (x + y))
    return scale * out


def payoff_agent2_zero_sum(x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    """Zero-sum: J2 = -J1."""
    return -payoff_agent1_cartesian(x, y, **kwargs)


def payoff_cartesian(
    x: np.ndarray,
    y: np.ndarray,
    zero_sum: bool = True,
    n_modes: int = 3,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (J1, J2). J1 = f(x,y), J2 = -f(x,y) if zero_sum else g(x,y).
    For general-sum we could let J2 = f(y, x) or another nonlinear function.
    """
    J1 = payoff_agent1_cartesian(x, y, n_modes=n_modes, scale=scale)
    if zero_sum:
        J2 = -J1
    else:
        J2 = payoff_agent1_cartesian(y, x, n_modes=n_modes, scale=scale)
    return J1, J2


class CartesianGameEnv:
    """
    Environment for the Cartesian game: two agents control x and y.

    - state / observation: none (normal-form style) or (x, y) for logging
    - action space: agent 1 -> x in [x_min, x_max], agent 2 -> y in [y_min, y_max]
    - step(x, y): returns (J1, J2) and done=True (one-shot)
    """

    def __init__(self, config: CartesianGameConfig | None = None):
        self.config = config or CartesianGameConfig()
        self.x_min = self.config.x_min
        self.x_max = self.config.x_max
        self.y_min = self.config.y_min
        self.y_max = self.config.y_max
        self.n_modes = self.config.n_modes
        self.scale = self.config.scale

    def clip(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clip actions to bounds."""
        x = np.clip(x, self.x_min, self.x_max)
        y = np.clip(y, self.y_min, self.y_max)
        return x, y

    def step(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        zero_sum: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        One-shot game: agents have already chosen x and y.
        Returns (J1, J2, done=True).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x, y = self.clip(x, y)
        J1, J2 = payoff_cartesian(
            x, y,
            zero_sum=zero_sum,
            n_modes=self.n_modes,
            scale=self.scale,
        )
        return J1, J2, True

    def payoff_matrix_from_populations(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zero_sum: bool = True,
    ) -> np.ndarray:
        """
        xs: (n1,) strategies of agent-1 population
        ys: (n2,) strategies of agent-2 population
        Returns (n1, n2) matrix U where U[i,j] = J1(xs[i], ys[j]).
        """
        xx = np.asarray(xs).reshape(-1, 1)
        yy = np.asarray(ys).reshape(1, -1)
        J1, _ = payoff_cartesian(
            np.broadcast_to(xx, (xx.shape[0], yy.shape[1])),
            np.broadcast_to(yy, (xx.shape[0], yy.shape[1])),
            zero_sum=zero_sum,
            n_modes=self.n_modes,
            scale=self.scale,
        )
        return J1


# Optional: torch versions for use with NeuPL/diffusion (differentiable)
try:
    import torch

    def payoff_agent1_cartesian_torch(
        x: torch.Tensor,
        y: torch.Tensor,
        n_modes: int = 3,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Differentiable payoff for agent 1; x and y are tensors."""
        out = torch.zeros_like(x, dtype=torch.float32)
        for k in range(1, n_modes + 1):
            out = out + torch.sin(k * x) * torch.cos(k * y) + 0.3 * torch.sin(2 * k * (x + y))
        return scale * out
except ImportError:
    payoff_agent1_cartesian_torch = None
