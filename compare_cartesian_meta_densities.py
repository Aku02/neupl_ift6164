"""
Overlay meta-marginal densities from two Cartesian NeuPL runs (e.g. meanflow vs diffusion).

Each training script writes ``run_summary.npz`` into ``--save-dir`` with keys:
  x_grid, y_grid, row_density, col_density, exploitability, row_density_l1, col_density_l1, ...

Usage:
  python compare_cartesian_meta_densities.py \\
    --a cartesian_results_meanflow/run_summary.npz \\
    --b cartesian_results_diffusion/run_summary.npz \\
    --out cartesian_compare_coverage.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Compare meta-density coverage between two NeuPL runs")
    p.add_argument("--a", required=True, help="First run_summary.npz")
    p.add_argument("--a-label", type=str, default="meanflow")
    p.add_argument("--b", required=True, help="Second run_summary.npz")
    p.add_argument("--b-label", type=str, default="diffusion")
    p.add_argument("--out", type=str, default="cartesian_coverage_compare.png")
    args = p.parse_args()

    A = np.load(args.a, allow_pickle=True)
    B = np.load(args.b, allow_pickle=True)

    def _txt(z) -> str:
        parts = []
        if "exploitability" in z.files:
            parts.append(f"expl={float(z['exploitability']):.4f}")
        if "row_density_l1" in z.files:
            parts.append(f"L1_row={float(z['row_density_l1']):.3f}")
        if "col_density_l1" in z.files:
            parts.append(f"L1_col={float(z['col_density_l1']):.3f}")
        return " | ".join(parts)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    xa, ra = A["x_grid"], A["row_density"]
    xb, rb = B["x_grid"], B["row_density"]
    ya, ca = A["y_grid"], A["col_density"]
    yb, cb = B["y_grid"], B["col_density"]

    axes[0].plot(xa, ra, color="#0284c7", lw=2.2, label=args.a_label)
    axes[0].fill_between(xa, 0.0, ra, color="#0284c7", alpha=0.2)
    axes[0].plot(xb, rb, color="#c2410c", lw=2.2, label=args.b_label, alpha=0.9)
    axes[0].fill_between(xb, 0.0, rb, color="#c2410c", alpha=0.15)
    axes[0].set_ylabel("density / mass")
    axes[0].set_title("Meta-mixture coverage: row player (x)")
    axes[0].legend(loc="upper right", framealpha=0.9)

    axes[1].plot(ya, ca, color="#0284c7", lw=2.2, label=args.a_label)
    axes[1].fill_between(ya, 0.0, ca, color="#0284c7", alpha=0.2)
    axes[1].plot(yb, cb, color="#c2410c", lw=2.2, label=args.b_label, alpha=0.9)
    axes[1].fill_between(yb, 0.0, cb, color="#c2410c", alpha=0.15)
    axes[1].set_xlabel("action")
    axes[1].set_ylabel("density / mass")
    axes[1].set_title("Meta-mixture coverage: column player (y)")
    axes[1].legend(loc="upper right", framealpha=0.9)

    fig.text(
        0.02,
        0.02,
        f"{args.a_label}: {_txt(A)}\n{args.b_label}: {_txt(B)}",
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
