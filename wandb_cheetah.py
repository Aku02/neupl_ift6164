"""Compare `neupl-jpsro-seed42` vs `meanflow-jpsro-seed42` on W&B.

Both runs live in project `ndfl-uw/neupl-jpsro-cheetah`.  We pull:
  - per-step BR training returns:        br/p0_return, br/p1_return
  - per-iter JPSRO best-joint payoff:    jpsro/best_payoff
  - per-iter CCE gap:                    jpsro/cce_gap
and compute a side-by-side comparison + summary.

Key shape difference vs. the old halfcheetah_flow_benchmark script:
  - runs are named exactly (no prefix grouping); we match via `display_name`.
  - metric namespace is `br/*` and `jpsro/*` (not flat `actor_loss` etc.).
"""
import wandb
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

wandb_entity = "ndfl-uw"
wandb_project = "neupl-jpsro-cheetah"

# Exact wandb run display-names we want to compare.
RUN_NAMES = [
    "neupl-jpsro-seed42",
    "meanflow-jpsro-seed42",
]

PRETTY_NAMES = {
    "neupl-jpsro-seed42":     "NeuPL-JPSRO (PPO-BR)",
    "meanflow-jpsro-seed42":  "MeanFlow-NFT-JPSRO",
}

GROUP_COLORS = {
    "neupl-jpsro-seed42":     "#4CAF50",
    "meanflow-jpsro-seed42":  "#9C27B0",
}

folder_name = "exp_results_jpsro_cheetah"
plot_folder = f"{folder_name}/plots"
os.makedirs(folder_name, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

# ── Metric schemas logged by both scripts --------------------------------------
# per-step (x-axis = br/global_step):
#   br/p0_return, br/p1_return, br/p0_iter, br/p1_iter
# per-iter (x-axis = br/global_step at iter boundary):
#   jpsro/best_payoff, jpsro/cce_gap, jpsro/iteration,
#   jpsro/num_strategies_p0, jpsro/num_strategies_p1,
#   jpsro/iter_time_s
KEYS_OF_INTEREST = [
    "br/p0_return", "br/p1_return",
    "br/p0_iter",   "br/p1_iter",
    "jpsro/best_payoff", "jpsro/cce_gap", "jpsro/iteration",
    "jpsro/num_strategies_p0", "jpsro/num_strategies_p1",
    "jpsro/iter_time_s", "jpsro/initial_payoff",
]


# ── Data fetching -------------------------------------------------------------

def fetch_run_by_name(api, display_name, use_cache=True):
    """Pull the per-step history for a single exact-named run. Cached to disk."""
    cache_path = f"{folder_name}/{display_name}_history.json"
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    runs = list(api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters={"display_name": display_name},
    ))
    if not runs:
        print(f"  !! No run named '{display_name}' found in "
              f"{wandb_entity}/{wandb_project}")
        return None

    # Most-recent if duplicates.
    runs.sort(key=lambda r: r.created_at, reverse=True)
    run = runs[0]
    print(f"  Fetching history for {display_name} (id={run.id}, "
          f"state={run.state}, created={run.created_at}) ...")

    try:
        history = run.history(samples=100_000, pandas=False)
    except Exception as e:
        print(f"    Failed to fetch history: {e}")
        return None

    if not history:
        print(f"    (empty history)")
        return None

    all_cols = set()
    for row in history:
        all_cols.update(row.keys())

    wanted = {c for c in all_cols if c.startswith("_")}
    for c in all_cols:
        if c in KEYS_OF_INTEREST or c.startswith("br/") or c.startswith("jpsro/"):
            wanted.add(c)

    keep_cols = sorted(wanted)
    col_data = {c: [] for c in keep_cols}
    for row in history:
        for c in keep_cols:
            col_data[c].append(row.get(c))

    print(f"    Kept {len(keep_cols)} cols, {len(history)} rows.")
    with open(cache_path, "w") as f:
        json.dump(col_data, f)
    return col_data


# ── Extraction helpers --------------------------------------------------------

def extract_series(run_data, key, x_key="br/global_step"):
    """Return (xs, ys) numpy arrays of rows where `key` is not None."""
    if run_data is None or key not in run_data:
        return np.array([]), np.array([])
    y_raw = run_data[key]
    x_raw = run_data.get(x_key, run_data.get("_step"))
    if x_raw is None:
        return np.array([]), np.array([])
    xs, ys = [], []
    for x, y in zip(x_raw, y_raw):
        if y is None:
            continue
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isnan(yf):
            continue
        try:
            xf = float(x) if x is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if np.isnan(xf):
            continue
        xs.append(xf); ys.append(yf)
    return np.array(xs), np.array(ys)


def smooth(y, w=25):
    if len(y) == 0 or w <= 1:
        return y
    w = min(w, len(y))
    kernel = np.ones(w) / w
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")[:len(y)]


# ── Plotting ------------------------------------------------------------------

def plot_return_curve(ax, all_run_data, key, title, smooth_w=25):
    any_plotted = False
    for name, rd in all_run_data.items():
        xs, ys = extract_series(rd, key)
        if len(xs) == 0:
            print(f"    {name}: no data for {key}")
            continue
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        color = GROUP_COLORS.get(name, "#333333")
        label = PRETTY_NAMES.get(name, name)
        ax.plot(xs, ys, alpha=0.25, linewidth=0.9, color=color)
        ax.plot(xs, smooth(ys, smooth_w), linewidth=2.0, color=color,
                label=f"{label}  (n={len(ys)}, max={ys.max():.1f}, "
                      f"final={ys[-1]:.1f})")
        any_plotted = True
    ax.set_xlabel("BR global step")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if any_plotted:
        ax.legend(loc="best", framealpha=0.9, edgecolor="none")
    return any_plotted


def plot_iter_curve(ax, all_run_data, key, title):
    any_plotted = False
    for name, rd in all_run_data.items():
        xs_step, ys = extract_series(rd, key)
        if len(ys) == 0:
            print(f"    {name}: no data for {key}")
            continue
        # Prefer plotting against jpsro/iteration if it's logged at the same
        # step as `key` (otherwise fall back to row index).
        iters_x, iters_y = extract_series(rd, "jpsro/iteration")
        if len(iters_y) == len(ys):
            xs_iter = iters_y
        else:
            xs_iter = np.arange(1, len(ys) + 1)
        color = GROUP_COLORS.get(name, "#333333")
        label = PRETTY_NAMES.get(name, name)
        ax.plot(xs_iter, ys, marker="o", linewidth=2.0, color=color,
                label=f"{label}  (max={ys.max():.1f}, final={ys[-1]:.1f})")
        any_plotted = True
    ax.set_xlabel("JPSRO iteration")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if any_plotted:
        ax.legend(loc="best", framealpha=0.9, edgecolor="none")
    return any_plotted


# ── Summary table -------------------------------------------------------------

def summarize(all_run_data):
    rows = []
    for name, rd in all_run_data.items():
        entry = {"run": name}
        for key in ["br/p0_return", "br/p1_return"]:
            _, ys = extract_series(rd, key)
            entry[f"{key}/max"]   = float(ys.max())   if len(ys) else float("nan")
            entry[f"{key}/final"] = float(ys[-1])     if len(ys) else float("nan")
            entry[f"{key}/mean_last20pct"] = (
                float(ys[int(0.8 * len(ys)):].mean())
                if len(ys) else float("nan")
            )
        _, bp = extract_series(rd, "jpsro/best_payoff")
        entry["jpsro/best_payoff/max"]   = float(bp.max()) if len(bp) else float("nan")
        entry["jpsro/best_payoff/final"] = float(bp[-1])   if len(bp) else float("nan")

        _, gap = extract_series(rd, "jpsro/cce_gap")
        entry["jpsro/cce_gap/final"] = float(gap[-1]) if len(gap) else float("nan")

        _, n0 = extract_series(rd, "jpsro/num_strategies_p0")
        entry["jpsro/num_strategies_p0/final"] = int(n0[-1]) if len(n0) else -1

        _, itime = extract_series(rd, "jpsro/iter_time_s")
        entry["jpsro/iter_time_s/mean"] = (
            float(itime.mean()) if len(itime) else float("nan")
        )
        rows.append(entry)
    return rows


def print_summary_table(rows):
    if not rows:
        print("  (no runs)")
        return
    keys = [k for k in rows[0] if k != "run"]
    name_w = max(len(PRETTY_NAMES.get(r["run"], r["run"])) for r in rows)
    print()
    print(f"  {'Metric':<42}  " + "  ".join(
        f"{PRETTY_NAMES.get(r['run'], r['run']):>{name_w}}" for r in rows))
    print(f"  {'-' * 42}  " + "  ".join("-" * name_w for _ in rows))
    for k in keys:
        vals = [r.get(k, float("nan")) for r in rows]
        fmt_vals = []
        for v in vals:
            if isinstance(v, (int, np.integer)):
                fmt_vals.append(f"{v:>{name_w}d}")
            else:
                fmt_vals.append(f"{v:>{name_w}.3f}")
        print(f"  {k:<42}  " + "  ".join(fmt_vals))
    print()


# ── Main ---------------------------------------------------------------------

def main(use_cache=True):
    api = wandb.Api(timeout=120)
    print(f"Fetching runs from {wandb_entity}/{wandb_project} ...")
    all_run_data = {}
    for name in RUN_NAMES:
        print(f"\nRun: {name}")
        all_run_data[name] = fetch_run_by_name(api, name, use_cache=use_cache)

    rows = summarize(all_run_data)
    print("\n" + "=" * 80)
    print("  Summary (higher = better for returns & best_payoff; "
          "lower = better for cce_gap)")
    print("=" * 80)
    print_summary_table(rows)

    # ── Figure: BR per-step returns (p0, p1) + JPSRO best_payoff --------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    plot_return_curve(axes[0][0], all_run_data, "br/p0_return",
                      "Player 0 BR Training Returns (smoothed)")
    plot_return_curve(axes[0][1], all_run_data, "br/p1_return",
                      "Player 1 BR Training Returns (smoothed)")
    plot_iter_curve(axes[0][2], all_run_data, "jpsro/best_payoff",
                    "JPSRO best-joint Payoff per Iteration")
    fig.suptitle("NeuPL-JPSRO vs MeanFlow-NFT-JPSRO — HalfCheetah 2-agent (seed 42)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = f"{plot_folder}/returns_and_payoff.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")

    # ── Figure: CCE gap & population size per iteration ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    plot_iter_curve(axes[0][0], all_run_data, "jpsro/cce_gap",
                    "CCE Gap (lower = closer to equilibrium)")
    plot_iter_curve(axes[0][1], all_run_data, "jpsro/num_strategies_p0",
                    "Population size (P0)")
    fig.suptitle("JPSRO Diagnostics", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = f"{plot_folder}/jpsro_diagnostics.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")

    # ── Verdict ---------------------------------------------------------------
    if len(rows) == 2:
        a, b = rows[0], rows[1]
        print("=" * 80)
        print("  Head-to-head (a = {}, b = {})".format(
            PRETTY_NAMES.get(a["run"], a["run"]),
            PRETTY_NAMES.get(b["run"], b["run"])))
        print("=" * 80)
        for k in ["br/p0_return/max", "br/p0_return/final",
                  "br/p0_return/mean_last20pct",
                  "br/p1_return/max", "br/p1_return/final",
                  "br/p1_return/mean_last20pct",
                  "jpsro/best_payoff/max", "jpsro/best_payoff/final"]:
            va, vb = a.get(k), b.get(k)
            if va is None or vb is None or np.isnan(va) or np.isnan(vb):
                continue
            winner = (PRETTY_NAMES.get(a["run"], a["run"])
                      if va > vb
                      else PRETTY_NAMES.get(b["run"], b["run"]))
            diff_pct = 100.0 * (max(va, vb) - min(va, vb)) / max(
                abs(min(va, vb)), 1e-8)
            print(f"  {k:<42}  a={va:>9.2f}  b={vb:>9.2f}  "
                  f"-> {winner}  (+{diff_pct:.1f}%)")
        for k in ["jpsro/cce_gap/final"]:
            va, vb = a.get(k), b.get(k)
            if va is None or vb is None or np.isnan(va) or np.isnan(vb):
                continue
            winner = (PRETTY_NAMES.get(a["run"], a["run"])
                      if va < vb
                      else PRETTY_NAMES.get(b["run"], b["run"]))
            print(f"  {k:<42}  a={va:>9.4f}  b={vb:>9.4f}  -> {winner} (lower)")
        print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true",
                    help="Re-download histories from W&B instead of reading the local cache.")
    args = ap.parse_args()
    main(use_cache=not args.no_cache)
