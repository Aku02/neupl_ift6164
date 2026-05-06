#!/usr/bin/env python3
"""
Visualize NeuPL-JPSRO HalfCheetah artifacts saved by neupl_jpsro_cheetah.py /
flowpl_jpsro_cheetah.py.

Outputs (pick any combination via flags):
  * interactive brax HTML viewer with slowed-down playback  (--trajectory / --all-trajectories)
  * MP4 video, 2D side-view ("track" camera), real-time or fps-controlled  (--mp4)
  * filmstrip PNG of N evenly spaced snapshots  (--snapshots N)
  * payoff / BR training summary PNG  (--run-summary)
  * one-line summary per checkpoint  (--list-checkpoints)

Examples:
  python visualize_neupl_cheetah.py --save-dir results_neupl_cheetah \\
      --all-trajectories --mp4 --snapshots 8 --fps 24
  python visualize_neupl_cheetah.py --save-dir results_neupl_cheetah \\
      --trajectory results_neupl_cheetah/trajectories/iter_008_s7_vs_s8.pkl \\
      --mp4 --fps 20 --width 640 --height 360
  python visualize_neupl_cheetah.py --save-dir results_neupl_cheetah --run-summary
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

# MuJoCo OpenGL backend must be picked BEFORE importing mujoco/brax.io.image.
# 'egl' is fast/hardware, 'osmesa' is software (headless-safe fallback).
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import jax
jax.tree_map = jax.tree.map

import jax.numpy as jp
import numpy as np

JAXMARL_PATH = os.path.join(os.path.dirname(__file__), "JaxMARL")
sys.path.insert(0, JAXMARL_PATH)
import jaxmarl

from brax.base import Motion, State, Transform
from brax.io import html


# ---------------------------------------------------------------------------
# Lazy / resilient import for mujoco-based image rendering
# ---------------------------------------------------------------------------

def _get_image_renderer():
    """Import brax.io.image lazily; fall back from egl -> osmesa on failure."""
    try:
        from brax.io import image as _img
        return _img
    except Exception as e:
        if os.environ.get("MUJOCO_GL") == "egl":
            os.environ["MUJOCO_GL"] = "osmesa"
            from brax.io import image as _img
            return _img
        raise e


# ---------------------------------------------------------------------------
# Trajectory loading / state reconstruction
# ---------------------------------------------------------------------------

def load_trajectory_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def trajectory_to_states(data: dict) -> list:
    """Build brax State list from saved arrays (see save_trajectory in trainer)."""
    q = np.asarray(data["q"])
    qd = np.asarray(data["qd"])
    x_pos = np.asarray(data["x_pos"])
    x_rot = np.asarray(data["x_rot"])
    n = q.shape[0]
    nl = x_pos.shape[1]
    states = []
    for t in range(n):
        x = Transform(pos=jp.array(x_pos[t]), rot=jp.array(x_rot[t]))
        xd = Motion(vel=jp.zeros((nl, 3)), ang=jp.zeros((nl, 3)))
        states.append(
            State(
                q=jp.array(q[t]), qd=jp.array(qd[t]),
                x=x, xd=xd, contact=None,
            )
        )
    return states


def _slow_sys_for_playback(sys, fps: float):
    """
    Return a clone of the brax System whose opt.timestep corresponds to 1/fps.

    The brax HTML viewer uses sys.opt.timestep as the per-frame JS interval. Saved
    trajectories have ONE State per env-step (not per integrator substep), so we
    need to lie to the viewer about timestep to get the right playback rate.
    """
    new_dt = float(1.0 / max(1.0, float(fps)))
    try:
        return sys.tree_replace({"opt.timestep": new_dt})
    except Exception:
        try:
            return sys.replace(opt=sys.opt.replace(timestep=new_dt))
        except Exception:
            return sys


# ---------------------------------------------------------------------------
# HTML render
# ---------------------------------------------------------------------------

def render_trajectory_html(
    traj_path: str,
    out_path: str,
    subsample: int = 1,
    height: int = 480,
    fps: float = 24.0,
) -> str:
    """
    Interactive brax HTML viewer, playback rate = `fps` frames per second.
    """
    data = load_trajectory_pkl(traj_path)
    states = trajectory_to_states(data)
    if subsample > 1:
        states = states[::subsample]

    env = jaxmarl.make("halfcheetah_2x3")
    slow_sys = _slow_sys_for_playback(env.sys, fps)
    h = html.render(slow_sys, states, height=height, colab=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(h)
    return out_path


# ---------------------------------------------------------------------------
# MP4 render (mujoco offscreen + imageio-ffmpeg)
# ---------------------------------------------------------------------------

def render_trajectory_mp4(
    traj_path: str,
    out_path: str,
    fps: float = 24.0,
    height: int = 360,
    width: int = 640,
    camera: str = "track",
    subsample: int = 1,
) -> str:
    """
    Offscreen-render each frame to an RGB image via brax's MuJoCo backend and
    stitch them into an MP4 at `fps`. `camera='track'` = built-in 2D side-view.
    """
    import imageio
    image_mod = _get_image_renderer()

    data = load_trajectory_pkl(traj_path)
    states = trajectory_to_states(data)
    if subsample > 1:
        states = states[::subsample]

    env = jaxmarl.make("halfcheetah_2x3")
    sys_ = env.sys

    try:
        frames = image_mod.render_array(
            sys_, states, height=height, width=width, camera=camera
        )
    except Exception:
        frames = image_mod.render_array(
            sys_, states, height=height, width=width, camera=None
        )

    if not isinstance(frames, list):
        frames = [np.asarray(frames)]
    else:
        frames = [np.asarray(fr) for fr in frames]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(
        out_path, frames, fps=float(fps), codec="libx264", quality=8,
        macro_block_size=1,
    )
    return out_path


# ---------------------------------------------------------------------------
# Filmstrip / snapshot PNG
# ---------------------------------------------------------------------------

def render_trajectory_snapshots(
    traj_path: str,
    out_path: str,
    num_snapshots: int = 8,
    height: int = 240,
    width: int = 320,
    camera: str = "track",
) -> str:
    """
    Pick N evenly-spaced frames, render each offscreen, and arrange them as a
    horizontal filmstrip PNG with time labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_mod = _get_image_renderer()

    data = load_trajectory_pkl(traj_path)
    states = trajectory_to_states(data)
    n = len(states)
    if num_snapshots < 2:
        num_snapshots = 2
    idx = np.linspace(0, n - 1, num_snapshots).astype(int)
    picked = [states[i] for i in idx]

    env = jaxmarl.make("halfcheetah_2x3")
    try:
        frames = image_mod.render_array(
            env.sys, picked, height=height, width=width, camera=camera
        )
    except Exception:
        frames = image_mod.render_array(
            env.sys, picked, height=height, width=width, camera=None
        )
    if not isinstance(frames, list):
        frames = [frames]

    fig, axes = plt.subplots(1, num_snapshots, figsize=(2.2 * num_snapshots, 2.8))
    if num_snapshots == 1:
        axes = [axes]
    for ax, frame, frame_idx in zip(axes, frames, idx):
        ax.imshow(np.asarray(frame))
        ax.set_title(f"step {int(frame_idx)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(os.path.basename(traj_path), fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Run-summary plot
# ---------------------------------------------------------------------------

def plot_run_summary(save_dir: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_path = os.path.join(save_dir, "run_data.pkl")
    with open(run_path, "rb") as f:
        run = pickle.load(f)

    payoff_history = run["payoff_history"]
    returns_history = run["returns_history"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    bests = [float(np.max(pm)) for pm in payoff_history]
    ax.plot(range(1, len(bests) + 1), bests, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("JPSRO outer iteration")
    ax.set_ylabel("max payoff in metagame matrix")
    ax.set_title("Best joint payoff vs iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if payoff_history:
        pm = np.array(payoff_history[-1])
        im = ax.imshow(pm, cmap="viridis", aspect="auto", origin="upper")
        n0, n1 = pm.shape
        ax.set_xticks(np.arange(n1))
        ax.set_yticks(np.arange(n0))
        ax.set_xticklabels([f"s{j}" for j in range(n1)])
        ax.set_yticklabels([f"s{i}" for i in range(n0)])
        ax.set_xlabel("Player 1 (front leg) strategy index ->")
        ax.set_ylabel("<- Player 0 (rear leg) strategy index")
        ax.set_title(
            "Final payoff G[i,j] = E[return | P0 plays strat i, P1 plays strat j]\n"
            "(pure joint policies, not a mixture)"
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(n0):
            for j in range(n1):
                c = "white" if pm[i, j] < pm.mean() else "black"
                ax.text(j, i, f"{pm[i, j]:.0f}",
                        ha="center", va="center", fontsize=7, color=c)

    ax = axes[1, 0]
    offset = 0
    for idx, ret in enumerate(returns_history):
        ret = np.asarray(ret).ravel()
        x = np.arange(offset, offset + len(ret))
        ax.plot(x, ret, alpha=0.7, label=f"BR block {idx}")
        offset += len(ret)
    ax.set_xlabel("Global BR update step (concatenated)")
    ax.set_ylabel("Mean rollout return (training envs)")
    ax.set_title("BR training returns (all players, all iterations)")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sizes = [pm.shape[0] for pm in payoff_history]
    ax.plot(range(1, len(sizes) + 1), sizes, "gs-", linewidth=2, markersize=8)
    ax.set_xlabel("JPSRO iteration")
    ax.set_ylabel("Number of strategies per player")
    ax.set_title("Population size (n x n metagame after each iter)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_checkpoint_summaries(save_dir: str, max_files: int = 20) -> None:
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"No checkpoints dir: {ckpt_dir}")
        return
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pkl"))[:max_files]
    for name in files:
        path = os.path.join(ckpt_dir, name)
        with open(path, "rb") as f:
            ck = pickle.load(f)
        pm = ck["payoff_matrix"]
        best = np.unravel_index(np.argmax(pm), pm.shape)
        print(
            f"{name}: iter={ck['iteration']}  payoff_shape={pm.shape}  "
            f"best_joint={best}  best_value={pm[best]:.2f}  "
            f"n_embeds=({len(ck['embeds_p0'])}, {len(ck['embeds_p1'])})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Visualize NeuPL-JPSRO cheetah artifacts")
    p.add_argument("--save-dir", type=str, default="results_neupl_cheetah",
                   help="Directory with trajectories/, checkpoints/, run_data.pkl")
    p.add_argument("--trajectory", type=str, default="",
                   help="Path to a single trajectory .pkl")
    p.add_argument("--all-trajectories", action="store_true",
                   help="Process every trajectories/*.pkl under save-dir")
    p.add_argument("--html-out-dir", type=str, default="",
                   help="Where to write HTML files (default: save-dir/viz_html)")
    p.add_argument("--mp4-out-dir", type=str, default="",
                   help="Where to write MP4 files (default: save-dir/viz_mp4)")
    p.add_argument("--snapshot-out-dir", type=str, default="",
                   help="Where to write filmstrip PNGs (default: save-dir/viz_snapshots)")

    # Playback / rendering controls
    p.add_argument("--fps", type=float, default=24.0,
                   help="Playback / encoding fps (HTML + MP4). Default 24. "
                        "Use 20 for 'real-time' physics (env.dt=0.05).")
    p.add_argument("--subsample", type=int, default=1,
                   help="Keep every k-th frame. 1 = all frames (default).")
    p.add_argument("--height", type=int, default=360, help="Frame height (pixels)")
    p.add_argument("--width", type=int, default=640, help="Frame width (pixels)")
    p.add_argument("--camera", type=str, default="track",
                   help="MuJoCo camera name for MP4/snapshots. 'track' = side-view. "
                        "Pass empty string to use default free camera.")
    p.add_argument("--mp4", action="store_true", help="Render MP4 video(s)")
    p.add_argument("--snapshots", type=int, default=0,
                   help="If >0, save an N-frame filmstrip PNG per trajectory")
    p.add_argument("--no-html", action="store_true",
                   help="Skip HTML rendering (useful with --mp4 alone)")

    p.add_argument("--run-summary", action="store_true",
                   help="Save payoff/BR summary figure from run_data.pkl")
    p.add_argument("--list-checkpoints", action="store_true",
                   help="Print one-line summary for each checkpoint .pkl")

    args = p.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    html_out = args.html_out_dir or os.path.join(save_dir, "viz_html")
    mp4_out = args.mp4_out_dir or os.path.join(save_dir, "viz_mp4")
    snap_out = args.snapshot_out_dir or os.path.join(save_dir, "viz_snapshots")
    camera = args.camera or None
    fps = max(1.0, float(args.fps))
    sub = max(1, int(args.subsample))

    if args.list_checkpoints:
        print_checkpoint_summaries(save_dir)

    if args.run_summary:
        out = os.path.join(save_dir, "viz_run_summary.png")
        plot_run_summary(save_dir, out)
        print(f"Saved run summary plot: {out}")

    def _process(traj_path: str):
        base = os.path.splitext(os.path.basename(traj_path))[0]

        if not args.no_html:
            out = os.path.join(html_out, f"{base}.html")
            render_trajectory_html(
                traj_path, out, subsample=sub, height=args.height, fps=fps,
            )
            print(f"  HTML  ({fps:.0f} fps): {out}")

        if args.mp4:
            out = os.path.join(mp4_out, f"{base}.mp4")
            render_trajectory_mp4(
                traj_path, out, fps=fps, height=args.height, width=args.width,
                camera=camera, subsample=sub,
            )
            print(f"  MP4   ({fps:.0f} fps, {args.width}x{args.height}, cam={camera}): {out}")

        if args.snapshots > 0:
            out = os.path.join(snap_out, f"{base}_strip.png")
            render_trajectory_snapshots(
                traj_path, out, num_snapshots=int(args.snapshots),
                height=max(180, args.height // 2),
                width=max(320, args.width // 2),
                camera=camera,
            )
            print(f"  STRIP ({args.snapshots} frames): {out}")

    if args.trajectory:
        print(f"Processing trajectory: {args.trajectory}")
        _process(args.trajectory)

    if args.all_trajectories:
        traj_dir = os.path.join(save_dir, "trajectories")
        if not os.path.isdir(traj_dir):
            print(f"No trajectories directory: {traj_dir}")
            return
        names = sorted(f for f in os.listdir(traj_dir) if f.endswith(".pkl"))
        print(f"Processing {len(names)} trajectories from {traj_dir}")
        for name in names:
            src = os.path.join(traj_dir, name)
            print(f"[{name}]")
            _process(src)

    if not any([
        args.trajectory, args.all_trajectories,
        args.run_summary, args.list_checkpoints,
    ]):
        p.print_help()
        print(
            "\nTip: read the payoff matrix as G[i,j] = expected return when "
            "P0 (rear) plays pure strategy i and P1 (front) plays pure strategy j. "
            "Mixed payoff = sum_{i,j} sigma(i,j) G[i,j]."
        )


if __name__ == "__main__":
    main()
