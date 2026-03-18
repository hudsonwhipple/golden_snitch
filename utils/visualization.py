"""Visualization utilities for trajectories, metrics, and training curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_trajectory_overlay(
    trajectories: list[np.ndarray],
    gate_positions: np.ndarray,
    gate_orientations: np.ndarray,
    gate_sizes: np.ndarray,
    outcomes: list[str] | None = None,
    expert_trajectory: np.ndarray | None = None,
    title: str = "Flight Trajectories",
    save_path: str | Path | None = None,
) -> None:
    """Plot top-down (XY) and side (XZ) views of flight trajectories with gates.

    Parameters
    ----------
    trajectories : list of (T, 3) arrays of positions
    gate_positions : (N, 3) gate center positions
    gate_orientations : (N, 3) gate euler angles
    gate_sizes : (N, 2) gate widths and heights
    outcomes : per-trajectory outcome strings ("completed", "crashed", "timeout")
    expert_trajectory : optional (T, 3) reference trajectory
    title : plot title
    save_path : if provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    from utils.math_utils import euler_to_rotation_matrix

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=14)

    color_map = {"completed": "green", "crashed": "red", "timeout": "orange"}
    default_colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))

    for i, traj in enumerate(trajectories):
        outcome = outcomes[i] if outcomes else None
        color = color_map.get(outcome, default_colors[i])
        alpha = 0.6 if len(trajectories) > 5 else 0.9
        ax_xy.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=0.8)
        ax_xz.plot(traj[:, 0], traj[:, 2], color=color, alpha=alpha, linewidth=0.8)

        if outcome == "crashed":
            ax_xy.scatter(traj[-1, 0], traj[-1, 1], color="red", marker="x", s=50, zorder=5)
            ax_xz.scatter(traj[-1, 0], traj[-1, 2], color="red", marker="x", s=50, zorder=5)

    if expert_trajectory is not None:
        ax_xy.plot(expert_trajectory[:, 0], expert_trajectory[:, 1], "k--", linewidth=1.5, label="Expert")
        ax_xz.plot(expert_trajectory[:, 0], expert_trajectory[:, 2], "k--", linewidth=1.5, label="Expert")

    for i in range(len(gate_positions)):
        gp = gate_positions[i]
        R = euler_to_rotation_matrix(gate_orientations[i])
        hw, hh = gate_sizes[i, 0] / 2, gate_sizes[i, 1] / 2
        corners_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])

        gate_xy = np.zeros((5, 2))
        gate_xz = np.zeros((5, 2))
        for j in range(5):
            corner_3d = gp + R @ np.array([0, corners_local[j, 0], corners_local[j, 1]])
            gate_xy[j] = [corner_3d[0], corner_3d[1]]
            gate_xz[j] = [corner_3d[0], corner_3d[2]]

        ax_xy.plot(gate_xy[:, 0], gate_xy[:, 1], "b-", linewidth=2)
        ax_xz.plot(gate_xz[:, 0], gate_xz[:, 1], "b-", linewidth=2)

        fwd = R[:, 0] * 0.3
        ax_xy.annotate("", xy=(gp[0] + fwd[0], gp[1] + fwd[1]), xytext=(gp[0], gp[1]),
                        arrowprops=dict(arrowstyle="->", color="blue"))

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Top-Down View (XY)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)

    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("Side View (XZ)")
    ax_xz.grid(True, alpha=0.3)

    if expert_trajectory is not None:
        ax_xy.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reward_breakdown(
    reward_components: dict[str, list[float]],
    title: str = "Reward Component Breakdown",
    save_path: str | Path | None = None,
) -> None:
    """Stacked bar chart of reward components per episode."""
    import matplotlib.pyplot as plt

    episodes = range(len(next(iter(reward_components.values()))))
    fig, ax = plt.subplots(figsize=(12, 5))

    bottom = np.zeros(len(episodes))
    colors = plt.cm.Set2(np.linspace(0, 1, len(reward_components)))

    for (name, values), color in zip(reward_components.items(), colors):
        vals = np.array(values)
        ax.bar(episodes, vals, bottom=bottom, label=name, color=color, alpha=0.8)
        bottom += vals

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curriculum_progress(
    steps: list[int],
    gate_success_rates: list[float],
    curriculum_levels: list[int],
    promotions: list[int] | None = None,
    title: str = "Curriculum Training Progress",
    save_path: str | Path | None = None,
) -> None:
    """Timeline plot of gate success rate and curriculum level progression."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(steps, gate_success_rates, "b-", linewidth=1.5, label="Gate Success Rate")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Gate Success Rate", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.step(steps, curriculum_levels, "r-", linewidth=2, alpha=0.7, label="Curriculum Level", where="post")
    ax2.set_ylabel("Curriculum Level", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    if promotions:
        for step in promotions:
            ax1.axvline(x=step, color="green", linestyle="--", alpha=0.5, linewidth=1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    name_a: str = "Model A",
    name_b: str = "Model B",
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Side-by-side bar chart comparing two models on shared metrics."""
    import matplotlib.pyplot as plt

    shared_keys = [k for k in metrics_a if k in metrics_b]
    x = np.arange(len(shared_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [metrics_a[k] for k in shared_keys], width, label=name_a)
    ax.bar(x + width / 2, [metrics_b[k] for k in shared_keys], width, label=name_b)

    ax.set_xticks(x)
    ax.set_xticklabels(shared_keys, rotation=30, ha="right")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
