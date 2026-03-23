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
    stats: dict | None = None,
) -> None:
    """Plot top-down (XY) and side (XZ) views of ALL flight trajectories with gates.

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
    stats : optional dict of summary statistics to annotate on the plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from utils.math_utils import euler_to_rotation_matrix

    n_traj = len(trajectories)
    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    color_map = {"completed": "green", "crashed": "red", "timeout": "orange"}
    alpha = max(0.15, min(0.8, 5.0 / max(n_traj, 1)))

    outcome_counts = {"completed": 0, "crashed": 0, "timeout": 0}

    for i, traj in enumerate(trajectories):
        outcome = outcomes[i] if outcomes else "timeout"
        color = color_map.get(outcome, "gray")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        ax_xy.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=0.7)
        ax_xz.plot(traj[:, 0], traj[:, 2], color=color, alpha=alpha, linewidth=0.7)

        ax_xy.scatter(traj[0, 0], traj[0, 1], color="green", marker="o", s=12, alpha=0.4, zorder=4)
        ax_xz.scatter(traj[0, 0], traj[0, 2], color="green", marker="o", s=12, alpha=0.4, zorder=4)

        if outcome == "crashed":
            ax_xy.scatter(traj[-1, 0], traj[-1, 1], color="red", marker="x", s=30, alpha=0.5, zorder=5)
            ax_xz.scatter(traj[-1, 0], traj[-1, 2], color="red", marker="x", s=30, alpha=0.5, zorder=5)

    if expert_trajectory is not None:
        ax_xy.plot(expert_trajectory[:, 0], expert_trajectory[:, 1], "k--", linewidth=2, label="Expert", zorder=6)
        ax_xz.plot(expert_trajectory[:, 0], expert_trajectory[:, 2], "k--", linewidth=2, label="Expert", zorder=6)

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

        ax_xy.plot(gate_xy[:, 0], gate_xy[:, 1], "b-", linewidth=2.5, zorder=7)
        ax_xz.plot(gate_xz[:, 0], gate_xz[:, 1], "b-", linewidth=2.5, zorder=7)

        fwd = R[:, 0] * 0.3
        ax_xy.annotate("", xy=(gp[0] + fwd[0], gp[1] + fwd[1]), xytext=(gp[0], gp[1]),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

        ax_xy.text(gp[0], gp[1] + 0.15, f"G{i}", fontsize=8, ha="center",
                   fontweight="bold", color="darkblue", zorder=8)
        ax_xz.text(gp[0], gp[2] + 0.15, f"G{i}", fontsize=8, ha="center",
                   fontweight="bold", color="darkblue", zorder=8)

    # Legend with outcome counts
    legend_elements = []
    for outcome, color in color_map.items():
        count = outcome_counts.get(outcome, 0)
        if count > 0:
            legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=f"{outcome} ({count})"))
    legend_elements.append(Line2D([0], [0], color="green", marker="o", linestyle="None",
                                  markersize=5, label="Start"))
    if expert_trajectory is not None:
        legend_elements.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Expert"))
    ax_xy.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    # Stats annotation
    if stats:
        stats_text = "\n".join([
            f"Episodes: {stats.get('n_episodes', n_traj)}",
            f"Gate Success: {stats.get('gate_success_rate', 0):.1%}",
            f"Crash Rate: {stats.get('crash_rate', 0):.1%}",
            f"Avg Gates: {stats.get('avg_gates_passed', 0):.1f}",
            f"Mean Steps: {stats.get('mean_episode_steps', 0):.0f}",
        ])
        ax_xz.text(0.02, 0.98, stats_text, transform=ax_xz.transAxes, fontsize=8,
                   verticalalignment="top", fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Top-Down View (XY)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)

    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("Side View (XZ)")
    ax_xz.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_episode_statistics(
    episodes_data: list[dict],
    title: str = "Episode Statistics",
    save_path: str | Path | None = None,
) -> None:
    """Plot histograms and scatter of per-episode statistics.

    Parameters
    ----------
    episodes_data : list of dicts with keys: gates_passed, steps, total_reward, outcome
    title : plot title
    save_path : if provided, save figure to this path
    """
    import matplotlib.pyplot as plt

    gates = [e["gates_passed"] for e in episodes_data]
    steps = [e["steps"] for e in episodes_data]
    rewards = [e["total_reward"] for e in episodes_data]
    outcomes = [e["outcome"] for e in episodes_data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Histogram of episode lengths
    ax = axes[0, 0]
    color_map = {"completed": "green", "crashed": "red", "timeout": "orange"}
    for outcome in ["completed", "crashed", "timeout"]:
        subset = [s for s, o in zip(steps, outcomes) if o == outcome]
        if subset:
            ax.hist(subset, bins=30, alpha=0.6, color=color_map[outcome], label=outcome)
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Histogram of gates passed
    ax = axes[0, 1]
    max_gates = max(gates) if gates else 0
    bins = range(0, max_gates + 2)
    ax.hist(gates, bins=bins, color="steelblue", edgecolor="white", alpha=0.8, align="left")
    ax.set_xlabel("Gates Passed")
    ax.set_ylabel("Count")
    ax.set_title("Gates Passed Distribution")
    ax.set_xticks(range(0, max_gates + 1))
    ax.grid(True, alpha=0.3)

    # Histogram of total rewards
    ax = axes[1, 0]
    for outcome in ["completed", "crashed", "timeout"]:
        subset = [r for r, o in zip(rewards, outcomes) if o == outcome]
        if subset:
            ax.hist(subset, bins=30, alpha=0.6, color=color_map[outcome], label=outcome)
    ax.set_xlabel("Total Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scatter: episode length vs gates passed
    ax = axes[1, 1]
    for outcome in ["completed", "crashed", "timeout"]:
        s_sub = [s for s, o in zip(steps, outcomes) if o == outcome]
        g_sub = [g for g, o in zip(gates, outcomes) if o == outcome]
        if s_sub:
            ax.scatter(s_sub, g_sub, c=color_map[outcome], alpha=0.5, s=20, label=outcome)
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Gates Passed")
    ax.set_title("Episode Length vs Gates Passed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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
    """Bar chart of cumulative reward components per episode."""
    import matplotlib.pyplot as plt

    n_episodes = len(next(iter(reward_components.values())))
    episodes = range(n_episodes)
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(16, 6),
                                          gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(title, fontsize=14, fontweight="bold")

    component_colors = {
        "gate": "#2ecc71", "progress": "#3498db", "velocity": "#9b59b6",
        "smoothness": "#f39c12", "collision": "#e74c3c", "time": "#95a5a6",
    }

    # Stacked bar chart
    bottom_pos = np.zeros(n_episodes)
    bottom_neg = np.zeros(n_episodes)
    labeled = set()
    for name, values in reward_components.items():
        vals = np.array(values)
        color = component_colors.get(name, "gray")
        pos_vals = np.maximum(vals, 0)
        neg_vals = np.minimum(vals, 0)
        label = name if name not in labeled else None
        if np.any(pos_vals > 0):
            ax_bar.bar(episodes, pos_vals, bottom=bottom_pos, label=label, color=color, alpha=0.8)
            labeled.add(name)
            bottom_pos += pos_vals
        if np.any(neg_vals < 0):
            lbl = name if name not in labeled else None
            ax_bar.bar(episodes, neg_vals, bottom=bottom_neg, label=lbl, color=color, alpha=0.8)
            labeled.add(name)
            bottom_neg += neg_vals

    ax_bar.set_xlabel("Episode")
    ax_bar.set_ylabel("Cumulative Reward")
    ax_bar.set_title("Per-Episode Reward Breakdown")
    ax_bar.legend(loc="upper right", fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.axhline(y=0, color="black", linewidth=0.5)

    # Pie chart of mean absolute contributions
    means = {name: float(np.mean(np.abs(values))) for name, values in reward_components.items()}
    total = sum(means.values())
    if total > 0:
        labels = [f"{k}\n({v:.1f})" for k, v in means.items()]
        colors = [component_colors.get(k, "gray") for k in means.keys()]
        ax_pie.pie(means.values(), labels=labels, colors=colors, autopct="%1.0f%%",
                   textprops={"fontsize": 8})
        ax_pie.set_title("Mean |Reward| Share")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speed_altitude_profiles(
    trajectories: list[np.ndarray],
    velocities: list[np.ndarray],
    outcomes: list[str] | None = None,
    dt: float = 1.0 / 48,
    max_traces: int = 20,
    title: str = "Speed & Altitude Profiles",
    save_path: str | Path | None = None,
) -> None:
    """Plot speed and altitude over time for sampled episodes.

    Parameters
    ----------
    trajectories : list of (T, 3) position arrays
    velocities : list of (T, 3) velocity arrays
    outcomes : per-trajectory outcome strings
    dt : policy timestep in seconds
    max_traces : max number of episodes to plot
    title : plot title
    save_path : if provided, save figure to this path
    """
    import matplotlib.pyplot as plt

    n = min(max_traces, len(trajectories))
    color_map = {"completed": "green", "crashed": "red", "timeout": "orange"}

    fig, (ax_speed, ax_alt) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    alpha = max(0.2, min(0.8, 5.0 / max(n, 1)))

    for i in range(n):
        traj = trajectories[i]
        vel = velocities[i]
        outcome = outcomes[i] if outcomes else "timeout"
        color = color_map.get(outcome, "gray")
        t = np.arange(len(traj)) * dt

        speeds = np.linalg.norm(vel, axis=1)
        ax_speed.plot(t, speeds, color=color, alpha=alpha, linewidth=0.8)
        ax_alt.plot(t, traj[:, 2], color=color, alpha=alpha, linewidth=0.8)

    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_title("Speed Over Time")
    ax_speed.grid(True, alpha=0.3)

    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Altitude Over Time")
    ax_alt.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Ground")
    ax_alt.legend(fontsize=8)
    ax_alt.grid(True, alpha=0.3)

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
