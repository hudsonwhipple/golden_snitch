"""Replay recorded trajectories with matplotlib visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def replay_trajectory(
    trajectories_path: str,
    track_name: str = "simple_oval",
    episode_idx: int = 0,
) -> None:
    """Replay a recorded trajectory with matplotlib animation."""
    from environments.tracks.track_loader import load_track
    from utils.visualization import plot_trajectory_overlay
    from utils.math_utils import euler_to_rotation_matrix

    data = np.load(trajectories_path, allow_pickle=True)
    positions = data["positions"]
    outcomes = data["outcomes"]

    print(f"Loaded {len(positions)} episodes")
    print(f"Outcomes: {dict(zip(*np.unique(outcomes, return_counts=True)))}")

    track = load_track(track_name)
    gate_positions = np.array([g.position for g in track.gates])
    gate_orientations = np.array([g.orientation for g in track.gates])
    gate_sizes = np.array([g.size for g in track.gates])

    valid_trajs = []
    valid_outcomes = []
    for i in range(min(10, len(positions))):
        traj = positions[i]
        mask = ~np.isnan(traj[:, 0])
        if mask.sum() > 1:
            valid_trajs.append(traj[mask])
            valid_outcomes.append(str(outcomes[i]))

    if valid_trajs:
        plot_trajectory_overlay(
            trajectories=valid_trajs,
            gate_positions=gate_positions,
            gate_orientations=gate_orientations,
            gate_sizes=gate_sizes,
            outcomes=valid_outcomes,
            title=f"Trajectory Replay - {track_name}",
            save_path=f"outputs/replay_{track_name}.png",
        )
        print(f"Plot saved to outputs/replay_{track_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay recorded trajectories")
    parser.add_argument("--trajectories", required=True, help="Path to trajectories.npz")
    parser.add_argument("--track", default="simple_oval")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    replay_trajectory(args.trajectories, args.track, args.episode)
