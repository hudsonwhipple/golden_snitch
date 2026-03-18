"""Minimum-jerk trajectory generation through gate waypoints."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def generate_min_jerk_trajectory(
    waypoints: np.ndarray,
    total_time: float | None = None,
    dt: float = 1.0 / 48,
    avg_speed: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth trajectory through a sequence of 3D waypoints.

    Uses cubic spline interpolation (approximating minimum-jerk behavior)
    with time allocation proportional to segment length.

    Parameters
    ----------
    waypoints : (N, 3) array of positions to pass through
    total_time : override total trajectory duration (seconds)
    dt : timestep for trajectory sampling
    avg_speed : average speed for automatic time allocation (m/s)

    Returns
    -------
    positions : (T, 3) sampled positions along trajectory
    velocities : (T, 3) sampled velocities
    timestamps : (T,) time values
    """
    n_waypoints = len(waypoints)
    if n_waypoints < 2:
        raise ValueError("Need at least 2 waypoints")

    segments = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    total_length = np.sum(segment_lengths)

    if total_time is None:
        total_time = total_length / avg_speed

    cumulative = np.zeros(n_waypoints)
    cumulative[1:] = np.cumsum(segment_lengths)
    t_waypoints = cumulative / total_length * total_time

    cs_x = CubicSpline(t_waypoints, waypoints[:, 0], bc_type="clamped")
    cs_y = CubicSpline(t_waypoints, waypoints[:, 1], bc_type="clamped")
    cs_z = CubicSpline(t_waypoints, waypoints[:, 2], bc_type="clamped")

    n_steps = int(total_time / dt)
    timestamps = np.linspace(0, total_time, n_steps)

    positions = np.column_stack([cs_x(timestamps), cs_y(timestamps), cs_z(timestamps)])
    velocities = np.column_stack([cs_x(timestamps, 1), cs_y(timestamps, 1), cs_z(timestamps, 1)])

    return positions, velocities, timestamps


def generate_racing_trajectory(
    gate_positions: np.ndarray,
    start_position: np.ndarray,
    avg_speed: float = 2.0,
    dt: float = 1.0 / 48,
    approach_distance: float = 0.5,
    gate_forward_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a racing trajectory through gates with approach waypoints.

    Adds waypoints slightly before each gate to ensure smooth gate entry.

    Parameters
    ----------
    gate_positions : (N, 3) gate center positions
    start_position : (3,) starting position
    avg_speed : target average speed (m/s)
    dt : timestep
    approach_distance : distance before gate to add approach waypoint
    gate_forward_vectors : (N, 3) optional forward vectors for approach direction

    Returns
    -------
    positions, velocities, timestamps
    """
    waypoints = [start_position.copy()]

    for i, gate_pos in enumerate(gate_positions):
        if gate_forward_vectors is not None:
            approach = gate_pos - gate_forward_vectors[i] * approach_distance
            waypoints.append(approach)
        waypoints.append(gate_pos.copy())

    waypoints = np.array(waypoints)
    return generate_min_jerk_trajectory(waypoints, dt=dt, avg_speed=avg_speed)
