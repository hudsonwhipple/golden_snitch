"""Gate passage detection using oriented plane intersection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.math_utils import euler_to_rotation_matrix, gate_forward_vector


@dataclass
class Gate:
    """A racing gate with position, orientation, and dimensions."""

    id: int
    position: np.ndarray
    orientation: np.ndarray  # (roll, pitch, yaw) in radians
    size: np.ndarray  # (width, height)

    def forward_vector(self) -> np.ndarray:
        return gate_forward_vector(self.orientation)

    def world_to_local(self, point_world: np.ndarray) -> np.ndarray:
        """Project a world-frame point into the gate's local 2D frame (y_local, z_local)."""
        R = euler_to_rotation_matrix(self.orientation)
        local = R.T @ (point_world - self.position)
        return local[1:]  # y_local, z_local (drop x which is the through-axis)


def check_gate_passage(
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    gate: Gate,
    margin: float = 0.05,
) -> bool:
    """Detect if the drone passed through a gate between two timesteps.

    Uses oriented plane intersection with direction validation.
    The drone must cross the gate plane from the approach side (negative dot
    product with passage_direction) to the exit side (positive dot product).
    See ``gate_forward_vector`` for the convention: the gate's +X axis points
    in the intended direction of travel.

    Parameters
    ----------
    prev_pos : (3,) previous drone position
    curr_pos : (3,) current drone position
    gate : Gate object
    margin : extra margin around gate bounds (meters)

    Returns
    -------
    True if the drone passed through the gate this step
    """
    passage_dir = gate.forward_vector()

    # Signed distances to the gate plane along the passage direction.
    # Negative = approach side, Positive = exit side.
    d_prev = np.dot(prev_pos - gate.position, passage_dir)
    d_curr = np.dot(curr_pos - gate.position, passage_dir)

    crossed_plane = (d_prev <= 0) and (d_curr > 0)
    if not crossed_plane:
        return False

    t = d_prev / (d_prev - d_curr + 1e-12)
    crossing_point = prev_pos + t * (curr_pos - prev_pos)

    local = gate.world_to_local(crossing_point)
    half_w = gate.size[0] / 2 + margin
    half_h = gate.size[1] / 2 + margin

    return bool(abs(local[0]) <= half_w and abs(local[1]) <= half_h)


def compute_gate_relative_position(
    drone_pos: np.ndarray,
    gate: Gate,
) -> np.ndarray:
    """Compute drone position relative to the gate in the gate's local frame.

    Returns (3,) vector: [forward_distance, lateral_offset, vertical_offset].
    """
    R = euler_to_rotation_matrix(gate.orientation)
    return R.T @ (drone_pos - gate.position)


def distance_to_gate(drone_pos: np.ndarray, gate: Gate) -> float:
    """Euclidean distance from drone to gate center."""
    return float(np.linalg.norm(drone_pos - gate.position))
