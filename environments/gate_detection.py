"""Gate passage detection using oriented plane intersection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.math_utils import euler_to_rotation_matrix, gate_forward_vector


DEFAULT_FRAME_THICKNESS = 0.05


@dataclass
class Gate:
    """A racing gate with position, orientation, and dimensions.

    ``size`` is the clear aperture. ``frame_thickness`` is the solid border around it, so the
    gate occupies the band between ``size/2`` and ``size/2 + frame_thickness`` in its own
    plane; beyond that there is nothing to hit.
    """

    id: int
    position: np.ndarray
    orientation: np.ndarray  # (roll, pitch, yaw) in radians
    size: np.ndarray  # (width, height) of the clear aperture
    frame_thickness: float = DEFAULT_FRAME_THICKNESS

    def forward_vector(self) -> np.ndarray:
        return gate_forward_vector(self.orientation)

    def world_to_local(self, point_world: np.ndarray) -> np.ndarray:
        """Project a world-frame point into the gate's local 2D frame (y_local, z_local)."""
        R = euler_to_rotation_matrix(self.orientation)
        local = R.T @ (point_world - self.position)
        return local[1:]  # y_local, z_local (drop x which is the through-axis)


def _plane_crossing(
    prev_pos: np.ndarray, curr_pos: np.ndarray, gate: Gate
) -> tuple[np.ndarray, bool] | tuple[None, bool]:
    """Where a swept segment crosses the gate's plane, and which way it went.

    Returns ``(local_offsets, forward)`` where ``local_offsets`` is the (lateral, vertical)
    offset of the crossing point in the gate's own frame and ``forward`` says the segment
    travelled in the gate's intended direction of travel. Returns ``(None, False)`` when the
    segment does not cross the plane at all.

    Interpolating the crossing point rather than sampling the endpoints is what stops the
    drone tunnelling: at 5 m/s and a 1/48 s policy step it moves ~0.10 m per step, twice the
    default frame thickness.
    """
    passage_dir = gate.forward_vector()
    d_prev = np.dot(prev_pos - gate.position, passage_dir)
    d_curr = np.dot(curr_pos - gate.position, passage_dir)

    if (d_prev <= 0) == (d_curr <= 0):
        return None, False

    t = d_prev / (d_prev - d_curr + 1e-12)
    crossing_point = prev_pos + t * (curr_pos - prev_pos)
    return gate.world_to_local(crossing_point), bool(d_prev <= 0 < d_curr)


def check_gate_passage(
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    gate: Gate,
    clearance: float = 0.0,
) -> bool:
    """Detect if the drone passed cleanly through a gate between two timesteps.

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
    clearance : body radius of the drone (meters). The aperture is *shrunk* by this, so a
        drone whose body overlaps the frame does not count as having passed. This used to
        be a ``margin`` that grew the aperture by 5 cm instead, which made every window 10%
        wider than its track file declared and -- once gates became solid -- let a single
        crossing score as a gate pass and a frame strike at the same time. Pass the same
        value to ``check_gate_frame_collision`` and the two become complementary.

    Returns
    -------
    True if the drone passed through the gate this step
    """
    local, forward = _plane_crossing(prev_pos, curr_pos, gate)
    if local is None or not forward:
        return False

    half_w = gate.size[0] / 2 - clearance
    half_h = gate.size[1] / 2 - clearance

    return bool(abs(local[0]) <= half_w and abs(local[1]) <= half_h)


def check_gate_frame_collision(
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    gate: Gate,
    clearance: float = 0.0,
) -> bool:
    """Detect if the drone struck a gate's frame between two timesteps.

    The gate plane is infinite but the frame is not, so a crossing is classified into three
    bands on ``d = max(|lateral|, |vertical|)`` of the crossing point::

        d <= size/2 - clearance                      clean pass
        size/2 - clearance < d <= size/2 + thickness + clearance   frame strike
        d > size/2 + thickness + clearance           flew wide, no contact

    The first band is exactly ``check_gate_passage``'s test, so a crossing is never both a
    pass and a strike. The third band is what keeps a drone flying several metres wide of a
    gate from registering a hit on its infinite plane.

    Unlike passage this is not direction-gated: flying into the back of a gate is still a
    crash.

    Known limitation: this only fires on a plane crossing, so a drone that grazes the outer
    edge of a frame while travelling parallel to it is not detected. That needs swept-volume
    contact against the frame solid, which the point-mass dynamics do not model.
    """
    local, _ = _plane_crossing(prev_pos, curr_pos, gate)
    if local is None:
        return False

    lateral, vertical = abs(local[0]), abs(local[1])
    inside_aperture = (
        lateral <= gate.size[0] / 2 - clearance and vertical <= gate.size[1] / 2 - clearance
    )
    if inside_aperture:
        return False

    outer_w = gate.size[0] / 2 + gate.frame_thickness + clearance
    outer_h = gate.size[1] / 2 + gate.frame_thickness + clearance
    return bool(lateral <= outer_w and vertical <= outer_h)


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
