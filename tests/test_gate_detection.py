"""Tests for gate passage detection."""

import numpy as np
import pytest

from utils.math_utils import euler_to_rotation_matrix

from environments.gate_detection import (
    Gate,
    check_gate_frame_collision,
    check_gate_passage,
    compute_gate_relative_position,
)


def _make_gate(pos=(2, 0, 1), orientation=(0, 0, 0), size=(1, 1), frame_thickness=0.05):
    return Gate(
        id=0,
        position=np.array(pos, dtype=np.float64),
        orientation=np.array(orientation, dtype=np.float64),
        size=np.array(size, dtype=np.float64),
        frame_thickness=frame_thickness,
    )


def _cross_at(gate, lateral, vertical=0.0, backwards=False):
    """Fly a segment through the gate plane, offset by (lateral, vertical) in gate frame."""
    R = euler_to_rotation_matrix(gate.orientation)
    offset = R @ np.array([0.0, lateral, vertical])
    fwd = gate.forward_vector()
    a = gate.position + offset - fwd
    b = gate.position + offset + fwd
    return (b, a) if backwards else (a, b)


class TestCheckGatePassage:
    def test_straight_through(self):
        gate = _make_gate()
        prev = np.array([1.5, 0.0, 1.0])
        curr = np.array([2.5, 0.0, 1.0])
        assert check_gate_passage(prev, curr, gate) is True

    def test_miss_above(self):
        gate = _make_gate()
        prev = np.array([1.5, 0.0, 3.0])
        curr = np.array([2.5, 0.0, 3.0])
        assert check_gate_passage(prev, curr, gate) is False

    def test_miss_lateral(self):
        gate = _make_gate()
        prev = np.array([1.5, 2.0, 1.0])
        curr = np.array([2.5, 2.0, 1.0])
        assert check_gate_passage(prev, curr, gate) is False

    def test_backwards_rejected(self):
        gate = _make_gate()
        prev = np.array([2.5, 0.0, 1.0])
        curr = np.array([1.5, 0.0, 1.0])
        assert check_gate_passage(prev, curr, gate) is False

    def test_no_plane_crossing(self):
        gate = _make_gate()
        prev = np.array([1.0, 0.0, 1.0])
        curr = np.array([1.5, 0.0, 1.0])
        assert check_gate_passage(prev, curr, gate) is False

    def test_rotated_gate(self):
        gate = _make_gate(pos=(0, 2, 1), orientation=(0, 0, np.pi / 2))
        prev = np.array([0.0, 1.5, 1.0])
        curr = np.array([0.0, 2.5, 1.0])
        assert check_gate_passage(prev, curr, gate) is True

    def test_edge_margin(self):
        gate = _make_gate(size=(1, 1))
        prev = np.array([1.5, 0.49, 1.0])
        curr = np.array([2.5, 0.49, 1.0])
        assert check_gate_passage(prev, curr, gate) is True

    def test_just_outside(self):
        gate = _make_gate(size=(1, 1))
        prev = np.array([1.5, 0.6, 1.0])
        curr = np.array([2.5, 0.6, 1.0])
        assert check_gate_passage(prev, curr, gate) is False


class TestComputeGateRelativePosition:
    def test_in_front(self):
        gate = _make_gate(pos=(2, 0, 1))
        drone_pos = np.array([0.0, 0.0, 1.0])
        rel = compute_gate_relative_position(drone_pos, gate)
        assert rel[0] < 0  # drone is behind the gate (negative forward)

    def test_at_gate(self):
        gate = _make_gate(pos=(2, 0, 1))
        drone_pos = np.array([2.0, 0.0, 1.0])
        rel = compute_gate_relative_position(drone_pos, gate)
        np.testing.assert_allclose(rel, [0, 0, 0], atol=1e-10)


class TestGateFrameCollision:
    """A gate is a solid frame around its aperture, not an infinitely thin window.

    With size 1.0 and a 0.08 m drone radius the three bands are:
        d <= 0.42          clean pass
        0.42 < d <= 0.63   frame strike   (0.5 + 0.05 thickness + 0.08 radius)
        d  > 0.63          flew wide, no contact
    """

    RADIUS = 0.08

    def test_centre_pass_is_not_a_collision(self):
        gate = _make_gate()
        prev, curr = _cross_at(gate, 0.0)
        assert check_gate_frame_collision(prev, curr, gate, clearance=self.RADIUS) is False
        assert check_gate_passage(prev, curr, gate, clearance=self.RADIUS) is True

    def test_body_clipping_the_frame_is_a_collision(self):
        gate = _make_gate()
        prev, curr = _cross_at(gate, 0.46)  # centre inside the aperture, body is not
        assert check_gate_frame_collision(prev, curr, gate, clearance=self.RADIUS) is True
        assert check_gate_passage(prev, curr, gate, clearance=self.RADIUS) is False

    def test_flying_wide_touches_nothing(self):
        gate = _make_gate()
        prev, curr = _cross_at(gate, 1.5)
        assert check_gate_frame_collision(prev, curr, gate, clearance=self.RADIUS) is False
        assert check_gate_passage(prev, curr, gate, clearance=self.RADIUS) is False

    def test_vertical_offset_behaves_like_lateral(self):
        gate = _make_gate()
        prev, curr = _cross_at(gate, 0.0, vertical=0.46)
        assert check_gate_frame_collision(prev, curr, gate, clearance=self.RADIUS) is True

    def test_rotated_gate_uses_its_own_frame(self):
        """A world/gate frame mix-up would put this crossing in the wrong band."""
        gate = _make_gate(orientation=(0, 0, np.pi / 2))
        prev, curr = _cross_at(gate, 0.0)
        assert check_gate_frame_collision(prev, curr, gate, clearance=self.RADIUS) is False
        prev, curr = _cross_at(gate, 0.46)
        assert check_gate_frame_collision(prev, curr, gate, clearance=self.RADIUS) is True

    def test_hitting_the_frame_from_behind_still_counts(self):
        """Unlike passage, a frame strike is not direction-gated."""
        gate = _make_gate()
        prev, curr = _cross_at(gate, 0.46, backwards=True)
        assert check_gate_frame_collision(prev, curr, gate, clearance=self.RADIUS) is True
        assert check_gate_passage(prev, curr, gate, clearance=self.RADIUS) is False

    def test_no_plane_crossing_is_never_a_collision(self):
        gate = _make_gate()
        prev = np.array([0.0, 0.0, 1.0])
        curr = np.array([1.0, 0.0, 1.0])  # approaches but never reaches x=2
        assert check_gate_frame_collision(prev, curr, gate, clearance=self.RADIUS) is False

    def test_thicker_frame_widens_only_the_strike_band(self):
        thin, thick = _make_gate(frame_thickness=0.05), _make_gate(frame_thickness=0.40)
        prev, curr = _cross_at(thin, 0.85)  # outside thin's 0.63, inside thick's 0.98
        assert check_gate_frame_collision(prev, curr, thin, clearance=self.RADIUS) is False
        assert check_gate_frame_collision(prev, curr, thick, clearance=self.RADIUS) is True

    def test_pass_and_strike_are_mutually_exclusive(self):
        """The invariant that matters: no crossing may score +200 and -150 at once."""
        gate = _make_gate()
        for lateral in np.linspace(0.0, 0.8, 161):
            prev, curr = _cross_at(gate, float(lateral))
            passed = check_gate_passage(prev, curr, gate, clearance=self.RADIUS)
            struck = check_gate_frame_collision(prev, curr, gate, clearance=self.RADIUS)
            assert not (passed and struck), f"both fired at lateral={lateral:.3f}"

    def test_zero_clearance_matches_the_declared_aperture(self):
        """A point-mass drone passes right up to the aperture edge the track file declares."""
        gate = _make_gate()
        prev, curr = _cross_at(gate, 0.49)
        assert check_gate_passage(prev, curr, gate) is True
        assert check_gate_frame_collision(prev, curr, gate) is False
