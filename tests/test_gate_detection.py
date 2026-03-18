"""Tests for gate passage detection."""

import numpy as np
import pytest

from environments.gate_detection import Gate, check_gate_passage, compute_gate_relative_position


def _make_gate(pos=(2, 0, 1), orientation=(0, 0, 0), size=(1, 1)):
    return Gate(
        id=0,
        position=np.array(pos, dtype=np.float64),
        orientation=np.array(orientation, dtype=np.float64),
        size=np.array(size, dtype=np.float64),
    )


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
