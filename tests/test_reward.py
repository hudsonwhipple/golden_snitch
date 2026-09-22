"""Tests for the composite reward function."""

import numpy as np
import pytest

from environments.reward_functions.composite_reward import CompositeReward, RewardConfig


@pytest.fixture
def reward():
    return CompositeReward(RewardConfig())


class TestCompositeReward:
    def test_gate_passage_bonus(self, reward):
        info = reward.compute(
            drone_pos=np.array([2.0, 0.0, 1.0]),
            prev_drone_pos=np.array([1.5, 0.0, 1.0]),
            drone_vel=np.array([1.0, 0.0, 0.0]),
            gate_pos=np.array([2.0, 0.0, 1.0]),
            action=np.zeros(4),
            prev_action=np.zeros(4),
            gate_passed=True,
            collided=False,
        )
        assert info.gate == RewardConfig().gate_bonus

    def test_collision_penalty(self, reward):
        info = reward.compute(
            drone_pos=np.array([0.0, 0.0, 0.0]),
            prev_drone_pos=np.array([0.0, 0.0, 0.5]),
            drone_vel=np.array([0.0, 0.0, -1.0]),
            gate_pos=np.array([2.0, 0.0, 1.0]),
            action=np.zeros(4),
            prev_action=np.zeros(4),
            gate_passed=False,
            collided=True,
        )
        assert info.collision == RewardConfig().collision_penalty
        assert info.total < 0

    def test_progress_positive_when_approaching(self, reward):
        info = reward.compute(
            drone_pos=np.array([1.5, 0.0, 1.0]),
            prev_drone_pos=np.array([1.0, 0.0, 1.0]),
            drone_vel=np.array([1.0, 0.0, 0.0]),
            gate_pos=np.array([2.0, 0.0, 1.0]),
            action=np.zeros(4),
            prev_action=np.zeros(4),
            gate_passed=False,
            collided=False,
        )
        assert info.progress > 0

    def test_progress_negative_when_moving_away(self, reward):
        info = reward.compute(
            drone_pos=np.array([0.5, 0.0, 1.0]),
            prev_drone_pos=np.array([1.0, 0.0, 1.0]),
            drone_vel=np.array([-1.0, 0.0, 0.0]),
            gate_pos=np.array([2.0, 0.0, 1.0]),
            action=np.zeros(4),
            prev_action=np.zeros(4),
            gate_passed=False,
            collided=False,
        )
        assert info.progress < 0

    def test_smoothness_penalty_on_jerky_action(self, reward):
        info = reward.compute(
            drone_pos=np.array([1.0, 0.0, 1.0]),
            prev_drone_pos=np.array([0.5, 0.0, 1.0]),
            drone_vel=np.array([1.0, 0.0, 0.0]),
            gate_pos=np.array([2.0, 0.0, 1.0]),
            action=np.array([1.0, 1.0, 1.0, 1.0]),
            prev_action=np.array([-1.0, -1.0, -1.0, -1.0]),
            gate_passed=False,
            collided=False,
        )
        assert info.smoothness < 0

    def test_time_penalty_always_negative(self, reward):
        info = reward.compute(
            drone_pos=np.array([1.0, 0.0, 1.0]),
            prev_drone_pos=np.array([1.0, 0.0, 1.0]),
            drone_vel=np.zeros(3),
            gate_pos=np.array([2.0, 0.0, 1.0]),
            action=np.zeros(4),
            prev_action=np.zeros(4),
            gate_passed=False,
            collided=False,
        )
        assert info.time < 0

    def test_velocity_reward_positive_toward_gate(self, reward):
        info = reward.compute(
            drone_pos=np.array([0.0, 0.0, 1.0]),
            prev_drone_pos=np.array([0.0, 0.0, 1.0]),
            drone_vel=np.array([3.0, 0.0, 0.0]),
            gate_pos=np.array([2.0, 0.0, 1.0]),
            action=np.zeros(4),
            prev_action=np.zeros(4),
            gate_passed=False,
            collided=False,
        )
        assert info.velocity > 0


class TestProgressIsTelescoping:
    """The progress term must measure metres closed, not a fraction of the remaining gap.

    A relative form -- ``(d_prev - d_curr) / d_prev`` -- pays far more for a small wiggle
    next to a gate than for real progress across the track, and lets a policy farm reward
    by oscillating. These guard both properties.
    """

    def test_reward_per_metre_is_independent_of_distance(self, reward):
        gate = np.array([10.0, 0.0, 1.0])
        step = 0.05

        def progress_at(distance):
            prev = gate - np.array([distance, 0.0, 0.0])
            curr = gate - np.array([distance - step, 0.0, 0.0])
            return reward.compute(
                drone_pos=curr, prev_drone_pos=prev, drone_vel=np.array([1.0, 0.0, 0.0]),
                gate_pos=gate, action=np.zeros(4), prev_action=np.zeros(4),
                gate_passed=False, collided=False,
            ).progress

        far, near = progress_at(8.0), progress_at(0.5)
        assert far == pytest.approx(near, rel=1e-6)

    def test_oscillating_nets_zero(self, reward):
        gate = np.array([10.0, 0.0, 1.0])
        a = np.array([2.0, 0.0, 1.0])
        b = np.array([2.4, 0.0, 1.0])

        def progress(prev, curr):
            return reward.compute(
                drone_pos=curr, prev_drone_pos=prev, drone_vel=np.zeros(3),
                gate_pos=gate, action=np.zeros(4), prev_action=np.zeros(4),
                gate_passed=False, collided=False,
            ).progress

        assert progress(a, b) + progress(b, a) == pytest.approx(0.0, abs=1e-9)

    def test_sum_telescopes_to_total_distance_closed(self, reward):
        gate = np.array([10.0, 0.0, 1.0])
        xs = np.linspace(0.0, 9.0, 200)
        total = sum(
            reward.compute(
                drone_pos=np.array([xs[i + 1], 0.0, 1.0]),
                prev_drone_pos=np.array([xs[i], 0.0, 1.0]),
                drone_vel=np.zeros(3), gate_pos=gate, action=np.zeros(4),
                prev_action=np.zeros(4), gate_passed=False, collided=False,
            ).progress
            for i in range(len(xs) - 1)
        )
        expected = reward.cfg.progress_weight * (
            np.linalg.norm(np.array([xs[0], 0.0, 1.0]) - gate)
            - np.linalg.norm(np.array([xs[-1], 0.0, 1.0]) - gate)
        )
        assert total == pytest.approx(expected, rel=1e-6)
