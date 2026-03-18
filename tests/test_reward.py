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
        assert info.gate == 10.0

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
        assert info.collision == -50.0
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
