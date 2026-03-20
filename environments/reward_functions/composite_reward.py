"""Composite reward function for drone racing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    gate_bonus: float = 200.0
    collision_penalty: float = -150.0
    progress_weight: float = 1.0
    velocity_weight: float = 0.3
    smooth_weight: float = 0.05
    time_penalty: float = -0.01
    max_velocity: float = 5.0


@dataclass
class RewardInfo:
    """Breakdown of reward components for logging."""

    total: float = 0.0
    gate: float = 0.0
    progress: float = 0.0
    velocity: float = 0.0
    smoothness: float = 0.0
    collision: float = 0.0
    time: float = 0.0


class CompositeReward:
    """Multi-component reward for drone racing training.

    Components:
    - gate: sparse bonus for passing through a gate
    - progress: dense signal for approaching the next gate
    - velocity: reward for speed toward the gate
    - smoothness: penalty for jerky controls
    - collision: terminal penalty for crashing
    - time: small negative per step to incentivize speed
    """

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()

    @classmethod
    def from_dict(cls, d: dict) -> CompositeReward:
        return cls(RewardConfig(**{k: v for k, v in d.items() if k in RewardConfig.__dataclass_fields__}))

    def compute(
        self,
        drone_pos: np.ndarray,
        prev_drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        gate_pos: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        gate_passed: bool,
        collided: bool,
    ) -> RewardInfo:
        info = RewardInfo()

        # Gate passage (sparse)
        info.gate = self.cfg.gate_bonus if gate_passed else 0.0

        # Progress toward next gate (dense)
        d_prev = np.linalg.norm(prev_drone_pos - gate_pos)
        d_curr = np.linalg.norm(drone_pos - gate_pos)
        if d_prev > 1e-6:
            info.progress = self.cfg.progress_weight * (d_prev - d_curr) / d_prev
        else:
            info.progress = 0.0

        # Velocity along gate direction
        gate_dir = gate_pos - drone_pos
        gate_dist = np.linalg.norm(gate_dir)
        if gate_dist > 1e-6:
            gate_dir_norm = gate_dir / gate_dist
            speed_toward = np.dot(drone_vel, gate_dir_norm)
            info.velocity = self.cfg.velocity_weight * speed_toward / self.cfg.max_velocity
        else:
            info.velocity = 0.0

        # Smoothness penalty (normalized to [-1, 0])
        action_delta = action - prev_action
        info.smoothness = self.cfg.smooth_weight * (-np.sum(action_delta**2) / 16.0)

        # Collision (terminal)
        info.collision = self.cfg.collision_penalty if collided else 0.0

        # Time penalty
        info.time = self.cfg.time_penalty

        info.total = (
            info.gate + info.progress + info.velocity + info.smoothness + info.collision + info.time
        )
        return info
