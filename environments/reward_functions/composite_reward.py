"""Composite reward function for drone racing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    gate_bonus: float = 200.0
    collision_penalty: float = -150.0
    # Per metre closed toward the next gate. Gates are ~2.2 m apart on simple_oval, so
    # approaching one is worth ~22 against a 200 gate bonus: enough to guide the policy
    # between gates without competing with actually passing them.
    progress_weight: float = 10.0
    progress_clip: float = 1.0
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

        # Progress toward the next gate: metres closed this step, so the episode total
        # telescopes to (start distance - end distance). The previous form divided by
        # d_prev, which made it a *relative* measure: closing 0.1 m was worth 100x more
        # at 0.2 m from the gate than at 20 m, so it paid to hover near a gate rather
        # than fly the track, and a whole perfect approach summed to only ~4 against a
        # 200 gate bonus. The clip guards against a pathological jump, not normal flight
        # (at 5 m/s and a 1/48 s policy step a step closes at most ~0.10 m).
        d_prev = np.linalg.norm(prev_drone_pos - gate_pos)
        d_curr = np.linalg.norm(drone_pos - gate_pos)
        info.progress = self.cfg.progress_weight * float(
            np.clip(d_prev - d_curr, -self.cfg.progress_clip, self.cfg.progress_clip)
        )

        # Velocity along the gate direction (clipped to [-1, 1]). Note this is close to a
        # rescaling of the progress term above (progress ~= speed_toward * dt), so the two
        # add rather than measure different things -- tune them together, not separately.
        gate_dir = gate_pos - drone_pos
        gate_dist = np.linalg.norm(gate_dir)
        if gate_dist > 1e-6:
            gate_dir_norm = gate_dir / gate_dist
            speed_toward = np.dot(drone_vel, gate_dir_norm)
            raw_vel = speed_toward / self.cfg.max_velocity
            info.velocity = self.cfg.velocity_weight * float(np.clip(raw_vel, -1.0, 1.0))
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
