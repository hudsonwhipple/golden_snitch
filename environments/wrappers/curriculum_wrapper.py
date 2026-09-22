"""Curriculum wrapper that modifies environment difficulty based on curriculum level."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from learning.training.curriculum import CurriculumLevel, OBS_NOISE_STD


class CurriculumWrapper(gym.Wrapper):
    """Wraps a DroneRacingEnv to apply curriculum-based difficulty scaling.

    Modifies:
    - Start-position perturbation
    - Observation noise
    - Number of active gates (subset of track; off by default, see CurriculumLevel)
    """

    def __init__(self, env: gym.Env, level: CurriculumLevel | None = None):
        super().__init__(env)
        self._level = level
        self._n_active_gates: int | None = None
        self._obs_noise_std: float = 0.0
        if level:
            self.set_level(level)

    def set_level(self, level: CurriculumLevel) -> None:
        """Apply a level to the underlying racing env.

        Everything is written straight to ``self.env.unwrapped``. Gymnasium 1.x removed
        ``Wrapper.__getattr__``, so reaching through an intermediate wrapper with
        ``hasattr(self.env, ...)`` silently returns False and the setting is dropped --
        that is exactly how PhysicsRandomizationWrapper became a no-op.
        """
        base = self.env.unwrapped
        self._level = level
        self._n_active_gates = level.n_gates if level.n_gates > 0 else None
        self._obs_noise_std = OBS_NOISE_STD if level.obs_noise else 0.0

        base._obs_noise_std = self._obs_noise_std
        base._initial_perturbation = level.start_perturbation
        # Always assign, including the None case. Only setting it for positive n_gates
        # left a stale count behind when promoting to an all-gates level, so levels above
        # the last gate-limited one silently kept running the shorter track.
        base._active_gate_count = (
            min(self._n_active_gates, len(base.track.gates))
            if self._n_active_gates is not None
            else None
        )

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._n_active_gates is not None:
            if self.env.unwrapped.racing_state.gates_passed >= self._n_active_gates:
                truncated = True

        return obs, reward, terminated, truncated, info
