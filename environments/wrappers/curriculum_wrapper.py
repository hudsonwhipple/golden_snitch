"""Curriculum wrapper that modifies environment difficulty based on curriculum level."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from learning.training.curriculum import CurriculumLevel


class CurriculumWrapper(gym.Wrapper):
    """Wraps a DroneRacingEnv to apply curriculum-based difficulty scaling.

    Modifies:
    - Number of active gates (subset of track)
    - Observation noise
    """

    def __init__(self, env: gym.Env, level: CurriculumLevel | None = None):
        super().__init__(env)
        self._level = level
        self._n_active_gates: int | None = None
        self._obs_noise_std: float = 0.0
        if level:
            self.set_level(level)

    def set_level(self, level: CurriculumLevel) -> None:
        self._level = level
        self._n_active_gates = level.n_gates if level.n_gates > 0 else None
        self._obs_noise_std = 0.02 if level.obs_noise else 0.0
        if hasattr(self.env, '_obs_noise_std'):
            self.env._obs_noise_std = self._obs_noise_std

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        if self._n_active_gates is not None and hasattr(self.env, 'track'):
            all_gates = self.env.track.gates
            n = min(self._n_active_gates, len(all_gates))
            self.env._active_gate_count = n
        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._n_active_gates is not None and hasattr(self.env, 'racing_state'):
            n = self._n_active_gates
            if self.env.racing_state.gates_passed >= n:
                truncated = True

        return obs, reward, terminated, truncated, info
