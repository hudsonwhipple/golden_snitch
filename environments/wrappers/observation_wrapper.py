"""Observation wrapper that constructs the 24-dim state vector from BaseAviary state."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environments.gate_detection import Gate, compute_gate_relative_position


class RacingObservationWrapper(gym.ObservationWrapper):
    """Wraps a racing env to produce a flat 24-dim observation vector.

    State vector layout:
        [0:3]   position (x, y, z) - world frame
        [3:6]   velocity (vx, vy, vz) - body frame
        [6:9]   orientation (roll, pitch, yaw) - radians
        [9:12]  angular velocity (wx, wy, wz) - body frame
        [12:16] previous action (throttle, roll, pitch, yaw)
        [16:19] gate relative position (forward, lateral, vertical) - gate frame
        [19:27] gate keypoints normalized (u1,v1,...,u4,v4) - placeholder zeros in state mode
    """

    OBS_DIM = 24

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self._prev_action = np.zeros(4, dtype=np.float32)

    def observation(self, obs: dict) -> np.ndarray:
        """Convert structured observation dict to flat 24-dim vector."""
        flat = np.zeros(self.OBS_DIM, dtype=np.float32)

        flat[0:3] = obs["position"]
        flat[3:6] = obs["velocity"]
        flat[6:9] = obs["orientation"]
        flat[9:12] = obs["angular_velocity"]
        flat[12:16] = obs.get("previous_action", self._prev_action)
        flat[16:19] = obs.get("gate_relative_pos", np.zeros(3))
        flat[19:24] = obs.get("gate_detection_confidence_and_extra", np.zeros(5))

        return flat

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._prev_action = np.array(action, dtype=np.float32)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros(4, dtype=np.float32)
        return self.observation(obs), info
