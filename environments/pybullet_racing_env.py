import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PyBulletRacingEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

    def step(self, action):
        obs = np.zeros(12, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(12, dtype=np.float32), {}