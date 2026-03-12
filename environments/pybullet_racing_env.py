import numpy as np
import gym
from gym import spaces

class PyBulletRacingEnv(gym.Env):

    def __init__(self):

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

    def step(self, action):

        obs = np.zeros(12)

        reward = 0.0

        done = False

        info = {}

        return obs, reward, done, info

    def reset(self):

        return np.zeros(12)