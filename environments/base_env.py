import gymnasium as gym


class BaseDroneEnv(gym.Env):
    def __init__(self):
        super().__init__()

    def step(self, action):
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raise NotImplementedError