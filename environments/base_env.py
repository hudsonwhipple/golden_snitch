import gym

class BaseDroneEnv(gym.Env):

    def __init__(self):
        super().__init__()

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError