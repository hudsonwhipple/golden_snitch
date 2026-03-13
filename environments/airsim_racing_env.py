import numpy as np
import gymnasium as gym

try:
    import airsim
except ImportError:
    airsim = None


class AirSimRacingEnv(gym.Env):
    def __init__(self):
        if airsim is None:
            raise ImportError("Install AirSim client: pip install --no-build-isolation -r requirements-airsim.txt")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def step(self, action):
        vx, vy, vz, yaw = action
        self.client.moveByVelocityAsync(vx, vy, vz, 0.1).join()
        state = self.client.getMultirotorState()
        obs = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val,
        ], dtype=np.float32)
        reward = -float(abs(obs[2]))
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        return np.zeros(3, dtype=np.float32), {}