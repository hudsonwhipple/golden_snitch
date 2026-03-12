import airsim
import gym
import numpy as np

class AirSimRacingEnv(gym.Env):

    def __init__(self):

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def step(self, action):

        vx, vy, vz, yaw = action

        self.client.moveByVelocityAsync(
            vx, vy, vz, 0.1
        ).join()

        state = self.client.getMultirotorState()

        obs = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])

        reward = -abs(obs[2])

        done = False

        return obs, reward, done, {}

    def reset(self):

        self.client.reset()

        return np.zeros(3)