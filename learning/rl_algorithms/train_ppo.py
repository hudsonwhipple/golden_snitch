from stable_baselines3 import PPO
from environments.pybullet_racing_env import PyBulletRacingEnv

env = PyBulletRacingEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=1_000_000
)

model.save("ppo_drone_policy")