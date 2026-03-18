import gymnasium as gym

gym.register(
    id="DroneRacing-v0",
    entry_point="environments.racing_env:DroneRacingEnv",
)

try:
    gym.register(
        id="DroneRacingPyBullet-v0",
        entry_point="environments.pybullet_racing_env:PyBulletRacingEnv",
    )
except Exception:
    pass
