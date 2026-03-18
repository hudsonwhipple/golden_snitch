"""AirSim racing environment for high-fidelity validation and vision training.

Requires:
- AirSim binary running (see simulators/airsim/)
- pip install -r requirements-airsim.txt

This environment provides:
- Full RGB rendering for vision pipeline training
- Higher-fidelity dynamics than the lightweight sim
- Same observation/action interface as DroneRacingEnv for direct comparison
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environments.base_env import OBS_DIM, ACT_DIM, RacingMixin
from environments.tracks.track_loader import load_track
from utils.math_utils import world_to_body

try:
    import airsim
except ImportError:
    airsim = None


class AirSimRacingEnv(gym.Env, RacingMixin):
    """AirSim-based racing environment for validation and vision training.

    Provides the same interface as DroneRacingEnv (24-dim obs, 4-dim action)
    but with full rendering capabilities for Phase 2 training.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        track_name: str = "simple_oval",
        max_episode_steps: int = 1000,
        reward_cfg: dict | None = None,
        render_mode: str | None = None,
        image_size: tuple[int, int] = (128, 96),
    ):
        super().__init__()

        if airsim is None:
            raise ImportError(
                "AirSim not installed. Run: pip install -r requirements-airsim.txt\n"
                "Also ensure the AirSim binary is running."
            )

        self.render_mode = render_mode
        self.image_size = image_size
        self._max_velocity = 5.0

        track = load_track(track_name)
        self._init_racing(track, reward_cfg)
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self._last_image = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_racing()

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        start = self.track.start_position
        self.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(float(start[0]), float(start[1]), float(-start[2])),
                airsim.to_quaternion(0, 0, float(self.track.start_orientation[2])),
            ),
            ignore_collision=True,
        )

        self.client.takeoffAsync().join()

        self.racing_state.prev_pos = start.copy()
        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()[:ACT_DIM]
        self.racing_state.step_count += 1

        vx = float(action[0] * self._max_velocity)
        vy = float(action[1] * self._max_velocity)
        vz = float(-action[2] * self._max_velocity)
        yaw_rate = float(action[3] * np.pi)

        self.client.moveByVelocityAsync(vx, vy, vz, duration=1.0 / 48).join()

        state = self.client.getMultirotorState()
        pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            -state.kinematics_estimated.position.z_val,
        ])
        vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            -state.kinematics_estimated.linear_velocity.z_val,
        ])

        gate_passed = self._update_gate_progress(self.racing_state.prev_pos, pos)
        collided = bool(self.client.simGetCollisionInfo().has_collided)
        reward_info = self._compute_racing_reward(pos, vel, action, gate_passed, collided)

        self.racing_state.prev_pos = pos.copy()
        self.racing_state.prev_action = action.copy()
        self.racing_state.episode_reward += reward_info.total
        self.racing_state.collided = collided
        self._last_reward_info = reward_info
        self._last_gate_passed = gate_passed

        obs = self._get_obs()
        terminated = collided
        truncated = self.racing_state.step_count >= self.max_episode_steps

        return obs, float(reward_info.total), terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        state = self.client.getMultirotorState()
        pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            -state.kinematics_estimated.position.z_val,
        ])
        vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            -state.kinematics_estimated.linear_velocity.z_val,
        ])

        q = state.kinematics_estimated.orientation
        rpy = airsim.to_eularian_angles(q)
        rpy = np.array(rpy)

        ang_vel = np.array([
            state.kinematics_estimated.angular_velocity.x_val,
            state.kinematics_estimated.angular_velocity.y_val,
            state.kinematics_estimated.angular_velocity.z_val,
        ])

        vel_body = world_to_body(vel, rpy)
        ang_vel_body = world_to_body(ang_vel, rpy)

        return self._build_obs(pos, vel_body, rpy, ang_vel_body)

    def get_image(self) -> np.ndarray | None:
        """Capture an RGB image from the drone's forward camera."""
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        if not responses or responses[0].width == 0:
            return None

        img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img.reshape(responses[0].height, responses[0].width, 3)
        return img

    def _get_info(self) -> dict:
        reward_info = getattr(self, "_last_reward_info", None)
        return {
            "gates_passed": self.racing_state.gates_passed,
            "next_gate_idx": self.racing_state.next_gate_idx,
            "collided": self.racing_state.collided,
            "gate_passed_this_step": getattr(self, "_last_gate_passed", False),
            "episode_reward": self.racing_state.episode_reward,
            "step_count": self.racing_state.step_count,
        }

    def close(self):
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception:
            pass
