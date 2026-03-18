"""Drone racing environment using lightweight quadrotor dynamics.

This is the primary training environment -- fast, pure numpy, no pybullet dependency.
Action space: [thrust, roll, pitch, yaw_rate] in [-1, 1]
Observation space: flat 24-dim state vector
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environments.base_env import OBS_DIM, ACT_DIM, RacingMixin
from environments.quadrotor_dynamics import QuadrotorSim, QuadrotorParams
from environments.gate_detection import compute_gate_relative_position
from environments.tracks.track_loader import load_track
from utils.math_utils import world_to_body


class DroneRacingEnv(gym.Env, RacingMixin):
    """Gymnasium environment for drone racing with built-in quadrotor dynamics.

    This environment is designed for fast RL training on any platform (no pybullet needed).
    For higher-fidelity validation, use PyBulletRacingEnv when available.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        track_name: str = "simple_oval",
        max_episode_steps: int = 1000,
        reward_cfg: dict | None = None,
        physics_hz: int = 240,
        policy_hz: int = 48,
        initial_perturbation: float = 0.0,
        obs_noise_std: float = 0.0,
        render_mode: str | None = None,
    ):
        super().__init__()

        self._track_name = track_name
        self._physics_hz = physics_hz
        self._policy_hz = policy_hz
        self._physics_steps_per_policy = physics_hz // policy_hz
        self._initial_perturbation = initial_perturbation
        self._obs_noise_std = obs_noise_std
        self.render_mode = render_mode

        track = load_track(track_name)
        self._init_racing(track, reward_cfg)
        self.max_episode_steps = max_episode_steps

        self._sim = QuadrotorSim(
            params=QuadrotorParams(),
            dt=1.0 / physics_hz,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_racing()

        start_pos = self.track.start_position.copy()
        start_rpy = self.track.start_orientation.copy()

        if self._initial_perturbation > 0:
            start_pos += self._rng.uniform(
                -self._initial_perturbation, self._initial_perturbation, size=3
            )
            start_pos[2] = max(start_pos[2], 0.3)

        self._sim.reset(start_pos, start_rpy)
        self.racing_state.prev_pos = start_pos.copy()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()[:ACT_DIM]
        self.racing_state.step_count += 1

        for _ in range(self._physics_steps_per_policy):
            state = self._sim.step(action)

        pos = state.position
        vel_world = state.velocity
        rpy = state.orientation

        gate_passed = False
        if self.racing_state.prev_pos is not None:
            gate_passed = self._update_gate_progress(self.racing_state.prev_pos, pos)

        collided = self._check_collision(pos)

        prev_action = self.racing_state.prev_action
        if prev_action is None:
            prev_action = np.zeros(ACT_DIM, dtype=np.float32)

        reward_info = self._compute_racing_reward(pos, vel_world, action, gate_passed, collided)

        self.racing_state.prev_pos = pos.copy()
        self.racing_state.prev_action = action.copy()
        self.racing_state.episode_reward += reward_info.total
        self.racing_state.collided = collided
        self._last_reward_info = reward_info
        self._last_gate_passed = gate_passed

        obs = self._get_obs()
        terminated = collided
        truncated = self._check_truncated()
        info = self._get_info()

        return obs, float(reward_info.total), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        s = self._sim.state
        vel_body = world_to_body(s.velocity, s.orientation)
        ang_vel_body = world_to_body(s.angular_velocity, s.orientation)

        obs = self._build_obs(s.position, vel_body, s.orientation, ang_vel_body)

        if self._obs_noise_std > 0:
            noise = self._rng.normal(0, self._obs_noise_std, size=obs.shape).astype(np.float32)
            obs = obs + noise

        return obs

    def _check_collision(self, pos: np.ndarray) -> bool:
        if pos[2] < 0.05:
            return True
        if np.any(np.abs(pos) > 20.0):
            return True
        return False

    def _check_truncated(self) -> bool:
        if self.racing_state.step_count >= self.max_episode_steps:
            return True
        gates = self._get_active_gates()
        if self.racing_state.gates_passed >= len(gates):
            return True
        return False

    def _get_info(self) -> dict:
        reward_info = getattr(self, "_last_reward_info", None)
        return {
            "gates_passed": self.racing_state.gates_passed,
            "next_gate_idx": self.racing_state.next_gate_idx,
            "collided": self.racing_state.collided,
            "gate_passed_this_step": getattr(self, "_last_gate_passed", False),
            "episode_reward": self.racing_state.episode_reward,
            "step_count": self.racing_state.step_count,
            "position": self._sim.state.position.copy(),
            "velocity": self._sim.state.velocity.copy(),
            "reward_breakdown": {
                "gate": reward_info.gate if reward_info else 0.0,
                "progress": reward_info.progress if reward_info else 0.0,
                "velocity": reward_info.velocity if reward_info else 0.0,
                "smoothness": reward_info.smoothness if reward_info else 0.0,
                "collision": reward_info.collision if reward_info else 0.0,
                "time": reward_info.time if reward_info else 0.0,
            },
        }
