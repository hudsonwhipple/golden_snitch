"""PyBullet racing environment built on gym-pybullet-drones BaseAviary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from environments.base_env import OBS_DIM, ACT_DIM, RacingMixin
from environments.tracks.track_loader import load_track
from utils.math_utils import world_to_body


class PyBulletRacingEnv(BaseAviary, RacingMixin):
    """High-speed drone racing environment using PyBullet physics.

    The policy outputs velocity targets in [-1, 1] which are scaled to
    real velocities and tracked by the DSLPIDControl controller.

    Observations are a flat 24-dim vector containing telemetry + gate-relative state.
    """

    def __init__(
        self,
        track_name: str = "simple_oval",
        max_velocity: float = 10.0,
        max_yaw_rate: float = np.pi,
        max_episode_steps: int = 1000,
        reward_cfg: dict | None = None,
        pyb_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        initial_perturbation: float = 0.0,
    ):
        self._track_name = track_name
        self.max_velocity = max_velocity
        self.max_yaw_rate = max_yaw_rate
        self._initial_perturbation = initial_perturbation

        track = load_track(track_name)
        self._init_racing(track, reward_cfg)
        self.max_episode_steps = max_episode_steps

        initial_xyz = track.start_position.copy().reshape(1, 3)
        initial_rpy = track.start_orientation.copy().reshape(1, 3)

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=initial_xyz,
            initial_rpys=initial_rpy,
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=False,
            obstacles=True,
            user_debug_gui=False,
            vision_attributes=False,
        )

        self._pid = DSLPIDControl(drone_model=DroneModel.CF2X)

    def _actionSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def _addObstacles(self):
        """Draw gate frames in the simulation."""
        self._gate_visual_ids = []
        for gate in self.track.gates:
            self._draw_gate(gate)

    def _draw_gate(self, gate):
        """Draw a gate frame using PyBullet debug lines."""
        R = np.eye(3)
        from utils.math_utils import euler_to_rotation_matrix

        R = euler_to_rotation_matrix(gate.orientation)
        hw, hh = gate.size[0] / 2, gate.size[1] / 2

        corners_local = np.array([
            [0, -hw, -hh],
            [0,  hw, -hh],
            [0,  hw,  hh],
            [0, -hw,  hh],
        ])
        corners_world = np.array([gate.position + R @ c for c in corners_local])

        color = [1.0, 0.2, 0.2]
        width = 3.0
        for i in range(4):
            j = (i + 1) % 4
            p.addUserDebugLine(
                corners_world[i].tolist(),
                corners_world[j].tolist(),
                lineColorRGB=color,
                lineWidth=width,
                physicsClientId=self.CLIENT,
            )

        forward = gate.position + R[:, 0] * 0.3
        p.addUserDebugLine(
            gate.position.tolist(),
            forward.tolist(),
            lineColorRGB=[0.2, 1.0, 0.2],
            lineWidth=2.0,
            physicsClientId=self.CLIENT,
        )

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel_world = state[10:13]
        ang_vel_world = state[13:16]

        vel_body = world_to_body(vel_world, rpy)
        ang_vel_body = world_to_body(ang_vel_world, rpy)

        return self._build_obs(pos, vel_body, rpy, ang_vel_body)

    def _preprocessAction(self, action):
        """Convert normalized velocity targets to motor RPMs via PID.

        Action: [vx, vy, vz, yaw_rate] in [-1, 1]
        Scaled to: [vx, vy, vz] * max_velocity, yaw_rate * max_yaw_rate
        """
        action = np.asarray(action, dtype=np.float64).flatten()[:4]

        vel_target = action[0:3] * self.max_velocity
        yaw_rate_target = action[3] * self.max_yaw_rate

        state = self._getDroneStateVector(0)
        cur_pos = state[0:3]
        cur_quat = state[3:7]
        cur_vel = state[10:13]
        cur_ang_vel = state[13:16]
        cur_yaw = state[9]

        target_pos = cur_pos + vel_target * self.CTRL_TIMESTEP

        rpm, _, _ = self._pid.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, cur_yaw + yaw_rate_target * self.CTRL_TIMESTEP]),
            target_vel=vel_target,
        )

        return rpm.reshape(1, 4)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]

        gate_passed = False
        if self.racing_state.prev_pos is not None:
            gate_passed = self._update_gate_progress(self.racing_state.prev_pos, pos)

        collided = self._check_collision(pos)

        action = self.racing_state.prev_action
        if action is None:
            action = np.zeros(ACT_DIM, dtype=np.float32)

        reward_info = self._compute_racing_reward(pos, vel, action, gate_passed, collided)

        self.racing_state.prev_pos = pos.copy()
        self.racing_state.episode_reward += reward_info.total
        self.racing_state.collided = collided
        self._last_reward_info = reward_info
        self._last_gate_passed = gate_passed

        return float(reward_info.total)

    def _check_collision(self, pos: np.ndarray) -> bool:
        """Check for ground collision or out-of-bounds."""
        if pos[2] < 0.05:
            return True
        if np.any(np.abs(pos) > 20.0):
            return True
        contact_points = p.getContactPoints(
            bodyA=self.DRONE_IDS[0],
            bodyB=self.PLANE_ID,
            physicsClientId=self.CLIENT,
        )
        if len(contact_points) > 0:
            return True
        return False

    def _computeTerminated(self):
        return bool(self.racing_state.collided)

    def _computeTruncated(self):
        steps = self.step_counter // self.PYB_STEPS_PER_CTRL
        if steps >= self.max_episode_steps:
            return True
        gates = self._get_active_gates()
        if self.racing_state.gates_passed >= len(gates):
            return True
        return False

    def _computeInfo(self):
        reward_info = getattr(self, "_last_reward_info", None)
        return {
            "gates_passed": self.racing_state.gates_passed,
            "next_gate_idx": self.racing_state.next_gate_idx,
            "collided": self.racing_state.collided,
            "gate_passed_this_step": getattr(self, "_last_gate_passed", False),
            "episode_reward": self.racing_state.episode_reward,
            "reward_breakdown": {
                "gate": reward_info.gate if reward_info else 0.0,
                "progress": reward_info.progress if reward_info else 0.0,
                "velocity": reward_info.velocity if reward_info else 0.0,
                "smoothness": reward_info.smoothness if reward_info else 0.0,
                "collision": reward_info.collision if reward_info else 0.0,
                "time": reward_info.time if reward_info else 0.0,
            },
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()[:4]
        self.racing_state.step_count += 1

        obs, reward, terminated, truncated, info = super().step(action)

        self.racing_state.prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self._pid.reset()
        self._reset_racing()

        if self._initial_perturbation > 0:
            rng = np.random.default_rng(seed)
            perturb = rng.uniform(
                -self._initial_perturbation, self._initial_perturbation, size=3
            )
            self.INIT_XYZS = (self.track.start_position + perturb).reshape(1, 3)
            self.INIT_XYZS[0, 2] = max(self.INIT_XYZS[0, 2], 0.3)
        else:
            self.INIT_XYZS = self.track.start_position.copy().reshape(1, 3)

        self.INIT_RPYS = self.track.start_orientation.copy().reshape(1, 3)

        obs, info = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        self.racing_state.prev_pos = state[0:3].copy()
        return obs, info
