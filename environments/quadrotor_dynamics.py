"""Lightweight quadrotor dynamics simulation using pure numpy.

Based on the standard rigid-body quadrotor model (see Mellinger & Kumar, 2011).
Designed for fast RL training without pybullet dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.math_utils import euler_to_rotation_matrix


@dataclass
class QuadrotorParams:
    """Physical parameters for a Crazyflie 2.x-class quadrotor."""

    mass: float = 0.027          # kg
    arm_length: float = 0.0397   # m
    inertia: np.ndarray = None   # 3x3 inertia matrix
    gravity: float = 9.81        # m/s^2
    max_thrust: float = 0.6      # N (total, all 4 motors)
    max_torque_xy: float = 0.005 # Nm
    max_torque_z: float = 0.002  # Nm
    drag_coeff: float = 0.01     # linear drag coefficient

    def __post_init__(self):
        if self.inertia is None:
            ixx = 1.4e-5
            iyy = 1.4e-5
            izz = 2.17e-5
            self.inertia = np.diag([ixx, iyy, izz])


@dataclass
class QuadrotorState:
    """Full state of the quadrotor."""

    position: np.ndarray      # (3,) world frame [m]
    velocity: np.ndarray      # (3,) world frame [m/s]
    orientation: np.ndarray   # (3,) roll, pitch, yaw [rad]
    angular_velocity: np.ndarray  # (3,) body frame [rad/s]

    def copy(self) -> QuadrotorState:
        return QuadrotorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )


class QuadrotorSim:
    """Simple quadrotor physics simulation.

    Control input: [thrust_normalized, roll_cmd, pitch_cmd, yaw_rate_cmd] in [-1, 1].
    - thrust_normalized: 0 = hover thrust, -1 = zero, +1 = max thrust
    - roll_cmd, pitch_cmd: desired angles (scaled by max_angle)
    - yaw_rate_cmd: desired yaw rate (scaled by max_yaw_rate)
    """

    def __init__(
        self,
        params: QuadrotorParams | None = None,
        dt: float = 1 / 48,
        max_angle: float = np.pi / 6,
        max_yaw_rate: float = np.pi,
    ):
        self.params = params or QuadrotorParams()
        self.dt = dt
        self.max_angle = max_angle
        self.max_yaw_rate = max_yaw_rate
        self.hover_thrust = self.params.mass * self.params.gravity

        self.state = QuadrotorState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3),
        )

    def reset(self, position: np.ndarray, orientation: np.ndarray) -> QuadrotorState:
        self.state = QuadrotorState(
            position=position.copy(),
            velocity=np.zeros(3),
            orientation=orientation.copy(),
            angular_velocity=np.zeros(3),
        )
        return self.state.copy()

    def step(self, action: np.ndarray) -> QuadrotorState:
        """Advance simulation by one timestep.

        Parameters
        ----------
        action : (4,) array in [-1, 1]
            [thrust_norm, roll_cmd, pitch_cmd, yaw_rate_cmd]

        Returns
        -------
        Updated state
        """
        action = np.clip(action, -1, 1)
        thrust_norm, roll_cmd, pitch_cmd, yaw_rate_cmd = action

        thrust = self.hover_thrust * (1.0 + thrust_norm)
        thrust = max(0.0, min(thrust, self.params.max_thrust))

        target_roll = roll_cmd * self.max_angle
        target_pitch = pitch_cmd * self.max_angle
        target_yaw_rate = yaw_rate_cmd * self.max_yaw_rate

        p = self.params
        s = self.state
        R = euler_to_rotation_matrix(s.orientation)

        thrust_world = R @ np.array([0, 0, thrust / p.mass])
        gravity = np.array([0, 0, -p.gravity])
        drag = -p.drag_coeff * s.velocity
        accel = thrust_world + gravity + drag

        roll_err = target_roll - s.orientation[0]
        pitch_err = target_pitch - s.orientation[1]
        yaw_rate_err = target_yaw_rate - s.angular_velocity[2]

        kp_rp = 20.0
        kd_rp = 5.0
        kp_yaw = 10.0

        torque_roll = kp_rp * roll_err - kd_rp * s.angular_velocity[0]
        torque_pitch = kp_rp * pitch_err - kd_rp * s.angular_velocity[1]
        torque_yaw = kp_yaw * yaw_rate_err

        torque_roll = np.clip(torque_roll, -p.max_torque_xy, p.max_torque_xy)
        torque_pitch = np.clip(torque_pitch, -p.max_torque_xy, p.max_torque_xy)
        torque_yaw = np.clip(torque_yaw, -p.max_torque_z, p.max_torque_z)

        torques = np.array([torque_roll, torque_pitch, torque_yaw])
        I = p.inertia
        ang_accel = np.linalg.solve(I, torques - np.cross(s.angular_velocity, I @ s.angular_velocity))

        s.velocity = s.velocity + accel * self.dt
        s.position = s.position + s.velocity * self.dt
        s.angular_velocity = s.angular_velocity + ang_accel * self.dt
        s.orientation = s.orientation + s.angular_velocity * self.dt

        s.orientation[0] = np.clip(s.orientation[0], -np.pi / 2, np.pi / 2)
        s.orientation[1] = np.clip(s.orientation[1], -np.pi / 2, np.pi / 2)
        s.orientation[2] = (s.orientation[2] + np.pi) % (2 * np.pi) - np.pi

        return s.copy()
