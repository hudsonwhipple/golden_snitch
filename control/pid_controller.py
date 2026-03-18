"""PID controller with anti-windup and rate limiting for drone velocity tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PIDGains:
    kp: float = 6.0
    ki: float = 0.5
    kd: float = 3.0


class PIDController:
    """Single-axis PID with integral anti-windup and output rate limiting."""

    def __init__(
        self,
        gains: PIDGains,
        output_limit: float = 1.0,
        integral_limit: float = 2.0,
        rate_limit: float = 10.0,
    ):
        self.gains = gains
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.rate_limit = rate_limit

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0
        self._initialized = False

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        p_term = self.gains.kp * error

        self._integral += error * dt
        self._integral = np.clip(
            self._integral, -self.integral_limit, self.integral_limit
        )
        i_term = self.gains.ki * self._integral

        if self._initialized:
            d_term = self.gains.kd * (error - self._prev_error) / dt
        else:
            d_term = 0.0
            self._initialized = True

        output = p_term + i_term + d_term
        output = np.clip(output, -self.output_limit, self.output_limit)

        if self.rate_limit > 0:
            delta = output - self._prev_output
            max_delta = self.rate_limit * dt
            delta = np.clip(delta, -max_delta, max_delta)
            output = self._prev_output + delta

        self._prev_error = error
        self._prev_output = output
        return float(output)


@dataclass
class CascadedPIDConfig:
    """Configuration for 4-axis velocity-to-command PID controller."""

    vx_gains: PIDGains = field(default_factory=lambda: PIDGains(kp=6.0, ki=0.5, kd=3.0))
    vy_gains: PIDGains = field(default_factory=lambda: PIDGains(kp=6.0, ki=0.5, kd=3.0))
    vz_gains: PIDGains = field(default_factory=lambda: PIDGains(kp=8.0, ki=1.0, kd=4.0))
    yaw_gains: PIDGains = field(default_factory=lambda: PIDGains(kp=4.0, ki=0.1, kd=1.0))
    output_limit: float = 1.0
    integral_limit: float = 2.0
    rate_limit: float = 10.0


class CascadedPIDController:
    """4-axis PID: velocity errors -> normalized actuator commands.

    Maps velocity targets (vx, vy, vz, yaw_rate) to (throttle, roll, pitch, yaw)
    commands in [-1, 1].
    """

    def __init__(self, config: CascadedPIDConfig | None = None):
        config = config or CascadedPIDConfig()
        common = dict(
            output_limit=config.output_limit,
            integral_limit=config.integral_limit,
            rate_limit=config.rate_limit,
        )
        self.pids = {
            "vx": PIDController(config.vx_gains, **common),
            "vy": PIDController(config.vy_gains, **common),
            "vz": PIDController(config.vz_gains, **common),
            "yaw": PIDController(config.yaw_gains, **common),
        }

    def reset(self) -> None:
        for pid in self.pids.values():
            pid.reset()

    def compute(
        self,
        vel_target: np.ndarray,
        vel_current: np.ndarray,
        yaw_rate_target: float,
        yaw_rate_current: float,
        dt: float,
    ) -> np.ndarray:
        """Compute actuator commands from velocity + yaw rate errors.

        Parameters
        ----------
        vel_target : (3,) desired velocity in body frame [vx, vy, vz]
        vel_current : (3,) current velocity in body frame [vx, vy, vz]
        yaw_rate_target : desired yaw rate (rad/s)
        yaw_rate_current : current yaw rate (rad/s)
        dt : timestep (seconds)

        Returns
        -------
        (4,) array of [throttle, roll, pitch, yaw] in [-1, 1]
        """
        errors = vel_target - vel_current

        pitch_cmd = self.pids["vx"].compute(errors[0], dt)
        roll_cmd = -self.pids["vy"].compute(errors[1], dt)
        throttle_cmd = self.pids["vz"].compute(errors[2], dt)
        yaw_cmd = self.pids["yaw"].compute(yaw_rate_target - yaw_rate_current, dt)

        return np.array([throttle_cmd, roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float32)
