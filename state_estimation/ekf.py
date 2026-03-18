"""Extended Kalman Filter for fusing PnP gate estimates with IMU telemetry.

State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
Measurements:
- IMU: position, velocity, orientation (high rate, noisy)
- PnP: gate-relative position (low rate, noisy when confident)
"""

from __future__ import annotations

import numpy as np


class DroneEKF:
    """9-state EKF for drone state estimation.

    Fuses IMU telemetry with PnP gate pose estimates for robust state
    estimation during perception-challenged conditions.
    """

    STATE_DIM = 9

    def __init__(
        self,
        dt: float = 1.0 / 48,
        process_noise: np.ndarray | None = None,
        imu_noise: np.ndarray | None = None,
        pnp_noise: np.ndarray | None = None,
    ):
        self.dt = dt

        self.x = np.zeros(self.STATE_DIM)
        self.P = np.eye(self.STATE_DIM) * 0.1

        if process_noise is None:
            self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        else:
            self.Q = process_noise

        if imu_noise is None:
            self.R_imu = np.diag([0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
        else:
            self.R_imu = imu_noise

        if pnp_noise is None:
            self.R_pnp = np.diag([0.1, 0.1, 0.1])
        else:
            self.R_pnp = pnp_noise

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(self.STATE_DIM)
        self.P = np.eye(self.STATE_DIM) * 0.1

    def predict(self) -> np.ndarray:
        """Predict step using constant-velocity model."""
        F = np.eye(self.STATE_DIM)
        F[0, 3] = self.dt
        F[1, 4] = self.dt
        F[2, 5] = self.dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy()

    def update_imu(self, measurement: np.ndarray) -> np.ndarray:
        """Update with full IMU measurement [x,y,z, vx,vy,vz, r,p,y]."""
        H = np.eye(self.STATE_DIM)
        y = measurement - H @ self.x
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.STATE_DIM) - K @ H) @ self.P
        return self.x.copy()

    def update_pnp(
        self,
        gate_position_world: np.ndarray,
        pnp_relative_position: np.ndarray,
        confidence: float,
    ) -> np.ndarray:
        """Update with PnP gate-relative position measurement.

        Parameters
        ----------
        gate_position_world : (3,) known gate position in world frame
        pnp_relative_position : (3,) estimated drone-to-gate vector from PnP
        confidence : detector confidence (0-1), scales measurement noise
        """
        inferred_pos = gate_position_world - pnp_relative_position

        H = np.zeros((3, self.STATE_DIM))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        noise_scale = 1.0 / max(confidence, 0.1)
        R = self.R_pnp * noise_scale

        y = inferred_pos - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.STATE_DIM) - K @ H) @ self.P

        return self.x.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def orientation(self) -> np.ndarray:
        return self.x[6:9]
