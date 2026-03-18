"""Math utilities for quaternion/euler conversions and coordinate transforms."""

from __future__ import annotations

import numpy as np


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in radians."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def euler_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix."""
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ])
    return R


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def body_to_world(vec_body: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Transform a vector from body frame to world frame."""
    R = euler_to_rotation_matrix(rpy)
    return R @ vec_body


def world_to_body(vec_world: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Transform a vector from world frame to body frame."""
    R = euler_to_rotation_matrix(rpy)
    return R.T @ vec_world


def gate_forward_vector(gate_orientation: np.ndarray) -> np.ndarray:
    """Compute the forward (through) vector of a gate from its Euler orientation.

    Gates face along their local +X axis.
    """
    R = euler_to_rotation_matrix(gate_orientation)
    return R[:, 0]
