"""Gate pose estimation from detected keypoints using PnP."""

from __future__ import annotations

import numpy as np


# Default gate 3D corners in gate-local frame (Y-Z plane, gate faces +X)
DEFAULT_GATE_SIZE = (1.0, 1.0)  # width, height in meters


def get_gate_3d_corners(width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """Get 3D gate corner coordinates in the gate's local frame.

    Returns (4, 3) array: top-left, top-right, bottom-right, bottom-left.
    Gate lies in the Y-Z plane facing +X direction.
    """
    hw, hh = width / 2, height / 2
    return np.array([
        [0.0, -hw,  hh],  # top-left
        [0.0,  hw,  hh],  # top-right
        [0.0,  hw, -hh],  # bottom-right
        [0.0, -hw, -hh],  # bottom-left
    ], dtype=np.float64)


def estimate_gate_pose(
    keypoints: np.ndarray,
    image_size: tuple[int, int] = (128, 96),
    gate_size: tuple[float, float] = DEFAULT_GATE_SIZE,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Estimate gate 6DoF pose from detected 2D keypoints using solvePnP.

    Parameters
    ----------
    keypoints : (4, 2) normalized keypoint coordinates in [0, 1]
    image_size : (width, height) of the image
    gate_size : (width, height) of the physical gate in meters
    camera_matrix : 3x3 camera intrinsic matrix
    dist_coeffs : distortion coefficients

    Returns
    -------
    position : (3,) estimated gate position relative to camera
    orientation : (3,) estimated gate orientation (rodrigues -> euler)
    success : whether PnP converged
    """
    try:
        import cv2
    except ImportError:
        return np.zeros(3), np.zeros(3), False

    if camera_matrix is None:
        w, h = image_size
        fx = fy = w * 0.8  # rough focal length estimate
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)

    if dist_coeffs is None:
        dist_coeffs = np.zeros(4, dtype=np.float64)

    object_points = get_gate_3d_corners(*gate_size)

    w, h = image_size
    image_points = keypoints.copy() * np.array([w, h])
    image_points = image_points.astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success:
        return np.zeros(3), np.zeros(3), False

    position = tvec.flatten()
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    orientation = np.array([roll, pitch, yaw])
    return position, orientation, True
