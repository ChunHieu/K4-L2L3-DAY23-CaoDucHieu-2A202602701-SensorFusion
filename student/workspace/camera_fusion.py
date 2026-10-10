"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

Matrix = np.matrix | np.ndarray

from fusion_lab.workspace_support import get_tracking_params


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    p = np.asarray(x, dtype=float).reshape(-1)[:3]
    if not np.isfinite(p).all():
        return False
    T = np.asarray(sensor.veh_to_sens)
    p_s = T[:3, :3] @ p + T[:3, 3]
    if not np.isfinite(p_s).all():
        return False
    if getattr(sensor, "name", None) == "camera" or hasattr(sensor, "f_i"):
        if p_s[0] <= 1e-6:
            return False
    angle = float(np.arctan2(p_s[1], p_s[0]))
    fov_min = min(sensor.fov)
    fov_max = max(sensor.fov)
    return bool(fov_min <= angle <= fov_max)


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    p = np.asarray(x, dtype=float).reshape(-1)[:3]
    if not np.isfinite(p).all():
        raise ValueError(f"State coordinates are nonfinite: {p.tolist()}")
    T = np.asarray(sensor.veh_to_sens)
    p_s = T[:3, :3] @ p + T[:3, 3]
    if not np.isfinite(p_s).all():
        raise ValueError(f"Sensor coordinates are nonfinite: {p_s.tolist()}")
    if p_s[0] <= 1e-6:
        raise ValueError(
            f"Camera depth {p_s[0]} is at most 1e-6; sensor position={p_s.tolist()}"
        )
    u = sensor.c_i - sensor.f_i * p_s[1] / p_s[0]
    v = sensor.c_j - sensor.f_j * p_s[2] / p_s[0]
    return np.matrix([[float(u)], [float(v)]], dtype=float)


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    z_mat = np.matrix([[float(z[0])], [float(z[1])]], dtype=float)
    R = np.matrix(
        np.diag([float(params.sigma_cam_i**2), float(params.sigma_cam_j**2)]),
        dtype=float,
    )
    return {"z": z_mat, "R": R, "sensor": sensor}

