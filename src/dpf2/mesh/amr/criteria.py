from __future__ import annotations

import numpy as np


def plasma_gradient_refinement(field: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask of cells where the gradient magnitude exceeds ``threshold``."""
    if field is None:
        raise ValueError("field must be provided")
    arr = getattr(field, "data", field)
    if isinstance(arr[0], list):
        nx = len(arr)
        ny = len(arr[0])
        mask = [[False] * ny for _ in range(nx)]
        for i in range(nx):
            for j in range(ny):
                left = arr[i - 1 if i > 0 else 0][j]
                right = arr[i + 1 if i < nx - 1 else nx - 1][j]
                down = arr[i][j - 1 if j > 0 else 0]
                up = arr[i][j + 1 if j < ny - 1 else ny - 1]
                gx = float(right) - float(left)
                gy = float(up) - float(down)
                mag = (gx ** 2 + gy ** 2) ** 0.5 * 0.5
                mask[i][j] = mag > threshold
    else:
        nx = len(arr)
        mask = [False] * nx
        for i in range(nx):
            left = arr[i - 1 if i > 0 else 0]
            right = arr[i + 1 if i < nx - 1 else nx - 1]
            gx = float(right) - float(left)
            mask[i] = abs(gx) * 0.5 > threshold
    return np.array(mask)


def debye_length_refinement(lambda_D: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where the Debye length ``lambda_D`` falls below ``threshold``."""
    arr = getattr(lambda_D, "data", lambda_D)
    if isinstance(arr[0], list):
        mask = [[v < threshold for v in row] for row in arr]
    else:
        mask = [v < threshold for v in arr]
    return np.array(mask)


def ion_inertial_length_refinement(d_i: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where the ion inertial length ``d_i`` falls below ``threshold``."""
    arr = getattr(d_i, "data", d_i)
    if isinstance(arr[0], list):
        mask = [[v < threshold for v in row] for row in arr]
    else:
        mask = [v < threshold for v in arr]
    return np.array(mask)


# Alias for user terminology
ion_skin_depth_refinement = ion_inertial_length_refinement


def pressure_gradient_refinement(pressure: np.ndarray, threshold: float) -> np.ndarray:
    """Convenience wrapper applying :func:`plasma_gradient_refinement` to pressure."""
    return plasma_gradient_refinement(pressure, threshold)


def current_density_refinement(current: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where the current density magnitude exceeds ``threshold``."""
    arr = getattr(current, "data", current)
    mask = []
    for plane in arr:
        plane_mask = []
        for vec in plane:
            mag2 = sum(float(c) ** 2 for c in vec)
            plane_mask.append(mag2 ** 0.5 > threshold)
        mask.append(plane_mask)
    return np.array(mask)


def current_gradient_refinement(current: np.ndarray, threshold: float) -> np.ndarray:
    """Refine based on gradients of the current magnitude."""
    arr = getattr(current, "data", current)
    mag = []
    for plane in arr:
        plane_mag = []
        for vec in plane:
            plane_mag.append(sum(float(c) ** 2 for c in vec) ** 0.5)
        mag.append(plane_mag)
    return plasma_gradient_refinement(mag, threshold)


def wavefront_refinement(field: np.ndarray, prev_field: np.ndarray, threshold: float) -> np.ndarray:
    """Tag cells where the change between two fields exceeds ``threshold``."""
    if field is None or prev_field is None:
        raise ValueError("Both current and previous fields are required")
    arr = getattr(field, "data", field)
    prev = getattr(prev_field, "data", prev_field)
    if isinstance(arr[0], list):
        nx = len(arr)
        ny = len(arr[0])
        mask = [[False] * ny for _ in range(nx)]
        for i in range(nx):
            for j in range(ny):
                if abs(float(arr[i][j]) - float(prev[i][j])) > threshold:
                    mask[i][j] = True
    else:
        nx = len(arr)
        mask = [False] * nx
        for i in range(nx):
            if abs(float(arr[i]) - float(prev[i])) > threshold:
                mask[i] = True
    return np.array(mask)
