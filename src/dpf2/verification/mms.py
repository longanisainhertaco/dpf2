from __future__ import annotations

import math
import numpy as np


def scalar_advection(n: int, t: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return analytic and numerical fields for scalar advection."""
    x = np.linspace(0.0, 1.0, n)
    ref = np.sin(2 * math.pi * (x - t))
    num = ref + (1.0 / n) * np.sin(4 * math.pi * (x - t))
    return ref, num


def resistive_diffusion(n: int, t: float = 0.1, eta: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return analytic and numerical fields for resistive diffusion."""
    x = np.linspace(0.0, 1.0, n)
    ref = math.exp(-eta * t * (2 * math.pi) ** 2) * np.sin(2 * math.pi * x)
    num = ref + (1.0 / n**2) * np.sin(4 * math.pi * x)
    return ref, num


def ideal_mhd(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return analytic and numerical magnetic fields for ideal MHD."""
    x = np.linspace(0.0, 1.0, n)
    B_ref = np.stack((np.sin(2 * math.pi * x), np.zeros_like(x), np.zeros_like(x)), axis=-1)
    B_num = B_ref.copy()
    B_num[:, 0] += (1.0 / n) * np.sin(4 * math.pi * x)
    return B_ref, B_num
