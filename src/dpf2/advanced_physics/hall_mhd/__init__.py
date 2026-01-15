"""Hall-MHD solver with constrained transport for 3D simulations.

This module provides a complete implementation of the Hall-MHD equations
using constrained transport to maintain the divergence-free constraint
on the magnetic field.
"""

from __future__ import annotations

from .hall_solver import HallMHDSolver3D, Grid3D, MHDState3D
from .constrained_transport import CTUpdate, compute_curl, compute_divergence, project_div_free
from .whistler_dispersion import whistler_frequency, dispersion_relation, phase_velocity

__all__ = [
    "HallMHDSolver3D",
    "Grid3D",
    "MHDState3D",
    "CTUpdate",
    "compute_curl",
    "compute_divergence",
    "project_div_free",
    "whistler_frequency",
    "dispersion_relation",
    "phase_velocity",
]
