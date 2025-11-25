"""Coaxial-geometry shakedown driver for the Hall-MHD solver."""
from __future__ import annotations

import numpy as np

from dpf2.hall_mhd_solver import HallMHDSolver, MHDState
from dpf2.physics.inductance import CoaxialGeometry


def coaxial_shakedown(
    *,
    nx: int = 8,
    ny: int = 8,
    nz: int = 16,
    dt: float = 1e-9,
) -> tuple[MHDState, float]:
    """Advance a few IMEX steps on a coarse coaxial mesh without AMR."""

    shape = (nx, ny, nz)
    rho = np.ones(shape) * 1e-3
    mom = np.zeros(shape + (3,))
    energy = np.ones(shape) * 1e3
    B = np.zeros(shape + (3,))

    state = MHDState(rho=rho, mom=mom, energy=energy, B=B)
    geom = CoaxialGeometry(anode_radius=7.5e-3, cathode_radius=4.0e-2, anode_length=0.18)
    solver = HallMHDSolver(geometry=geom, kappa_par=1.0)
    solver.enable_imex(use_petsc=False)
    solver.attach_optically_thin_radiation()

    for _ in range(2):
        state = solver.step(state, dt=dt, current=1e5, voltage=2e4)

    return state, solver.coupling_interface().Lp


__all__ = ["coaxial_shakedown"]
