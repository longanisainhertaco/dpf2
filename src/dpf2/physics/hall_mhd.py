from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .mhd import ResistiveMHD


@dataclass
class HallMHD(ResistiveMHD):
    """Resistive MHD with Hall term and circuit back-EMF.

    This lightweight model extends :class:`~dpf2.physics.mhd.ResistiveMHD` by
    incorporating a Hall electric field contribution and an optional
    back electromotive force (EMF) supplied by an external circuit.  The
    formulation is intentionally simple and is aimed at unit tests rather than
    high fidelity simulations.
    """

    hall_coeff: float = 0.0
    current: float = 0.0
    back_emf: float = 0.0

    # ------------------------------------------------------------------
    # Primitive ↔ conservative conversions
    # ------------------------------------------------------------------
    def primitive_variables(self, U: np.ndarray) -> np.ndarray:
        """Return primitive variables from conservative state ``U``."""

        rho, m_x, m_y, m_z, E, B_x, B_y, B_z, _ = U
        v_x = m_x / rho
        v_y = m_y / rho
        v_z = m_z / rho
        v2 = v_x ** 2 + v_y ** 2 + v_z ** 2
        B2 = B_x ** 2 + B_y ** 2 + B_z ** 2
        p = (E - 0.5 * rho * v2 - 0.5 * B2) * (self.gamma - 1.0)
        return np.array([rho, v_x, v_y, v_z, p, B_x, B_y, B_z])

    # conservative_variables inherited from ResistiveMHD

    # ------------------------------------------------------------------
    # Fluxes with Hall term and back EMF
    # ------------------------------------------------------------------
    def flux_function(self, U: np.ndarray, direction: str, J: np.ndarray | None = None) -> np.ndarray:
        """Compute fluxes including Hall effect and optional back EMF.

        Parameters
        ----------
        U:
            Conservative state vector.
        direction:
            Spatial direction (``'x'``, ``'y'`` or ``'z'``).
        J:
            Current density ``curl(B)`` at the cell.  When omitted the Hall
            contribution vanishes.
        """

        F = super().flux_function(U, direction)

        if J is not None and self.hall_coeff != 0.0:
            rho = U[0]
            B = U[5:8]
            hall_e = self.hall_coeff * np.cross(J, B) / rho
            if direction == "x":
                F[6] -= hall_e[2]
                F[7] -= hall_e[1]
            elif direction == "y":
                F[5] -= hall_e[2]
                F[7] -= hall_e[0]
            elif direction == "z":
                F[5] -= hall_e[1]
                F[6] -= hall_e[0]

        if self.current != 0.0 and self.back_emf != 0.0:
            F[4] += self.current * self.back_emf

        return F


__all__ = ["HallMHD"]
