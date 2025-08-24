from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .mhd import ResistiveMHD


@dataclass
class HallMHD(ResistiveMHD):
    """Resistive MHD with Hall term and circuit coupling.

    The model augments :class:`~dpf2.physics.mhd.ResistiveMHD` with two
    additional pieces of physics used throughout the unit tests:

    * **Hall electric field** – a dispersive correction proportional to
      ``J×B`` that is controlled by ``hall_coeff``.
    * **Dynamic inductance / back–EMF coupling** – the solver keeps track of
      a plasma current and an externally supplied voltage.  From the magnetic
      energy the plasma self‑inductance is estimated and fed back to external
      circuit models.

    Only the terms required for regression tests are implemented; the class is
    not intended to be a production ready Hall‑MHD solver.
    """

    hall_coeff: float = 0.0
    current: float = 0.0
    back_emf: float = 0.0

    # plasma inductance state (Henries)
    inductance: float = 0.0
    circuit_feedback: dict[str, float] | None = field(default=None, init=False)

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
    # Coupling helpers
    # ------------------------------------------------------------------
    def plasma_inductance(self, U: np.ndarray) -> float:
        """Estimate the plasma inductance from magnetic energy.

        The expression ``E_mag = 0.5 * Lp * I^2`` is inverted to give
        ``Lp``.  If the current is zero the inductance is taken as zero.
        """

        B = U[5:8]
        mag_energy = 0.5 * np.dot(B, B)
        if self.current == 0.0:
            return 0.0
        return 2.0 * mag_energy / (self.current**2)

    def step(
        self,
        state: np.ndarray,
        dt: float,
        current: float = 0.0,
        voltage: float = 0.0,
    ) -> np.ndarray:
        """Lightweight advance that updates circuit feedback only.

        Parameters
        ----------
        state:
            Plasma state (unused but kept for API compatibility).
        dt:
            Time step in seconds.
        current:
            Circuit current supplied by the external circuit.
        voltage:
            Deprecated and ignored.  Previously represented an externally
            applied back‑EMF.

        The routine estimates the plasma inductance and associated back‑EMF
        ``emf = Lp * dI/dt + I * dLp/dt`` which are exposed through the
        ``circuit_feedback`` attribute for use by external circuit solvers.
        """

        prev_I = self.current
        self.current = current

        Lp = self.plasma_inductance(state)
        dLpdt = (Lp - self.inductance) / max(dt, 1.0e-30)
        dIdt = (current - prev_I) / max(dt, 1.0e-30)
        emf = Lp * dIdt + current * dLpdt

        self.inductance = Lp
        self.back_emf = emf
        self.circuit_feedback = {"Lp": Lp, "emf": emf}

        return state

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

        return F


__all__ = ["HallMHD"]
