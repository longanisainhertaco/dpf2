from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mhd import ResistiveMHD
from ..mesh import Mesh2D, Mesh3D


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
        *,
        circuit: Any | None = None,
    ) -> np.ndarray:
        """Update circuit coupling information.

        The routine performs a very small subset of a full time advance.  It
        estimates the instantaneous plasma self–inductance from the magnetic
        energy and communicates this to an external circuit model.  No update
        of the plasma state itself is performed.

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
        circuit:
            Optional circuit solver implementing ``step(current, back_emf, dt,
            plasma_feedback)``.  When provided the circuit is advanced using the
            computed feedback terms and the updated current/back‑EMF are stored
            on the model instance.
        """

        prev_I = self.current
        self.current = current

        Lp = self.plasma_inductance(state)
        dLpdt = (Lp - self.inductance) / max(dt, 1.0e-30)

        dIdt = (current - prev_I) / max(dt, 1.0e-30)
        emf = Lp * dIdt + current * dLpdt

        # Store plasma feedback for the circuit solver.  The induced EMF is
        # recorded on ``back_emf`` while the external circuit is advanced with
        # the updated inductance information.  No separate ``back_emf`` term is
        # supplied to the circuit as the plasma induced voltage is already
        # accounted for via ``emf`` above.
        self.inductance = Lp
        self.back_emf = emf
        self.circuit_feedback = {"Lp": Lp, "emf": emf}

        if circuit is not None:
            # ``circuit.step`` returns the updated current and circuit voltage.
            # Only the current is retained; the plasma feedback remains stored
            # on ``self.back_emf`` and ``self.circuit_feedback``.
            new_current, _ = circuit.step(
                self.current, 0.0, dt, self.circuit_feedback
            )
            self.current = new_current

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

    # ------------------------------------------------------------------
    # Riemann solver and CTU update
    # ------------------------------------------------------------------
    def riemann_solver(
        self,
        UL: np.ndarray,
        UR: np.ndarray,
        direction: str,
        J_L: np.ndarray | None = None,
        J_R: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simple Rusanov solver for the Hall-MHD system."""

        F_L = self.flux_function(UL, direction, J=J_L)
        F_R = self.flux_function(UR, direction, J=J_R)
        smax = max(self.max_speed(UL, direction), self.max_speed(UR, direction))
        return 0.5 * (F_L + F_R) - 0.5 * smax * (UR - UL)

    def divergence_cleaning(
        self, U: np.ndarray, mesh: Mesh2D | Mesh3D, dt: float
    ) -> None:
        """Apply a simplified Dedner divergence-cleaning step in 1-D."""

        if self.c_h == 0.0 and self.c_p == 0.0:
            return

        dx = mesh.dx
        Bx = U[:, 5]
        psi = U[:, 8]
        divB = np.gradient(Bx, dx, edge_order=2)
        psi -= dt * (self.c_h ** 2 * divB + self.c_p ** 2 * psi)
        Bx -= dt * np.gradient(psi, dx, edge_order=2)
        U[:, 5] = Bx
        U[:, 8] = psi

    def ctu_update(
        self, U: np.ndarray, mesh: Mesh2D | Mesh3D, dt: float, *, periodic: bool = False
    ) -> np.ndarray:
        """Advance ``U`` by one CTU step in the ``x``-direction."""

        dx = mesh.dx
        n = U.shape[0]
        By = U[:, 6]
        Bz = U[:, 7]
        J = np.zeros((n, 3))
        J[:, 1] = -np.gradient(Bz, dx, edge_order=2)
        J[:, 2] = np.gradient(By, dx, edge_order=2)

        fluxes = np.zeros((n + 1, len(self.equations)))
        for i in range(n - 1):
            fluxes[i + 1] = self.riemann_solver(
                U[i], U[i + 1], "x", J[i], J[i + 1]
            )

        if periodic:
            fluxes[0] = self.riemann_solver(U[-1], U[0], "x", J[-1], J[0])
            fluxes[-1] = fluxes[0]
        else:
            fluxes[0] = self.flux_function(U[0], "x", J=J[0])
            fluxes[-1] = self.flux_function(U[-1], "x", J=J[-1])

        U_new = U.copy()
        for i in range(n):
            U_new[i] -= dt / dx * (fluxes[i + 1] - fluxes[i])
        # Apply diffusive/resistive source terms.  ``ResistiveMHD`` implements
        # a collection of local physics processes (Ohmic heating, viscosity,
        # thermal conduction, etc.) via ``source_terms``.  The Hall model only
        # requires the resistive piece so we explicitly zero the cleaning
        # contribution returned for ``psi`` (handled separately below).
        for i in range(n):
            src = self.source_terms(U_new[i])
            src[8] = 0.0  # divergence cleaning for ``psi`` handled explicitly
            U_new[i] += dt * src

        self.divergence_cleaning(U_new, mesh, dt)
        return U_new


__all__ = ["HallMHD"]
