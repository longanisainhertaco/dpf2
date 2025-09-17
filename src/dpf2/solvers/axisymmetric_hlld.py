"""Axisymmetric HLLD MHD solver with constrained transport.

This module implements a very small 2-D axisymmetric magneto-hydrodynamics
solver.  The solver is intentionally lightweight – it only includes the
pieces required for the unit tests in this kata.  The implementation follows
closely the structure of typical finite–volume MHD solvers:

* A Riemann solver (``hlld_flux``) that computes numerical fluxes using a
  simplified HLLD approximate solution of the Riemann problem.
* A constrained transport update which evolves the magnetic field using the
  curl of an edge centred electric field, guaranteeing a divergence free
  magnetic field to machine precision.

The solver operates on state dictionaries containing 2-D ``numpy.ndarray``
objects.  The expected fields are ``rho`` (density), the three momentum
components ``mom_r``, ``mom_phi`` and ``mom_z``, magnetic field components
``B_r``, ``B_phi`` and ``B_z`` and the total ``energy``.  The public
:class:`AxisymmetricHLLD` class conforms to
:class:`~dpf2.core.bases.PlasmaSolverBase` by providing a :meth:`step` method
advancing the state by ``dt`` seconds.

The goal of the tests is not to provide a production ready plasma solver, but
rather to exercise the interface and demonstrate the constrained transport
update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from ..gpu_utils import xp as np

from dpf2.core.bases import PlasmaSolverBase

# ``numpy`` may be replaced with a lightweight stub during testing which does
# not expose the ``ndarray`` attribute.  Using ``Any`` keeps the type hints
# flexible while still documenting the expected dictionary layout.
State = Dict[str, Any]


@dataclass
class AxisymmetricHLLD(PlasmaSolverBase):
    """Minimal 2-D axisymmetric MHD solver.

    Parameters
    ----------
    gamma:
        Ratio of specific heats.  A value of ``5/3`` (ideal monoatomic gas)
        is used by default.
    """

    gamma: float = 5.0 / 3.0

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def step(
        self,
        state: State,
        dt: float,
        dr: float = 1.0,
        dz: float = 1.0,
        sources: State | None = None,
        species_sources: State | None = None,
        wall_ablation: dict[str, float] | None = None,
        sources_only: bool = False,
    ) -> State:
        """Advance the ``state`` by ``dt`` seconds.

        The method performs a single explicit Euler update using the
        simplified HLLD flux and applies a constrained transport update to
        the magnetic field.  Additional source terms may be supplied via the
        ``sources`` mapping which are applied after the flux update.  For
        chemistry style problems a set of per-species source terms can be
        passed via ``species_sources``.  A very small wall-ablation model is
        supported through ``wall_ablation`` which injects mass for each named
        species into the boundary cell ``(0,0)``.
        """

        if not sources_only:
            # Stack conserved variables into a single array for convenience
            U = np.stack(
                [
                    state["rho"],
                    state["mom_r"],
                    state["mom_phi"],
                    state["mom_z"],
                    state["B_r"],
                    state["B_phi"],
                    state["B_z"],
                    state["energy"],
                ]
            )

            # Compute fluxes in r and z direction using a very small HLLD solver
            flux_r = self._interface_flux(U[:, :-1, :], U[:, 1:, :], axis=0)
            flux_z = self._interface_flux(U[:, :, :-1], U[:, :, 1:], axis=1)

            # Update conserved variables using divergence of fluxes
            U[:, 1:-1, 1:-1] -= dt / dr * (flux_r[:, 1:, 1:-1] - flux_r[:, :-1, 1:-1])
            U[:, 1:-1, 1:-1] -= dt / dz * (flux_z[:, 1:-1, 1:] - flux_z[:, 1:-1, :-1])

            # Constrained transport update of magnetic field ---------------------------------
            self._constrained_transport(U, dt, dr, dz)

            (
                state["rho"],
                state["mom_r"],
                state["mom_phi"],
                state["mom_z"],
                state["B_r"],
                state["B_phi"],
                state["B_z"],
                state["energy"],
            ) = U

        if sources:
            for key, src in sources.items():
                if key in state:
                    state[key] = state[key] + dt * src
                else:
                    state[key] = dt * src
        if species_sources:
            for sp, src in species_sources.items():
                if sp in state:
                    state[sp] = state[sp] + dt * src
                else:
                    state[sp] = dt * src
        if wall_ablation:
            for sp, rate in wall_ablation.items():
                if sp not in state:
                    state[sp] = np.zeros_like(state["rho"])
                state[sp][0, 0] += rate * dt
        return state

    # ------------------------------------------------------------------
    #   Numerical flux
    # ------------------------------------------------------------------
    def _interface_flux(self, UL: np.ndarray, UR: np.ndarray, axis: int) -> np.ndarray:
        """Return flux across cell interfaces.

        Parameters
        ----------
        UL, UR:
            Conserved variables to the left and right of the interface with
            shape ``(8, ...)``.
        axis:
            ``0`` for the radial direction and ``1`` for the axial (``z``)
            direction.
        """

        # Normal component of the magnetic field must be single valued at
        # the interface.  We take the arithmetic mean.
        Bn = 0.5 * (UL[4 + axis * 2] + UR[4 + axis * 2])

        # Compute physical fluxes for each side
        FL = self._phys_flux(UL, axis)
        FR = self._phys_flux(UR, axis)

        # Estimate signal speeds using Davis' method
        SL, SR = self._signal_speeds(UL, UR, axis, Bn)

        # HLL flux -- the full HLLD solver is considerably more involved.  For
        # the purposes of the exercises in this kata the HLL flux provides a
        # robust and easy to reason about approximation.
        Sdiff = SR - SL
        flux = (SR * FL - SL * FR + SL * SR * (UR - UL)) / Sdiff
        return flux

    # ------------------------------------------------------------------
    def _phys_flux(self, U: np.ndarray, axis: int) -> np.ndarray:
        """Compute physical flux for states ``U`` along ``axis``.

        The function implements the ideal MHD flux functions.  ``axis`` selects
        which momentum and magnetic field components represent the normal
        direction.
        """

        rho = U[0]
        mom_r, mom_phi, mom_z = U[1], U[2], U[3]
        Br, Bphi, Bz = U[4], U[5], U[6]
        E = U[7]

        # Velocities
        vr = mom_r / rho
        vphi = mom_phi / rho
        vz = mom_z / rho

        Bsq = Br**2 + Bphi**2 + Bz**2
        vsq = vr**2 + vphi**2 + vz**2
        p = (self.gamma - 1.0) * (E - 0.5 * rho * vsq - 0.5 * Bsq)
        ptot = p + 0.5 * Bsq

        if axis == 0:
            vn, vt1, vt2 = vr, vphi, vz
            Bn, Bt1, Bt2 = Br, Bphi, Bz
        else:
            vn, vt1, vt2 = vz, vphi, vr
            Bn, Bt1, Bt2 = Bz, Bphi, Br

        flux = np.empty_like(U)
        flux[0] = rho * vn
        flux[1] = mom_r * vn
        flux[2] = mom_phi * vn
        flux[3] = mom_z * vn
        flux[4] = Br * vn
        flux[5] = Bphi * vn
        flux[6] = Bz * vn
        flux[7] = (E + ptot) * vn - Bn * (vr * Br + vphi * Bphi + vz * Bz)

        # Momentum terms depend on orientation
        if axis == 0:
            flux[1] += p + 0.5 * (Bphi**2 + Bz**2) - 0.5 * Br**2
            flux[2] -= Br * Bphi
            flux[3] -= Br * Bz
            flux[4] = 0.0
            flux[5] = vn * Bphi - vt1 * Br
            flux[6] = vn * Bz - vt2 * Br
        else:
            flux[3] += p + 0.5 * (Br**2 + Bphi**2) - 0.5 * Bz**2
            flux[1] -= Bz * Br
            flux[2] -= Bz * Bphi
            flux[6] = 0.0
            flux[4] = vn * Br - vt2 * Bz
            flux[5] = vn * Bphi - vt1 * Bz
        return flux

    # ------------------------------------------------------------------
    def _signal_speeds(
        self, UL: np.ndarray, UR: np.ndarray, axis: int, Bn: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate minimal and maximal wave speeds for HLL flux."""

        def cfast(state: np.ndarray) -> np.ndarray:
            rho = state[0]
            mom_r, mom_phi, mom_z = state[1], state[2], state[3]
            Br, Bphi, Bz = state[4], state[5], state[6]
            E = state[7]

            vr = mom_r / rho
            vphi = mom_phi / rho
            vz = mom_z / rho

            Bsq = Br**2 + Bphi**2 + Bz**2
            vsq = vr**2 + vphi**2 + vz**2
            p = (self.gamma - 1.0) * (E - 0.5 * rho * vsq - 0.5 * Bsq)
            a2 = self.gamma * p / rho
            b2 = (Br**2 + Bphi**2 + Bz**2) / rho
            disc = (a2 + b2) ** 2 - 4 * a2 * Bn**2 / rho
            cf2 = 0.5 * (a2 + b2 + np.sqrt(np.maximum(disc, 0.0)))
            if axis == 0:
                vn = vr
            else:
                vn = vz
            return vn, np.sqrt(cf2)

        vL, cfL = cfast(UL)
        vR, cfR = cfast(UR)
        SL = np.minimum(vL - cfL, vR - cfR)
        SR = np.maximum(vL + cfL, vR + cfR)
        return SL, SR

    # ------------------------------------------------------------------
    def _constrained_transport(
        self, U: np.ndarray, dt: float, dr: float, dz: float
    ) -> None:
        """Update magnetic field using constrained transport.

        The method computes the edge centred electric field using the ideal
        Ohm's law ``E = -v x B`` and updates the radial and axial magnetic field
        components such that ``div(B)`` remains zero to machine precision.
        """

        rho = U[0]
        vr = U[1] / rho
        vz = U[3] / rho
        Br = U[4]
        Bz = U[6]

        # Edge centred electric field and curl update.  Loops are used here to
        # keep the implementation compact and easy to reason about.  The grid
        # sizes used in the unit tests are very small so the loops are more than
        # adequate.
        nr, nz = Br.shape
        E = np.zeros((nr + 1, nz + 1))
        for i in range(1, nr):
            for j in range(1, nz):
                vr_avg = np.mean(vr[i - 1 : i + 1, j - 1 : j + 1])
                vz_avg = np.mean(vz[i - 1 : i + 1, j - 1 : j + 1])
                Br_avg = np.mean(Br[i - 1 : i + 1, j - 1 : j + 1])
                Bz_avg = np.mean(Bz[i - 1 : i + 1, j - 1 : j + 1])
                E[i, j] = vr_avg * Bz_avg - vz_avg * Br_avg

        for i in range(1, nr - 1):
            for j in range(1, nz - 1):
                Br[i, j] -= dt / dz * (E[i, j + 1] - E[i, j])
                Bz[i, j] += dt / dr * (E[i + 1, j] - E[i, j])

        U[4] = Br
        U[6] = Bz
