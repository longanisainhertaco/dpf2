"""Three-dimensional resistive magnetohydrodynamics utilities.

This module provides a light–weight representation of the conservative
three–dimensional resistive MHD equations.  The formulation implemented here
extends the original two–dimensional system used in the project to full
``x-y-z`` support and incorporates a number of diffusive and geometric
processes that are frequently required in plasma simulations:

* Anisotropic viscosity and thermal conduction (coefficients may be zero to
  recover the ideal system).
* Simple resistive (Ohmic) dissipation.
* Optional geometric source terms for cylindrical coordinates.
* Hyperbolic divergence cleaning following the approach of Dedner et al.

The implementation is intentionally compact; it is *not* intended to be a
high–performance solver but rather to supply the minimal building blocks for
unit and integration tests throughout the code base.
"""

from __future__ import annotations

from dataclasses import dataclass
try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for test environment
    import types, math

    np = types.SimpleNamespace(
        array=lambda x: list(x) if isinstance(x, (list, tuple)) else [x],
        zeros=lambda shape: [[0.0] * shape[1] for _ in range(shape[0])] if isinstance(shape, tuple) else [0.0] * shape,
        sqrt=math.sqrt,
        dot=lambda a, b: sum(x * y for x, y in zip(a, b)),
        zeros_like=lambda arr: [0.0 for _ in arr],
        ndarray=list,
    )

from ..radiation.multigroup import MultiGroupDiffusion

from ..mesh import Mesh2D, Mesh3D


@dataclass
class ResistiveMHD:
    """Conservative 3‑D resistive MHD system with optional physics extensions."""

    gamma: float = 5 / 3
    eta: float = 0.0
    mu_parallel: float = 0.0
    mu_perp: float = 0.0
    kappa_parallel: float = 0.0
    kappa_perp: float = 0.0
    c_h: float = 0.0
    c_p: float = 0.0

    def __post_init__(self) -> None:
        # Order of the conservative variables used throughout the class.
        self.equations = [
            "density",
            "momentum_x",
            "momentum_y",
            "momentum_z",
            "energy",
            "B_x",
            "B_y",
            "B_z",
            "psi",  # divergence cleaning variable
        ]

    # ------------------------------------------------------------------
    # Primitive ↔ conservative conversions
    # ------------------------------------------------------------------
    def conservative_variables(self, primitives: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative form.

        Parameters
        ----------
        primitives:
            Array of ``[rho, v_x, v_y, v_z, p, B_x, B_y, B_z]``.  The cleaning
            variable ``psi`` is assumed to be zero in primitive form and is
            appended automatically.
        """

        rho, v_x, v_y, v_z, p, B_x, B_y, B_z = primitives
        kinetic = 0.5 * rho * (v_x ** 2 + v_y ** 2 + v_z ** 2)
        magnetic = 0.5 * (B_x ** 2 + B_y ** 2 + B_z ** 2)
        energy = p / (self.gamma - 1.0) + kinetic + magnetic
        return np.array(
            [rho, rho * v_x, rho * v_y, rho * v_z, energy, B_x, B_y, B_z, 0.0]
        )

    def _pressure(self, U: np.ndarray) -> float:
        """Return the thermal pressure from conservative variables."""

        rho, m_x, m_y, m_z, E, B_x, B_y, B_z, _ = U
        v_x = m_x / rho
        v_y = m_y / rho
        v_z = m_z / rho
        v2 = v_x ** 2 + v_y ** 2 + v_z ** 2
        B2 = B_x ** 2 + B_y ** 2 + B_z ** 2
        return (E - 0.5 * rho * v2 - 0.5 * B2) * (self.gamma - 1.0)

    # ------------------------------------------------------------------
    # Mesh-aware helpers
    # ------------------------------------------------------------------
    def stable_timestep(
        self, U: np.ndarray, mesh: Mesh2D | Mesh3D, cfl: float = 0.8
    ) -> float:
        """Return a CFL-limited stable timestep for ``mesh``.

        The routine inspects the mesh spacing in each coordinate direction and
        computes an estimate of the maximum allowable timestep using the
        fastest characteristic speeds of the system.  The implementation is
        intentionally simple but works for both :class:`~dpf2.mesh.Mesh2D`
        and :class:`~dpf2.mesh.Mesh3D` instances.
        """

        if isinstance(mesh, Mesh3D):
            spacings = [mesh.dx, mesh.dy, mesh.dz]
            directions = ["x", "y", "z"]
        else:  # Mesh2D treated as x-z
            spacings = [mesh.dr, mesh.dz]
            directions = ["x", "z"]

        speeds = [self.max_speed(U, d) for d in directions]
        dt = min(s / v if v > 0 else np.inf for s, v in zip(spacings, speeds))
        return cfl * dt

    # ------------------------------------------------------------------
    # Fluxes
    # ------------------------------------------------------------------
    def flux_function(self, U: np.ndarray, direction: str) -> np.ndarray:
        """Compute MHD fluxes in the requested ``direction``.

        The returned array has the same length as the conservative state
        vector.  Divergence cleaning is incorporated via the ``psi`` variable
        whose flux couples to the magnetic field components.
        """

        rho, m_x, m_y, m_z, E, B_x, B_y, B_z, psi = U
        v_x = m_x / rho
        v_y = m_y / rho
        v_z = m_z / rho
        v = np.array([v_x, v_y, v_z])
        B = np.array([B_x, B_y, B_z])
        B2 = np.dot(B, B)
        p = self._pressure(U)
        total_p = p + 0.5 * B2
        Bdotv = np.dot(B, v)

        F = np.zeros_like(U)

        if direction == "x":
            F[0] = m_x
            F[1] = m_x * v_x + total_p - B_x ** 2
            F[2] = m_y * v_x - B_x * B_y
            F[3] = m_z * v_x - B_x * B_z
            F[4] = (E + total_p) * v_x - B_x * Bdotv
            F[5] = psi
            F[6] = v_y * B_x - v_x * B_y
            F[7] = v_z * B_x - v_x * B_z
            F[8] = self.c_h ** 2 * B_x
            return F

        if direction == "y":
            F[0] = m_y
            F[1] = m_x * v_y - B_y * B_x
            F[2] = m_y * v_y + total_p - B_y ** 2
            F[3] = m_z * v_y - B_y * B_z
            F[4] = (E + total_p) * v_y - B_y * Bdotv
            F[5] = v_x * B_y - v_y * B_x
            F[6] = psi
            F[7] = v_z * B_y - v_y * B_z
            F[8] = self.c_h ** 2 * B_y
            return F

        if direction == "z":
            F[0] = m_z
            F[1] = m_x * v_z - B_z * B_x
            F[2] = m_y * v_z - B_z * B_y
            F[3] = m_z * v_z + total_p - B_z ** 2
            F[4] = (E + total_p) * v_z - B_z * Bdotv
            F[5] = v_x * B_z - v_z * B_x
            F[6] = v_y * B_z - v_z * B_y
            F[7] = psi
            F[8] = self.c_h ** 2 * B_z
            return F

        raise ValueError("direction must be 'x', 'y' or 'z'")

    # ------------------------------------------------------------------
    # Characteristic speeds (for Rusanov fluxes used in tests)
    # ------------------------------------------------------------------
    def max_speed(self, U: np.ndarray, direction: str) -> float:
        """Return an estimate of the fastest signal speed in ``direction``."""

        rho, m_x, m_y, m_z, _, B_x, B_y, B_z, _ = U
        v = {
            "x": m_x / rho,
            "y": m_y / rho,
            "z": m_z / rho,
        }[direction]
        p = self._pressure(U)
        a = np.sqrt(self.gamma * p / rho)
        B2 = B_x ** 2 + B_y ** 2 + B_z ** 2
        c_a = np.sqrt(B2 / rho)
        return abs(v) + np.sqrt(a ** 2 + c_a ** 2)

    # ------------------------------------------------------------------
    # Radiation coupling
    # ------------------------------------------------------------------
    def apply_radiation(
        self, U: np.ndarray, radiation: MultiGroupDiffusion, dt: float
    ) -> None:
        """Couple the fluid energy to a multi-group radiation model.

        Parameters
        ----------
        U:
            Conservative state vector(s).  May be either a single
            vector of length ``len(self.equations)`` or a two-dimensional
            array with shape ``(n_cells, len(self.equations))``.
        radiation:
            Instance of :class:`~dpf2.radiation.multigroup.MultiGroupDiffusion`
            holding group energies and opacities.
        dt:
            Time step for the coupling.
        """

        idx = self.equations.index("energy")
        if not isinstance(U[0], (list, tuple)):
            updated = radiation.couple([U[idx]], dt)
            U[idx] = updated[0]
        else:
            energies = [row[idx] for row in U]
            updated = radiation.couple(energies, dt)
            for i, val in enumerate(updated):
                U[i][idx] = val

    # ------------------------------------------------------------------
    # Source terms
    # ------------------------------------------------------------------
    def source_terms(
        self,
        U: np.ndarray,
        *,
        grad_v: np.ndarray | None = None,
        grad_T: np.ndarray | None = None,
        geometry: str | None = None,
        coord: float | None = None,
    ) -> np.ndarray:
        """Return diffusive and geometric source terms.

        Parameters
        ----------
        U:
            Conservative state vector.
        grad_v:
            Optional velocity gradient ``\nabla v`` used for anisotropic
            viscosity.  It should be a ``3x3`` array where ``grad_v[i,j]`` is
            ``∂v_i/∂x_j``.
        grad_T:
            Optional temperature gradient used for anisotropic thermal
            conduction.
        geometry:
            If ``"cylindrical"`` a minimal set of geometric source terms are
            applied using ``coord`` as the radial coordinate.
        coord:
            Radial coordinate when ``geometry`` is ``"cylindrical"``.
        """

        rho, m_x, m_y, m_z, E, B_x, B_y, B_z, psi = U
        v = np.array([m_x, m_y, m_z]) / rho
        B = np.array([B_x, B_y, B_z])
        src = np.zeros_like(U)

        # Resistive dissipation
        B2 = np.dot(B, B)
        src[4] += self.eta * B2
        src[5:8] -= self.eta * B

        # Divergence cleaning damping
        src[8] -= self.c_p ** 2 * psi

        # Anisotropic viscosity
        if grad_v is not None and (self.mu_parallel or self.mu_perp):
            grad_v = np.asarray(grad_v)
            b_hat = B / (np.linalg.norm(B) + 1.0e-20)
            grad_para = np.outer(b_hat, grad_v @ b_hat)
            grad_perp = grad_v - grad_para
            visc_tensor = self.mu_parallel * grad_para + self.mu_perp * grad_perp
            force = np.sum(visc_tensor, axis=1)
            src[1:4] += force
            src[4] += np.dot(force, v)

        # Anisotropic thermal conduction
        if grad_T is not None and (self.kappa_parallel or self.kappa_perp):
            grad_T = np.asarray(grad_T)
            b_hat = B / (np.linalg.norm(B) + 1.0e-20)
            grad_para_T = np.dot(grad_T, b_hat) * b_hat
            grad_perp_T = grad_T - grad_para_T
            heat_flux = -self.kappa_parallel * grad_para_T - self.kappa_perp * grad_perp_T
            src[4] -= np.sum(heat_flux)

        # Geometric source terms (very simplified) for cylindrical coordinates
        if geometry == "cylindrical" and coord is not None and coord != 0.0:
            r = coord
            v_x, v_y, v_z = v
            p = self._pressure(U)
            src[0] += -rho * v_x / r
            src[1] += (rho * (v_y ** 2 + v_z ** 2) + 0.5 * (B_y ** 2 + B_z ** 2) - B_x ** 2 + p) / r
            src[4] += ((E + p + 0.5 * B2) * v_x - B_x * (B @ v)) / r

        return src


__all__ = ["ResistiveMHD"]

