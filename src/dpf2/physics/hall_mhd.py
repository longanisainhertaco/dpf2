from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np

try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import e as q_e, m_e, m_p, mu_0
except Exception:  # pragma: no cover
    q_e = 1.602176634e-19
    m_e = 9.1093837015e-31
    m_p = 1.67262192369e-27
    mu_0 = 4e-7 * np.pi

from .mhd import ResistiveMHD
from .anomalous_resistivity import SpectralResistivity as LHDIResistivity
from ..mesh import Mesh2D, Mesh3D
from ..core.bases import CouplingState


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
    electron_inertia: float = 0.0
    hall_enabled: bool = True
    omega_ce_tau_e_min: float = 1.0
    di_over_L_min: float = 0.01
    nernst: float = 0.0
    righi_leduc: float = 0.0
    hall_active: bool = field(default=False, init=False)
    current: float = 0.0
    back_emf: float = 0.0
    beam_velocity: float = 0.0

    # optional radiation loss model; when ``None`` no radiative cooling is applied
    radiation_model: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None

    # plasma inductance state (Henries)
    inductance: float = 0.0
    circuit_feedback: CouplingState | None = field(default=None, init=False)

    def update_transport(self, ne: float, Te: float, B: float, L: float) -> None:
        """Update transport coefficients and activation state.

        The Braginskii coefficients are calculated using very small subsets of
        the NRL Formulary expressions.  Activation is gated on the electron
        magnetisation ``ω_ce τ_e`` and the ratio of the ion inertial length to
        the system size ``d_i/L``.  When the Hall model is inactive the
        coefficients are set to zero to recover classical resistive MHD.
        """

        omega_tau, di_over_L = hall_parameters(ne, Te, B, L)
        self.omega_ce_tau_e = omega_tau
        self.di_over_L = di_over_L
        self.hall_active = (
            self.hall_enabled
            and omega_tau > self.omega_ce_tau_e_min
            and di_over_L > self.di_over_L_min
        )

        if not self.hall_active:
            self.eta = 0.0
            self.kappa_parallel = 0.0
            self.kappa_perp = 0.0
            self.nernst = 0.0
            self.righi_leduc = 0.0
            return

        coeffs = braginskii_coefficients(ne, Te, B)
        self.kappa_parallel = coeffs["kappa_parallel"]
        self.kappa_perp = coeffs["kappa_perp"]
        self.eta = coeffs["eta_parallel"]
        self.nernst = coeffs["nernst"]
        self.righi_leduc = coeffs["righi_leduc"]

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
        instability_amp: np.ndarray | float = 0.0,
        mfp: float = 0.0,
        tau_e: float = 0.0,
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
            ``circuit.step`` ``(CouplingState, back_emf, dt)`` interface. When provided
            the circuit is advanced using the
            computed feedback terms and the updated current/back‑EMF are stored
            on the model instance.
        """

        self.current = current

        Lp = self.plasma_inductance(state)

        amp = instability_amp
        if isinstance(amp, (list, tuple)):
            amp_sum = sum(amp)
        else:
            amp_sum = float(amp)
        emf = amp_sum
        self.beam_velocity = abs(amp_sum)

        # Store plasma feedback for the circuit solver.  ``emf`` represents
        # only additional plasma-induced voltages beyond the inductance change
        # which is handled directly by the circuit solver.
        self.inductance = Lp
        self.back_emf = emf
        self.circuit_feedback = CouplingState(
            Lp=Lp,
            emf=emf,
            current=self.current,
            mutual_inductance=0.0,
            back_reaction=0.0,
        )

        if circuit is not None:
            updated = circuit.step(self.circuit_feedback, 0.0, dt)
            self.current = updated.current

        self.log_regime(state, mfp, tau_e)

        return state

    # ------------------------------------------------------------------
    # Fluxes with Hall term and back EMF
    # ------------------------------------------------------------------
    def flux_function(
        self,
        U: np.ndarray,
        direction: str,
        J: np.ndarray | None = None,
        grad_pe: np.ndarray | None = None,
        ne: float | None = None,
    ) -> np.ndarray:
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
        grad_pe:
            Gradient of the electron pressure.  Only used when the Hall model
            is active.
        ne:
            Electron number density used for the ``∇p_e`` term.
        """

        F = super().flux_function(U, direction)

        if not self.hall_active:
            return F

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

        if J is not None and self.electron_inertia != 0.0:
            inertia_e = self.electron_inertia * J
            if direction == "x":
                F[6] += inertia_e[2]
                F[7] += inertia_e[1]
            elif direction == "y":
                F[5] += inertia_e[2]
                F[7] += inertia_e[0]
            elif direction == "z":
                F[5] += inertia_e[1]
                F[6] += inertia_e[0]

        if grad_pe is not None and ne is not None and ne > 0.0:
            E_pe = grad_pe / (ne * q_e)
            if direction == "x":
                F[6] -= E_pe[2]
                F[7] -= E_pe[1]
            elif direction == "y":
                F[5] -= E_pe[2]
                F[7] -= E_pe[0]
            elif direction == "z":
                F[5] -= E_pe[1]
                F[6] -= E_pe[0]

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
        divB = np.gradient(Bx, dx, edge_order=1)
        psi -= dt * (self.c_h ** 2 * divB + self.c_p ** 2 * psi)
        Bx -= dt * np.gradient(psi, dx, edge_order=1)
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
        J[:, 1] = -np.gradient(Bz, dx, edge_order=1)
        J[:, 2] = np.gradient(By, dx, edge_order=1)

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
            if self.radiation_model is not None:
                rho = U_new[i, 0]
                mom = U_new[i, 1:4]
                B = U_new[i, 5:8]
                v2 = float(np.dot(mom, mom)) / (rho * rho)
                B2 = float(np.dot(B, B))
                p = (U_new[i, 4] - 0.5 * rho * v2 - 0.5 * B2) * (self.gamma - 1.0)
                n = rho
                Te = p / (n + 1.0e-30)
                rad = float(self.radiation_model(n, n, Te))
                src[4] -= rad
            U_new[i] += dt * src

        self.divergence_cleaning(U_new, mesh, dt)
        return U_new


def nrl_braginskii(rho: np.ndarray, T: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return simple Braginskii transport coefficients using NRL scalings.

    The implementation is intentionally reduced – it merely provides smooth
    temperature and density dependent coefficients sufficient for the unit
    tests.  Both the viscosity and thermal conductivity scale with ``T^2`` and
    inversely with density, matching the trends of the full expressions.
    Parameters are accepted as ``ndarray`` and returned with matching shape.
    """

    rho = np.asarray(rho)
    T = np.asarray(T)
    coeff = 1.0 / np.maximum(rho, 1.0e-30)
    nu_par = 1.0e-4 * T**2 * coeff
    kappa_par = 1.0e-2 * T**2 * coeff
    return nu_par, kappa_par


# ---------------------------------------------------------------------------
# Extended transport helpers
# ---------------------------------------------------------------------------


def electron_collision_time(ne: float, Te: float, Z: float = 1.0, ln_lambda: float = 10.0) -> float:
    """Return the electron-ion collision time ``τ_e`` in seconds."""

    return 3.44e5 * (Te ** 1.5) / (Z * ne * ln_lambda)


def hall_parameters(ne: float, Te: float, B: float, L: float, Z: float = 1.0) -> tuple[float, float]:
    """Return ``ω_ce τ_e`` and ``d_i/L`` for gating decisions."""

    tau_e = electron_collision_time(ne, Te, Z)
    omega_ce = abs(q_e) * B / m_e
    di = np.sqrt(m_p / (mu_0 * ne * q_e ** 2))
    return omega_ce * tau_e, di / L


def braginskii_coefficients(ne: float, Te: float, B: float, Z: float = 1.0, ln_lambda: float = 10.0) -> Dict[str, float]:
    """Return a subset of Braginskii transport coefficients."""

    tau_e = electron_collision_time(ne, Te, Z, ln_lambda)
    omega_ce = abs(q_e) * B / m_e
    omega_tau = omega_ce * tau_e
    kappa_parallel = 3.16e-12 * Te ** 2.5 / (Z * ln_lambda)
    kappa_perp = kappa_parallel / (1.0 + omega_tau ** 2)
    eta_parallel = 1.03e-4 * Z * ln_lambda / (Te ** 1.5)
    eta_perp = eta_parallel * (1.0 + omega_tau ** 2)
    nernst = 0.81 * omega_tau / (1.0 + omega_tau ** 2)
    righi_leduc = omega_tau ** 2 / (1.0 + omega_tau ** 2)
    return {
        "kappa_parallel": kappa_parallel,
        "kappa_perp": kappa_perp,
        "eta_parallel": eta_parallel,
        "eta_perp": eta_perp,
        "nernst": nernst,
        "righi_leduc": righi_leduc,
    }


def whistler_dispersion(k: float, ne: float, B: float) -> float:
    """Return the whistler-wave frequency for wavenumber ``k``."""

    di = np.sqrt(m_p / (mu_0 * ne * q_e ** 2))
    omega_ci = abs(q_e) * B / m_p
    return omega_ci * (k * di) ** 2


def hall_shock_speed(B: float, ne: float, L: float) -> float:
    """Return a characteristic Hall shock speed used in tests."""

    vA = B / np.sqrt(mu_0 * m_p * ne)
    di = np.sqrt(m_p / (mu_0 * ne * q_e ** 2))
    return vA * (1.0 + di / L)


__all__ = [
    "HallMHD",
    "LHDIResistivity",
    "nrl_braginskii",
    "electron_collision_time",
    "hall_parameters",
    "braginskii_coefficients",
    "whistler_dispersion",
    "hall_shock_speed",
]
