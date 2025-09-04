"""Hall-MHD solver operating on three-dimensional grids.

The solver advances mass density, momentum and magnetic fields using a
second-order Godunov scheme with a constrained-transport magnetic field
update to preserve ``∇·B = 0``.  Optional Braginskii transport
coefficients supply parallel viscosity and thermal conduction.  The
implementation also features lightweight MPI domain decomposition and
hooks for adaptive-mesh refinement so that applications can refine the
mesh or exchange ghost cells between ranks.  While still compact, the
module serves as a functional reference used by the regression tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple
from typing import Protocol

import logging
import numpy as np
try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import mu_0
except Exception:  # pragma: no cover
    mu_0 = 4e-7 * np.pi

try:  # pragma: no cover - physical constants
    from scipy.constants import e as q_e, m_e, m_p
except Exception:  # pragma: no cover
    q_e = 1.602176634e-19
    m_e = 9.1093837015e-31
    m_p = 1.67262192369e-27

try:  # pragma: no cover - MPI is optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None

from dpf2.core.bases import CircuitSolverBase, PlasmaSolverBase, CouplingState
from .eos import EOSBase, IdealGasEOS
from .boundary_conditions import KineticSheath
from .physics.energy import EnergyTracker
from .diagnostics.quality_dashboard import QualityDashboard
from .diagnostics.modes import azimuthal_mode_spectrum

logger = logging.getLogger(__name__)

class ChemistryModule(Protocol):
    """Minimal interface for chemistry plugins."""

    def ionization_state(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return mean charge state."""


class RadiationModule(Protocol):  # pragma: no cover - docs only
    """Minimal interface for radiation plugins."""

    def loss(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return volumetric energy-loss rate."""

    def opacity(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray: ...

    def emissivity(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray: ...

    def couple(self, energy: list[float], dt: float) -> list[float]: ...


try:  # pragma: no cover - chemistry is optional in tests
    from .chemistry import ChemistryModel, SahaEquilibrium
except Exception:  # pragma: no cover
    class ChemistryModel(ChemistryModule):
        def ionization_state(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
            return np.ones_like(rho)

    class SahaEquilibrium(ChemistryModel):
        pass

try:  # pragma: no cover - radiation package optional
    from .radiation import RadiationBase
except Exception:  # pragma: no cover
    class RadiationBase(RadiationModule):  # type: ignore[misc]
        def loss(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
            return np.zeros_like(rho)


__all__ = ["MHDState", "HallMHDSolver", "spitzer_resistivity"]


def spitzer_resistivity(ne: np.ndarray, Te: np.ndarray, Z: float | np.ndarray) -> np.ndarray:
    """Return classical Spitzer resistivity in ``Ω·m``.

    Parameters
    ----------
    ne:
        Electron number density ``[m⁻³]``.  The standard Spitzer formula is
        independent of density when the Coulomb logarithm is treated as
        constant, but the argument is kept for API completeness.
    Te:
        Electron temperature ``[K]``.
    Z:
        Effective ion charge state.

    Returns
    -------
    ndarray
        Spitzer resistivity ``η`` in SI units.  A fixed Coulomb logarithm of
        ``lnΛ = 10`` is assumed, yielding the characteristic ``η ∝ T_e^{-3/2}``
        scaling used in the tests.
    """

    try:  # pragma: no cover - allow running without SciPy
        from scipy.constants import e, epsilon_0, k, m_e
    except Exception:  # pragma: no cover
        e = 1.602176634e-19
        epsilon_0 = 8.8541878128e-12
        k = 1.380649e-23
        m_e = 9.1093837015e-31

    ne = np.asarray(ne)
    Te = np.asarray(Te)
    Z = np.asarray(Z)

    ln_lambda = 10.0  # typical value for many laboratory plasmas
    coeff = (
        4 * np.sqrt(2 * np.pi) * np.sqrt(m_e) * e**2 * ln_lambda
        / (3 * (4 * np.pi * epsilon_0) ** 2 * k ** 1.5)
    )
    return coeff * Z / (Te ** 1.5)


def _dd(f: np.ndarray, axis: int) -> np.ndarray:
    """Finite-volume forward difference with periodic boundaries."""
    return np.roll(f, -1, axis) - f


def _divergence(vec: np.ndarray) -> np.ndarray:
    """Compute a finite-volume divergence that preserves ``∇·(∇×A)=0``."""
    dims = vec.ndim - 1
    result = _dd(vec[..., 0], 0)
    if dims > 1:
        result += _dd(vec[..., 1], 1)
    if dims > 2:
        result += _dd(vec[..., 2], 2)
    return result


def _curl(vec: np.ndarray) -> np.ndarray:
    """Compute a finite-volume curl consistent with the divergence kernel."""
    dims = vec.ndim - 1

    def d(a: np.ndarray, ax: int) -> np.ndarray:
        return _dd(a, ax) if ax < dims else np.zeros_like(a)

    cx = d(vec[..., 2], 1) - d(vec[..., 1], 2)
    cy = d(vec[..., 0], 2) - d(vec[..., 2], 0)
    cz = d(vec[..., 1], 0) - d(vec[..., 0], 1)
    return np.stack((cx, cy, cz), axis=-1)


def _project_div_free(B: np.ndarray) -> np.ndarray:
    """Project a magnetic field onto its divergence-free component."""
    spatial_shape = B.shape[:-1]
    dims = len(spatial_shape)
    if dims == 2:
        B = B.reshape(spatial_shape + (1, 3))
    nx, ny, nz, _ = B.shape
    B_hat = np.fft.fftn(B, axes=(0, 1, 2))
    kx = 2 * np.pi * np.fft.fftfreq(nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny)
    kz = 2 * np.pi * np.fft.fftfreq(nz)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    k_dot_B = kx * B_hat[..., 0] + ky * B_hat[..., 1] + kz * B_hat[..., 2]
    for i, k in enumerate((kx, ky, kz)):
        B_hat[..., i] -= k * k_dot_B / k2
    B_proj = np.fft.ifftn(B_hat, axes=(0, 1, 2)).real
    return B_proj.reshape(spatial_shape + (3,))


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod limiter used for MUSCL reconstruction."""
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def _hll_flux(
    gamma: float,
    i: int,
    rho_L: np.ndarray,
    v_L: np.ndarray,
    B_L: np.ndarray,
    p_L: np.ndarray,
    rho_R: np.ndarray,
    v_R: np.ndarray,
    B_R: np.ndarray,
    p_R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return HLL fluxes for direction ``i``.

    The function implements a simple two-wave Harten–Lax–van Leer
    Riemann solver used by the CTU update.  It returns the mass, momentum
    and energy fluxes at the interface between ``L`` and ``R`` states.
    """

    B2_L = np.sum(B_L**2, axis=-1)
    B2_R = np.sum(B_R**2, axis=-1)
    kinetic_L = 0.5 * rho_L * np.sum(v_L**2, axis=-1)
    kinetic_R = 0.5 * rho_R * np.sum(v_R**2, axis=-1)
    energy_L = p_L / (gamma - 1.0) + kinetic_L + 0.5 * B2_L
    energy_R = p_R / (gamma - 1.0) + kinetic_R + 0.5 * B2_R

    mom_L = rho_L[..., None] * v_L
    mom_R = rho_R[..., None] * v_R

    vdotB_L = np.sum(v_L * B_L, axis=-1)
    vdotB_R = np.sum(v_R * B_R, axis=-1)

    F_rho_L = rho_L * v_L[..., i]
    F_rho_R = rho_R * v_R[..., i]

    F_mom_L = np.zeros_like(mom_L)
    F_mom_R = np.zeros_like(mom_R)
    for j in range(3):
        F_mom_L[..., j] = mom_L[..., j] * v_L[..., i]
        F_mom_R[..., j] = mom_R[..., j] * v_R[..., i]
        if i == j:
            F_mom_L[..., j] += p_L + 0.5 * B2_L
            F_mom_R[..., j] += p_R + 0.5 * B2_R
        F_mom_L[..., j] -= B_L[..., i] * B_L[..., j]
        F_mom_R[..., j] -= B_R[..., i] * B_R[..., j]

    F_energy_L = (energy_L + p_L + 0.5 * B2_L) * v_L[..., i] - vdotB_L * B_L[..., i]
    F_energy_R = (energy_R + p_R + 0.5 * B2_R) * v_R[..., i] - vdotB_R * B_R[..., i]

    cs_L = np.sqrt(gamma * p_L / rho_L)
    cs_R = np.sqrt(gamma * p_R / rho_R)
    ca_L = np.sqrt(B2_L / rho_L)
    ca_R = np.sqrt(B2_R / rho_R)
    cfast_L = np.sqrt(cs_L**2 + ca_L**2)
    cfast_R = np.sqrt(cs_R**2 + ca_R**2)
    SL = np.minimum(v_L[..., i] - cfast_L, v_R[..., i] - cfast_R)
    SR = np.maximum(v_L[..., i] + cfast_L, v_R[..., i] + cfast_R)

    U_L = np.stack((rho_L, mom_L[..., 0], mom_L[..., 1], mom_L[..., 2], energy_L), axis=-1)
    U_R = np.stack((rho_R, mom_R[..., 0], mom_R[..., 1], mom_R[..., 2], energy_R), axis=-1)
    F_L = np.stack((F_rho_L, F_mom_L[..., 0], F_mom_L[..., 1], F_mom_L[..., 2], F_energy_L), axis=-1)
    F_R = np.stack((F_rho_R, F_mom_R[..., 0], F_mom_R[..., 1], F_mom_R[..., 2], F_energy_R), axis=-1)

    denom = SR - SL
    denom[denom == 0] = 1e-30
    flux = np.where(
        (SL[..., None] >= 0),
        F_L,
        np.where(
            (SR[..., None] <= 0),
            F_R,
            (
                SR[..., None] * F_L
                - SL[..., None] * F_R
                + SL[..., None] * SR[..., None] * (U_R - U_L)
            )
            / denom[..., None],
        ),
    )

    return flux[..., 0], flux[..., 1:4], flux[..., 4]


@dataclass
class MHDState:
    """State container for the MHD variables.

    Attributes
    ----------
    rho : ndarray
        Mass density [kg/m^3].
    mom : ndarray
        Momentum density vector [kg/m^2/s].
    energy : ndarray
        Total energy density [J/m^3].
    B : ndarray
        Magnetic field vector [T].
    temperatures : dict[str, ndarray] | None
        Mapping of species names to temperature fields.
    Te : ndarray | None
        Electron temperature [K] when using two-temperature models.
    Ti : ndarray | None
        Ion temperature [K] when using two-temperature models.
    eta : ndarray | None
        Resistivity [Ω·m] when using spatially varying resistivity.
    """

    rho: np.ndarray
    mom: np.ndarray
    energy: np.ndarray
    B: np.ndarray
    psi: np.ndarray | None = None
    Te: np.ndarray | None = None
    Ti: np.ndarray | None = None
    eta: np.ndarray | None = None
    temperatures: Dict[str, np.ndarray] | None = None


@dataclass
class HallMHDSolver(PlasmaSolverBase):
    """Stub for a 3-D Hall-MHD solver with CT and AMR hooks.

    The solver now uses a second-order Godunov (MUSCL-Hancock)
    update with a constrained-transport magnetic-field advance and
    an optional Hall term.  Boundary-condition and AMR refinement
    hooks allow external customization of the update.

    Parameters
    ----------
    mesh : Any
        Placeholder for mesh/AMR hierarchy object.
    """

    mesh: Any = field(default=None)
    eos: EOSBase = field(default_factory=IdealGasEOS)
    chemistry: ChemistryModule = field(default_factory=SahaEquilibrium)
    radiation: RadiationModule | None = None
    sheath: KineticSheath | None = None
    eta: float = 0.0
    hall_coeff: float = 0.0
    rad_coeff: float = 0.0
    nu_par: float = 0.0
    kappa_par: float = 0.0
    nu: float = 0.0  # isotropic viscosity
    c_h: float = 0.0  # hyperbolic cleaning speed
    c_p: float = 0.0  # parabolic cleaning rate
    bc: Callable[[MHDState], None] | None = None
    refine: Callable[[MHDState], None] | None = None
    circuit: CircuitSolverBase | None = None
    comm: Any | None = None  # MPI communicator for domain decomposition
    amr: Any | None = None  # Optional AMReX mesh object
    braginskii: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float]] | None = None
    electron_inertia: float = 0.0
    hall_threshold: float = 1.0
    ei_threshold: float = 0.1
    scale_length: float = 1.0
    current: float = 0.0
    inductance: float = 0.0
    back_emf: float = 0.0
    circuit_feedback: CouplingState | None = field(init=False, default=None)
    anomalous_resistivity: Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray]] | None = None
    lower_hybrid_drift: Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray]] | None = None
    m0_instability: Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray]] | None = None
    instability_thresholds: Dict[str, float] | None = None
    sausage_onset: bool = field(init=False, default=False)
    kink_onset: bool = field(init=False, default=False)
    voltage_spikes: list[float] = field(default_factory=list)
    impedance_growth: list[float] = field(default_factory=list)
    last_voltage_spike: float = field(init=False, default=0.0)
    last_lh_power: float = field(init=False, default=0.0)
    last_eta_anom_mean: float = field(init=False, default=0.0)
    last_eta_total_mean: float = field(init=False, default=0.0)
    last_pressure: np.ndarray | None = field(init=False, default=None)
    last_ionization: np.ndarray | None = field(init=False, default=None)
    last_rad_loss: np.ndarray | None = field(init=False, default=None)
    last_divB: np.ndarray | None = field(init=False, default=None)
    last_J: np.ndarray | None = field(init=False, default=None)
    last_E: np.ndarray | None = field(init=False, default=None)
    last_E_anom: np.ndarray | None = field(init=False, default=None)
    last_opacity: np.ndarray | None = field(init=False, default=None)
    last_emissivity: np.ndarray | None = field(init=False, default=None)
    hall_active: bool = field(init=False, default=False)
    electron_inertia_active: bool = field(init=False, default=False)
    last_wce_tau_e: float = field(init=False, default=0.0)
    last_di_over_L: float = field(init=False, default=0.0)
    cart_comm: Any | None = field(init=False, default=None)
    quality: QualityDashboard | None = None
    step_count: int = 0

    def __post_init__(self) -> None:
        """Initialise MPI cartesian communicator for domain decomposition."""
        if self.comm is not None and MPI is not None:
            try:
                size = self.comm.Get_size()
                dims = MPI.Compute_dims(size, [0, 0, 0])
                self.cart_comm = self.comm.Create_cart(dims, periods=[True, True, True])
            except Exception:  # pragma: no cover - fall back to original comm
                self.cart_comm = self.comm

    def apply_boundary_conditions(self, state: MHDState) -> None:
        """Invoke the boundary-condition hook if provided."""
        if self.bc is not None:
            self.bc(state)

    def compute_anomalous_resistivity(self, J: np.ndarray) -> np.ndarray:
        """Evaluate anomalous resistivity models and record voltage spikes.

        In addition to the base ``anomalous_resistivity`` callback, optional
        ``lower_hybrid_drift`` and ``m0_instability`` modules may contribute to
        the resistivity and supply axial electric-field components.  All
        electric-field contributions are stored on ``last_E_anom`` for use by
        :meth:`step` while the combined resistivity is returned.
        """

        eta = np.zeros(J.shape[:-1])
        E = np.zeros_like(J)
        s_eta = np.zeros(J.shape[:-1])
        s_E = np.zeros_like(J)

        if hasattr(np, "abs"):
            mag = np.abs(J[..., 0]) + np.abs(J[..., 1]) + np.abs(J[..., 2])
        else:  # pragma: no cover - very small stub fallback
            mag = [abs(j[0]) + abs(j[1]) + abs(j[2]) for j in J]

        def _process(result: np.ndarray | tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            if isinstance(result, tuple):
                return result
            return result, np.zeros(J.shape[:-1])

        def _accumulate(e_eta: np.ndarray, e_E: np.ndarray, *, axial: bool = False) -> None:
            nonlocal eta, E
            eta += e_eta
            if e_E.ndim == E.ndim - 1:
                if axial:
                    E[..., 2] += e_E
                else:
                    E += e_E[..., None]
            else:
                E += e_E

        if self.anomalous_resistivity is not None:
            e_eta, e_E = _process(self.anomalous_resistivity(J))
            _accumulate(e_eta, e_E)

        if self.lower_hybrid_drift is not None:
            res = self.lower_hybrid_drift(J)
            e_eta, e_E = _process(res)
            _accumulate(e_eta, e_E, axial=True)
            s_eta += e_eta
            if e_E.ndim == E.ndim - 1:
                s_E[..., 2] += e_E
            else:
                s_E += e_E
            if hasattr(self.lower_hybrid_drift, "power"):
                try:
                    self.last_lh_power = float(np.max(self.lower_hybrid_drift.power()))
                except Exception:  # pragma: no cover - power optional
                    self.last_lh_power = 0.0
        else:
            self.last_lh_power = 0.0


        if self.instability_thresholds:
            self._check_instability_onset(J)

        if hasattr(np, "abs"):
            mag = np.abs(J[..., 0]) + np.abs(J[..., 1]) + np.abs(J[..., 2])
        else:  # pragma: no cover - very small stub fallback
            mag = [abs(j[0]) + abs(j[1]) + abs(j[2]) for j in J]

        spike = float(
            max(
                np.max(s_eta * mag),
                np.max(np.abs(s_E[..., 2]))
            )
        )
        if spike != 0.0:
            self.voltage_spikes.append(spike)
        self.last_voltage_spike = spike
        self.last_E_anom = E
        try:
            self.last_eta_anom_mean = float(np.mean(eta))
        except Exception:
            self.last_eta_anom_mean = 0.0
        return eta

    # ------------------------------------------------------------------
    def _check_instability_onset(self, J: np.ndarray) -> None:
        """Check azimuthal mode amplitudes against configured thresholds."""

        if not self.instability_thresholds:
            return
        try:
            J_mag = np.linalg.norm(J, axis=-1)
        except Exception:  # pragma: no cover - very small stub fallback
            return
        spectrum = azimuthal_mode_spectrum(J_mag, axis=-1)
        if (
            not self.sausage_onset
            and "sausage" in self.instability_thresholds
            and len(spectrum) > 0
            and spectrum[0] >= self.instability_thresholds["sausage"]
        ):
            self.sausage_onset = True
        if (
            not self.kink_onset
            and "kink" in self.instability_thresholds
            and len(spectrum) > 1
            and spectrum[1] >= self.instability_thresholds["kink"]
        ):
            self.kink_onset = True

    def amr_refinement(self, state: MHDState) -> None:
        """Invoke the refinement callback if provided."""
        if self.refine is not None:
            self.refine(state)

    def _exchange_array(self, arr: np.ndarray) -> None:
        """Exchange ghost cells of ``arr`` with neighbouring MPI ranks."""
        cart = self.cart_comm
        if cart is None:
            return
        spatial_dims = min(3, arr.ndim)
        for axis in range(spatial_dims):
            src, dest = cart.Shift(axis, 1)
            if dest == MPI.PROC_NULL and src == MPI.PROC_NULL:
                continue
            send_hi = [slice(None)] * arr.ndim
            recv_hi = [slice(None)] * arr.ndim
            send_lo = [slice(None)] * arr.ndim
            recv_lo = [slice(None)] * arr.ndim
            send_hi[axis] = slice(-2, -1)
            recv_hi[axis] = slice(-1, None)
            send_lo[axis] = slice(1, 2)
            recv_lo[axis] = slice(0, 1)
            cart.Sendrecv(arr[tuple(send_hi)], dest=dest, recvbuf=arr[tuple(recv_hi)], source=src)
            cart.Sendrecv(arr[tuple(send_lo)], dest=src, recvbuf=arr[tuple(recv_lo)], source=dest)

    def exchange_boundaries(self, state: MHDState) -> None:
        """Synchronise ghost zones across MPI ranks if a communicator is set."""
        if self.comm is None or MPI is None:
            return
        for arr in [state.rho, state.mom, state.energy, state.B]:
            self._exchange_array(arr)

    def amr_sync(self, state: MHDState) -> None:
        """Invoke AMReX mesh synchronisation if available."""
        if self.amr is not None and hasattr(self.amr, "sync"):
            self.amr.sync(state)

    def step(
        self,
        state: MHDState,
        dt: float,
        current: float = 0.0,
        voltage: float = 0.0,
        energy_tracker: EnergyTracker | None = None,
    ) -> MHDState:  # pragma: no cover - skeleton
        """Advance the state by ``dt`` seconds using a higher-order MHD update.

        A MUSCL-Hancock Godunov scheme with a corner-transport-upwind
        (CTU) flux computation is employed for the fluid variables and a
        constrained-transport magnetic-field update maintains
        ``∇·B = 0``.  Hall physics is included through the electric field
        appearing in the CT step.  Optional Braginskii transport terms
        act along the magnetic-field direction.  Boundary-condition and
        AMR refinement hooks are invoked before and after the update.
        """

        self.amr_refinement(state)
        self.apply_boundary_conditions(state)
        self.exchange_boundaries(state)
        self.amr_sync(state)

        rho = state.rho.copy()
        mom = state.mom.copy()
        energy = state.energy.copy()
        B = state.B.copy()
        Te = None if state.Te is None else state.Te.copy()
        Ti = None if state.Ti is None else state.Ti.copy()
        temps = (
            {k: v.copy() for k, v in state.temperatures.items()}
            if state.temperatures is not None
            else {}
        )
        if Te is not None:
            temps.setdefault("Te", Te)
        if Ti is not None:
            temps.setdefault("Ti", Ti)
        eta_field = state.eta.copy() if state.eta is not None else self.eta

        dims = len(rho.shape)

        v = mom / rho[..., None]
        B2 = np.sum(B**2, axis=-1)
        kinetic = 0.5 * rho * np.sum(v**2, axis=-1)
        magnetic = 0.5 * B2
        e_internal = energy - kinetic - magnetic
        specific_e = e_internal / rho
        T = self.eos.temperature(rho, specific_e)
        p = self.eos.pressure(rho, T)
        zbar = self.chemistry.ionization_state(rho, T)
        if self.braginskii is not None:
            nu_p, kappa_p = self.braginskii(rho, T, np.sqrt(B2))
            self.nu_par = nu_p
            self.kappa_par = kappa_p

        if self.sheath is not None:
            ni = float(np.mean(rho)) / max(self.sheath.ion_mass, 1e-30)
            ne = ni * float(np.mean(zbar))
            thickness, imp_flux, ion_flux = self.sheath.evolve(ni, ne, float(np.mean(T)), dt)
            rho += imp_flux * dt * self.sheath.ion_mass
            energy -= self.rad_coeff * imp_flux * dt
            current = min(current, ion_flux)
            self.last_sheath = {"thickness": thickness, "impurity_flux": imp_flux}

        # --- High-order Godunov fluxes (MUSCL-Hancock) ---
        gamma = getattr(self.eos, "gamma", 5.0 / 3.0)
        flux_rho = np.zeros((dims,) + rho.shape)
        flux_mom = np.zeros((dims,) + mom.shape)
        flux_energy = np.zeros((dims,) + energy.shape)
        flux_temp: Dict[str, np.ndarray] = {
            name: np.zeros((dims,) + arr.shape) for name, arr in temps.items()
        }

        prim_vars = [rho, v[..., 0], v[..., 1], v[..., 2], B[..., 0], B[..., 1], B[..., 2], p]
        for i in range(dims):
            slopes = [
                _minmod(var - np.roll(var, 1, axis=i), np.roll(var, -1, axis=i) - var)
                for var in prim_vars
            ]
            left_states = [var - 0.5 * s for var, s in zip(prim_vars, slopes)]
            right_states = [
                np.roll(var, -1, axis=i) + 0.5 * np.roll(s, -1, axis=i)
                for var, s in zip(prim_vars, slopes)
            ]

            rho_L, vx_L, vy_L, vz_L, Bx_L, By_L, Bz_L, p_L = [ls for ls in left_states]
            rho_R, vx_R, vy_R, vz_R, Bx_R, By_R, Bz_R, p_R = [rs for rs in right_states]
            v_L = np.stack((vx_L, vy_L, vz_L), axis=-1)
            v_R = np.stack((vx_R, vy_R, vz_R), axis=-1)
            B_L = np.stack((Bx_L, By_L, Bz_L), axis=-1)
            B_R = np.stack((Bx_R, By_R, Bz_R), axis=-1)

            fr, fm, fe = _hll_flux(gamma, i, rho_L, v_L, B_L, p_L, rho_R, v_R, B_R, p_R)
            flux_rho[i] = fr
            flux_mom[i] = fm
            flux_energy[i] = fe
            for name in temps:
                flux_temp[name][i] = temps[name] * v[..., i]

        # Corner-transport-upwind transverse flux correction
        for i in range(dims):
            for j in range(dims):
                if i == j:
                    continue
                flux_rho[i] -= 0.5 * _dd(flux_rho[j], j)
                for k in range(3):
                    flux_mom[i][..., k] -= 0.5 * _dd(flux_mom[j][..., k], j)
                flux_energy[i] -= 0.5 * _dd(flux_energy[j], j)
                for name in temps:
                    flux_temp[name][i] -= 0.5 * _dd(flux_temp[name][j], j)

        drho = np.zeros_like(rho)
        denergy = np.zeros_like(energy)
        dmom = np.zeros_like(mom)
        dtemp: Dict[str, np.ndarray] = {name: np.zeros_like(arr) for name, arr in temps.items()}
        for i in range(dims):
            drho += flux_rho[i] - np.roll(flux_rho[i], 1, axis=i)
            denergy += flux_energy[i] - np.roll(flux_energy[i], 1, axis=i)
            for j in range(3):
                dmom[..., j] += flux_mom[i][..., j] - np.roll(flux_mom[i][..., j], 1, axis=i)
            for name in temps:
                dtemp[name] += flux_temp[name][i] - np.roll(flux_temp[name][i], 1, axis=i)

        rho -= dt * drho
        energy -= dt * denergy
        mom -= dt * dmom
        for name in temps:
            temps[name] -= dt * dtemp[name] / np.maximum(rho, 1e-30)

        if self.radiation is not None:
            if hasattr(self.radiation, "opacity"):
                self.last_opacity = self.radiation.opacity(rho, T)
            if hasattr(self.radiation, "emissivity"):
                self.last_emissivity = self.radiation.emissivity(rho, T)
            if hasattr(self.radiation, "couple"):
                energy_before = energy.copy()
                flat = energy_before.ravel().tolist()
                updated = self.radiation.couple(flat, dt)
                energy = np.array(updated).reshape(energy.shape)
                self.last_rad_loss = (energy_before - energy) / dt
            else:
                loss = self.radiation.loss(rho, T)
                energy -= dt * loss
                self.last_rad_loss = loss
            if hasattr(self.radiation, "diffuse"):
                self.radiation.diffuse(1.0, dt)
        else:
            self.last_rad_loss = None

        v = mom / rho[..., None]
        B2 = np.sum(B**2, axis=-1)
        kinetic = 0.5 * rho * np.sum(v**2, axis=-1)
        magnetic = 0.5 * B2
        e_internal = energy - kinetic - magnetic
        specific_e = e_internal / rho
        T = self.eos.temperature(rho, specific_e)
        p = self.eos.pressure(rho, T)
        zbar = self.chemistry.ionization_state(rho, T)
        self.last_pressure = p
        self.last_ionization = zbar
        # electron number density
        ne = rho * np.maximum(zbar, 1e-30)

        # --- Runtime activation checks ---
        eta_local = (
            eta_field if isinstance(eta_field, np.ndarray) else np.full(rho.shape, float(eta_field))
        )
        w_ce = q_e * np.sqrt(B2) / m_e
        tau_e = m_e / (ne * q_e**2 * np.maximum(eta_local, 1e-30))
        wce_tau_e = w_ce * tau_e
        self.last_wce_tau_e = float(np.max(wce_tau_e))
        self.hall_active = self.last_wce_tau_e > self.hall_threshold

        ni = rho
        d_i = np.sqrt(m_p / (mu_0 * ni * q_e**2))
        self.last_di_over_L = float(np.max(d_i) / max(self.scale_length, 1e-30))
        self.electron_inertia_active = self.last_di_over_L > self.ei_threshold

        # electron pressure gradient for Ohm's law
        pe = p * zbar / (1.0 + np.maximum(zbar, 1e-30))
        grad_pe = [_dd(pe, i) for i in range(dims)]
        while len(grad_pe) < 3:
            grad_pe.append(np.zeros_like(pe))
        grad_pe_vec = np.stack(grad_pe, axis=-1)

        # --- Constrained transport via electric fields ---
        J = _curl(B)
        eta_local = (
            eta_field
            if isinstance(eta_field, np.ndarray)
            else np.full(rho.shape, float(eta_field))
        )
        eta_anom = self.compute_anomalous_resistivity(J)
        eta_total = eta_local + eta_anom
        try:
            self.last_eta_total_mean = float(np.mean(eta_total))
        except Exception:
            self.last_eta_total_mean = 0.0
        E = -np.cross(v, B) + eta_total[..., None] * J - grad_pe_vec / ne[..., None]
        if self.last_E_anom is not None:
            E += self.last_E_anom
        if self.hall_active:
            coeff_hall = self.hall_coeff if self.hall_coeff != 0.0 else 1.0
            E += coeff_hall * np.cross(J, B) / ne[..., None]
        if self.electron_inertia_active:
            coeff_ei = self.electron_inertia
            if coeff_ei == 0.0:
                coeff_ei = m_e / (q_e**2 * ne)
            E += coeff_ei[..., None] * J
        B -= dt * _curl(E)

        # --- Divergence cleaning (hyperbolic/parabolic) ---
        psi = state.psi.copy() if state.psi is not None else np.zeros_like(rho)
        divB = _divergence(B)
        if self.c_h != 0.0 or self.c_p != 0.0:
            psi -= dt * (self.c_h ** 2 * divB + self.c_p * psi)
            grad_psi = [_dd(psi, i) for i in range(dims)]
            while len(grad_psi) < 3:
                grad_psi.append(np.zeros_like(psi))
            B -= dt * np.stack(grad_psi, axis=-1)
        B = _project_div_free(B)
        self.last_divB = divB

        B2 = np.sum(B**2, axis=-1)

        # --- Braginskii viscosity (parallel component) ---
        if self.nu_par != 0.0:
            b = B / np.sqrt(B2 + 1e-30)[..., None]
            for comp in range(3):
                grad_par = sum(b[..., i] * _dd(v[..., comp], i) for i in range(dims))
                visc_flux = self.nu_par * b * grad_par[..., None]
                mom[..., comp] += dt * sum(_dd(visc_flux[..., i], i) for i in range(dims))
                energy += dt * self.nu_par * grad_par**2 * rho

        # --- Braginskii thermal conduction (parallel) ---
        if self.kappa_par != 0.0:
            T = p / rho
            b = B / np.sqrt(B2 + 1e-30)[..., None]
            gradT_par = sum(b[..., i] * _dd(T, i) for i in range(dims))
            q = -self.kappa_par * b * gradT_par[..., None]
            energy -= dt * sum(_dd(q[..., i], i) for i in range(dims))

        # --- Isotropic viscosity ---
        if self.nu != 0.0:
            lap_v = np.stack(
                [sum(_dd(_dd(v[..., k], j), j) for j in range(dims)) for k in range(3)],
                axis=-1,
            )
            mom += dt * self.nu * rho[..., None] * lap_v
            energy += dt * self.nu * rho * np.sum(v * lap_v, axis=-1)

        # --- Source terms ---
        if isinstance(eta_total, np.ndarray):
            heating = eta_total * np.sum(J**2, axis=-1)
        else:
            heating = float(eta_total) * np.sum(J**2, axis=-1)
        energy += dt * heating
        for name in temps:
            temps[name] += dt * heating / np.maximum(rho, 1e-30)

        self.last_J = J
        self.last_E = E

        Te_out = temps.pop("Te", None)
        Ti_out = temps.pop("Ti", None)
        new_state = MHDState(
            rho=rho,
            mom=mom,
            energy=energy,
            B=B,
            psi=psi,
            Te=Te_out,
            Ti=Ti_out,
            eta=None if np.isscalar(eta_total) else eta_total,
            temperatures=temps if temps else None,
        )

        self.apply_boundary_conditions(new_state)
        self.amr_refinement(new_state)
        self.exchange_boundaries(new_state)
        self.amr_sync(new_state)

        # Expose plasma inductance and induced EMF for circuit coupling
        L_new = self.compute_plasma_inductance(new_state, current)
        self.inductance = L_new
        emf = 0.0
        self.back_emf = emf
        self.circuit_feedback = CouplingState(
            Lp=L_new, emf=emf, current=current, voltage=voltage
        )

        if self.circuit is not None:
            updated = self.circuit.step(self.circuit_feedback, self.last_voltage_spike, dt)
            self.current = updated.current
            self.back_emf = updated.voltage
        else:
            self.current = current

        self.impedance_growth.append(
            self.last_voltage_spike / (abs(self.current) + 1e-30)
        )

        if energy_tracker is not None:
            v_final = mom / rho[..., None]
            B2_final = np.sum(B ** 2, axis=-1)
            kinetic_final = 0.5 * rho * np.sum(v_final**2, axis=-1)
            magnetic_final = 0.5 * B2_final
            thermal_final = energy - kinetic_final - magnetic_final
            rad = (
                float(np.sum(self.last_rad_loss) * dt)
                if self.last_rad_loss is not None
                else 0.0
            )
            energy_tracker.add(
                kinetic=float(np.sum(kinetic_final)),
                thermal=float(np.sum(thermal_final)),
                magnetic=float(np.sum(magnetic_final)),
                radiative=rad,
            )

        if self.quality is not None:
            self.step_count += 1
            dx = getattr(self.mesh, "dx", 1.0)
            dy = getattr(self.mesh, "dy", dx)
            dz = getattr(self.mesh, "dz", dx)
            cell_size = min(dx, dy, dz)
            cell_volume = dx * dy * dz
            max_speed = float(np.max(np.linalg.norm(v_final, axis=-1)))
            cfl = max_speed * dt / cell_size if cell_size > 0 else 0.0
            ion_mass = getattr(self, "ion_mass", getattr(self.sheath, "ion_mass", 1.6726219e-27))
            ne = rho * np.maximum(zbar, 1e-30) / ion_mass
            try:
                from scipy.constants import e, epsilon_0, k
            except Exception:  # pragma: no cover
                e = 1.602176634e-19
                epsilon_0 = 8.8541878128e-12
                k = 1.380649e-23
            lambda_D = np.sqrt(epsilon_0 * k * T / (ne * e**2))
            ppc = float(np.mean(ne) * cell_volume)
            lh_power = self.last_lh_power
            impedance = self.last_eta_total_mean
            self.quality.log(
                self.step_count,
                dt,
                cell_size,
                ppc,
                cfl,
                float(np.mean(lambda_D)),
                amr_level=getattr(self, "amr_level", None),
                lower_hybrid_power=lh_power,
                plasma_impedance=impedance,
                divergence_error=getattr(self, "divergence_error", 0.0),
                energy_drift=getattr(self, "energy_drift", 0.0),

            )

        return new_state

    def compute_plasma_inductance(self, state: MHDState, current: float, cell_volume: float = 1.0) -> float:
        """Estimate plasma inductance from magnetic energy.

        Parameters
        ----------
        state : MHDState
            Current plasma state.
        current : float
            Circuit current in amperes.
        cell_volume : float, optional
            Volume of a single cell, by default 1.0.

        Returns
        -------
        float
            Estimated inductance in henries.
        """
        B2 = np.sum(state.B ** 2)
        magnetic_energy = B2 * cell_volume / (2 * mu_0)
        if current == 0.0:
            return 0.0
        return 2 * magnetic_energy / (current ** 2)
