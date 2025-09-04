from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import e, m_e, m_p
except Exception:  # pragma: no cover
    e = 1.602176634e-19
    m_e = 9.10938356e-31
    m_p = 1.67262192369e-27


def _to_array(val: Any) -> np.ndarray:
    """Return ``val`` as a floating point array.

    Mirrors :func:`_to_array` from ``m0_instability`` and tolerates the light
    weight ``numpy`` substitute used in the tests.
    """

    try:  # pragma: no cover - real NumPy path
        return np.asarray(val, dtype=float)
    except TypeError:  # pragma: no cover - ``numpy_stub`` path
        return np.asarray(val)


@dataclass
class LowerHybridDrift:
    """Minimal lower-hybrid drift instability model.

    The model evolves a perturbation amplitude representing the strength of
    lower‑hybrid waves and exposes an ``anomalous_resistivity`` callback
    compatible with :class:`~dpf2.hall_mhd_solver.HallMHDSolver`.  When the
    solver requests a resistivity field the stored amplitude is returned along
    with an optional electric‑field contribution.
    """

    B: float  # Magnetic field strength [T]
    n_i: float  # Ion number density [m^-3]
    amplitude: Any | None = None  # Latest perturbation amplitude
    m_i: float = m_p  # Ion mass [kg]
    energy: Any | None = None  # Stored wave energy
    last_k: Any | None = None  # Wave number of last evolution
    last_phase_velocity: Any | None = None  # Cached phase velocity

    def frequency(self) -> float:
        omega_ci = e * self.B / self.m_i
        omega_ce = e * self.B / m_e
        return abs(omega_ci * omega_ce) ** 0.5

    def growth_rate(self, k: Any) -> np.ndarray:
        ks = _to_array(k)
        omega_lh = self.frequency()
        rates = 0.1 * omega_lh * np.exp(-(ks * ks))
        return rates

    def evolve(self, amplitude: Any, k: Any, dt: float):
        amp = _to_array(amplitude)
        ks = _to_array(k)
        rate = _to_array(self.growth_rate(ks))
        evolved = amp * np.exp(np.clip(rate * dt, -50.0, 50.0))
        self.amplitude = evolved
        self.last_k = ks
        self.wave_energy()  # update stored energy
        return evolved

    # ------------------------------------------------------------------
    def wave_energy(self) -> np.ndarray:
        """Return the current wave energy ``~ amplitude^2 / 2``."""

        if self.amplitude is None:
            self.energy = 0.0
        else:
            amp = _to_array(self.amplitude)
            self.energy = 0.5 * amp * amp
        return _to_array(self.energy)

    def phase_velocity(self, k: Any | None = None) -> np.ndarray:
        """Return the phase velocity ``ω/k`` for wavenumber ``k``."""

        if k is None:
            k = self.last_k if self.last_k is not None else 0.0
        ks = _to_array(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            vel = self.frequency() / ks
        self.last_phase_velocity = vel
        return vel

    def power(self) -> np.ndarray:
        """Return a simple estimate of wave power ``energy * ω``."""

        energy = self.wave_energy()
        return energy * self.frequency()



    def anomalous_resistivity(self, J: np.ndarray):
        base = np.zeros(J.shape[:-1])
        if self.amplitude is None:
            eta = base
        else:
            try:  # pragma: no cover - real NumPy path
                eta = np.broadcast_to(self.amplitude, base.shape)
            except Exception:  # pragma: no cover - minimal stub fallback
                amp = _to_array(self.amplitude)
                eta = base + amp
        return eta, np.zeros_like(J)

__all__ = ["LowerHybridDrift"]
