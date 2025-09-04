"""Regime diagnostic panel tracking dimensionless parameters over time."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict

from .plasma import (
    lundquist_number,
    plasma_beta,
    alfven_mach_number,
    magnetic_reynolds_number,
)
from .quality_dashboard import QualityDashboard

logger = logging.getLogger(__name__)

try:  # pragma: no cover - SciPy optional
    from scipy.constants import e, m_e
except Exception:  # pragma: no cover - fallback values
    e = 1.602176634e-19
    m_e = 9.1093837015e-31


@dataclass
class RegimePanel:
    """Compute and log dimensionless plasma regime parameters."""

    L: float
    quality: QualityDashboard | None = None
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "S": 1.0,
            "beta": 1.0,
            "M_A": 1.0,
            "R_m": 1.0,
            "K_n": 0.1,
            "omega_ce_tau_e": 1.0,
        }
    )
    history: list[Dict[str, float]] = field(default_factory=list)

    def log(
        self,
        step: int,
        n: float,
        T: float,
        B: float,
        v: float,
        eta: float,
        mfp: float,
        tau_e: float,
    ) -> Dict[str, float]:
        """Compute parameters for a simulation step.

        Parameters are specified in SI units. ``n`` is number density, ``T``
        the temperature, ``B`` the magnetic-field magnitude, ``v`` a
        characteristic flow speed, ``eta`` the resistivity, ``mfp`` the mean
        free path and ``tau_e`` the electron collision time.
        """

        sigma = 1.0 / eta if eta > 0 else float("inf")
        S = lundquist_number(B, n, self.L, sigma)
        beta = plasma_beta(n, T, B)
        M_A = alfven_mach_number(v, B, n)
        R_m = magnetic_reynolds_number(v, self.L, sigma)
        K_n = mfp / self.L
        omega_ce_tau_e = (e * B / m_e) * tau_e

        entry = {
            "step": step,
            "S": float(S),
            "beta": float(beta),
            "M_A": float(M_A),
            "R_m": float(R_m),
            "K_n": float(K_n),
            "omega_ce_tau_e": float(omega_ce_tau_e),
        }
        self.history.append(entry)

        if self.quality is not None:
            self.quality.log_regime(
                step,
                entry["S"],
                entry["beta"],
                entry["M_A"],
                entry["R_m"],
                entry["K_n"],
                entry["omega_ce_tau_e"],
            )

        violations: Dict[str, bool] = {}
        for key, limit in self.thresholds.items():
            val = entry[key]
            if key in {"beta", "M_A", "K_n"}:
                violations[key] = val > limit
            else:
                violations[key] = val < limit
            if violations[key]:
                logger.warning("Regime parameter %s out of bounds: %.3g", key, val)
        entry["violations"] = violations

        return entry

