from __future__ import annotations

"""Lower-hybrid drift instability effective resistivity model.

The implementation is intentionally compact and derives an effective
resistivity from the wave power and phase velocity provided by a
:class:`~dpf2.physics.lower_hybrid_drift.LowerHybridDrift` instance.
The resulting model is compatible with
:meth:`dpf2.hall_mhd_solver.HallMHDSolver.compute_anomalous_resistivity`
returning both a spatial resistivity field and an optional axial electric
field contribution.
"""

from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np

from .lower_hybrid_drift import LowerHybridDrift


@dataclass
class LHDIResistivity:
    """Map lower‑hybrid drift wave power to an anomalous resistivity.

    Parameters
    ----------
    lhd:
        Instability model supplying ``power`` and ``phase_velocity``.
    scale:
        Multiplicative factor applied to the power/velocity ratio.
    floor:
        Minimum resistivity returned by the model.
    """

    lhd: LowerHybridDrift
    scale: float = 1.0
    floor: float = 0.0

    def __call__(self, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return effective resistivity and axial electric field for ``J``.

        The resistivity is computed as ``scale * power / |v_phase|`` with
        ``floor`` enforcing a lower bound.  The same quantity is used as an
        axial electric‑field surge to mimic the impulsive response expected
        from unresolved lower‑hybrid drift waves.
        """

        try:
            power = self.lhd.power()
            v_phase = np.abs(self.lhd.phase_velocity(1.0)) + 1e-12
            eta = self.scale * power / v_phase + self.floor
        except Exception:  # pragma: no cover - numpy stub fallback
            eta = self.floor

        try:  # pragma: no cover - broadcast for real NumPy
            eta_field = np.broadcast_to(eta, J.shape[:-1])
            Ez = np.broadcast_to(eta, J.shape[:-1])
        except Exception:  # pragma: no cover - minimal stub fallback
            eta_field = np.ones(J.shape[:-1]) * eta
            Ez = np.ones(J.shape[:-1]) * eta
        return eta_field, Ez


__all__ = ["LHDIResistivity"]
