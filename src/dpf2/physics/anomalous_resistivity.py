from __future__ import annotations

"""Lower-hybrid drift spectral anomalous resistivity model.

This module provides a tiny utility class that estimates an effective
resistivity from the azimuthal spectrum of the current density.  The
intent is to mimic the enhanced resistivity that can arise from
lower-hybrid drift waves.  The implementation is intentionally simple
and primarily serves as a placeholder for more sophisticated models.

Example
-------
>>> from dpf2.physics.lower_hybrid_drift import LowerHybridDrift
>>> from dpf2.physics.anomalous_resistivity import SpectralResistivity
>>> lhd = LowerHybridDrift(B=1.0, n_i=1e18)
>>> model = SpectralResistivity(lhd, scale=1e-3)
>>> J = np.zeros((4, 4, 3))
>>> eta, E = model(J)
>>> eta.shape
(4, 4)
"""

from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np

from ..diagnostics.modes import azimuthal_mode_spectrum
from .lower_hybrid_drift import LowerHybridDrift


@dataclass
class SpectralResistivity:
    """Estimate anomalous resistivity from lower-hybrid drift spectra.

    Parameters
    ----------
    lhd:
        Instance supplying the lower-hybrid frequency through
        :meth:`~dpf2.physics.lower_hybrid_drift.LowerHybridDrift.frequency`.
    scale:
        Multiplicative factor applied to the spectral power.
    floor:
        Minimum resistivity returned by the model.
    """

    lhd: LowerHybridDrift
    scale: float = 1.0
    floor: float = 0.0

    def __call__(self, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return an anomalous resistivity estimate for ``J``.

        The magnitude of ``J`` is decomposed into azimuthal modes and the
        mode closest to the lower-hybrid frequency is used as a proxy for the
        turbulent wave power.  The resistivity is ``scale`` times this power
        with ``floor`` enforcing a minimum value.  A zero electric-field
        contribution is returned so the object is compatible with
        :meth:`HallMHDSolver.compute_anomalous_resistivity`.
        """

        try:
            magJ = np.linalg.norm(J, axis=-1)
            spectrum = azimuthal_mode_spectrum(magJ, axis=-1)
            if len(spectrum) == 0:
                power = 0.0
            else:
                m = int(round(self.lhd.frequency())) % len(spectrum)
                power = float(spectrum[m])
        except Exception:  # pragma: no cover - very small ``numpy`` stub
            power = 0.0

        eta = self.scale * power + self.floor
        try:  # pragma: no cover - real ``numpy`` path
            eta_field = np.broadcast_to(eta, J.shape[:-1])
        except Exception:  # pragma: no cover - minimal stub fallback
            eta_field = np.ones(J.shape[:-1]) * eta
        return eta_field, np.zeros_like(J)


__all__ = ["SpectralResistivity"]
