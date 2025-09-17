from __future__ import annotations

"""Simple neutral gas fluid model with gas puff injection.

This module provides a small 0D fluid model that evolves the neutral
mass density.  It is intentionally lightweight and aims to exercise the
coupling interfaces used throughout the code base rather than providing
an exhaustive physical model.

The model evolves the neutral density according to

.. math::

    \frac{d\rho_n}{dt} = S_{puff}(t) - \nu_i\rho_n

where ``S_puff`` represents a mass source due to a gas puff and
``nu_i`` is an effective ionisation rate.
"""

from dataclasses import dataclass


@dataclass
class NeutralGasFluid:
    """Zero‑dimensional neutral gas model.

    Parameters
    ----------
    rho : float
        Initial mass density ``[kg/m^3]``.
    volume : float
        Volume of the system ``[m^3]`` used to convert mass flow to
        density injection.
    mass_flow_rate : float, optional
        Mass flow through the puff nozzle ``[kg/s]``.
    puff_start : float, optional
        Start time of the gas puff ``[s]``.
    puff_end : float, optional
        End time of the gas puff ``[s]``.
    """

    rho: float
    volume: float
    mass_flow_rate: float = 0.0
    puff_start: float = 0.0
    puff_end: float = 0.0

    def puff_source(self, t: float) -> float:
        """Return the density source due to the gas puff."""
        if self.puff_start <= t <= self.puff_end and self.mass_flow_rate > 0:
            return self.mass_flow_rate / self.volume
        return 0.0

    def source(self, t: float, ionization_rate: float) -> float:
        """Combined source term from puff and ionisation."""
        return self.puff_source(t) - ionization_rate * self.rho

    def step(self, dt: float, t: float, ionization_rate: float) -> float:
        """Advance the density by ``dt`` seconds."""
        self.rho += dt * self.source(t, ionization_rate)
        return self.rho


__all__ = ["NeutralGasFluid"]
