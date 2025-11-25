from __future__ import annotations

"""Simple flashover/neutral replenishment model."""

from dataclasses import dataclass

from dpf2.paschen import paschen_breakdown_time


@dataclass
class NeutralGasPuff:
    """Parameterized neutral gas puff with simple coupling helper."""

    puff_time: float
    rise_time: float
    base_density: float
    coupling_efficiency: float = 0.5

    def density(self, t: float) -> float:
        """Return neutral density at time ``t`` using a smooth ramp."""

        if t < self.puff_time:
            return 0.0
        dt = max(t - self.puff_time, 0.0)
        if self.rise_time <= 0:
            return self.base_density
        ramp = min(dt / self.rise_time, 1.0)
        return self.base_density * ramp

    def couple_to_plasma(self, plasma_density: float, t: float) -> float:
        """Return effective plasma density after neutral coupling."""

        neutral = self.density(t)
        return plasma_density + self.coupling_efficiency * neutral


@dataclass
class FlashoverEvent:
    """Capture the timing of a surface flashover event."""

    breakdown_time: float
    recovered_density: float
    triggered: bool = False


@dataclass
class FlashoverModel:
    """Track neutral pressure recovery and flashover timing.

    The model is intentionally compact: it evaluates a Paschen-like delay,
    boosts the neutral density when flashover occurs, and exposes the resulting
    :class:`FlashoverEvent` for coupling into circuit/chemistry solvers.
    """

    gap: float
    pressure: float
    voltage: float
    recovery_factor: float = 1.5

    def evaluate(self) -> FlashoverEvent:
        t_break = paschen_breakdown_time(self.gap, self.pressure, self.voltage)
        recovered_density = self.pressure * self.recovery_factor
        return FlashoverEvent(breakdown_time=t_break, recovered_density=recovered_density)


__all__ = ["FlashoverEvent", "FlashoverModel", "NeutralGasPuff"]
