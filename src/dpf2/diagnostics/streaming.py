"""Real-time streaming diagnostics for Dense Plasma Focus simulations.

These diagnostics are intentionally lightweight and operate on the
``CouplingState`` exchanged between circuit and plasma solvers.  Each
diagnostic accepts a callback which is invoked with ``(time, value)`` for
streaming to user interfaces or loggers.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..core.bases import DiagnosticsBase, CouplingState


class NeutronYieldStreamer(DiagnosticsBase):
    """Stream approximate neutron production rate.

    The model here is intentionally simple and scales with the square of
    the circuit current.  It is sufficient for testing hooks and
    real-time visualisation but not for accurate physics predictions.
    """

    def __init__(self, callback: Callable[[float, float], None]) -> None:
        self.callback = callback
        self._total = 0.0

    def record(self, state: CouplingState, time: float) -> None:
        rate = 1.0e5 * state.current ** 2
        self._total += rate
        self.callback(time, rate)

    @property
    def total_yield(self) -> float:
        """Return accumulated neutron yield estimate."""

        return self._total


class XRayEmissionStreamer(DiagnosticsBase):
    """Stream simplified X-ray emission power.

    The emission is approximated by ``|I * V|`` scaled to a convenient
    magnitude.  Users may replace this with a more sophisticated model if
    desired.
    """

    def __init__(self, callback: Callable[[float, float], None]) -> None:
        self.callback = callback

    def record(self, state: CouplingState, time: float) -> None:
        power = abs(state.current * state.voltage) * 1.0e-3
        self.callback(time, power)


__all__ = ["NeutronYieldStreamer", "XRayEmissionStreamer"]

