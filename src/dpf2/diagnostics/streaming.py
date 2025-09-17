"""Real-time streaming diagnostics for Dense Plasma Focus simulations.

These diagnostics are intentionally lightweight and operate on the
``CouplingState`` exchanged between circuit and plasma solvers.  Each
diagnostic accepts a callback which is invoked with ``(time, value)`` for
streaming to user interfaces or loggers.
"""

from __future__ import annotations

from typing import Callable, Optional, Dict

from ..core.bases import DiagnosticsBase, CouplingState


class NeutronYieldStreamer(DiagnosticsBase):
    """Stream approximate neutron production rate.

    The model here is intentionally simple and scales with the square of
    the circuit current.  It is sufficient for testing hooks and
    real-time visualisation but not for accurate physics predictions.
    """

    def __init__(
        self,
        callback: Callable[[float, float], None],
        comparator: Optional["RealTimeComparator"] = None,
    ) -> None:
        self.callback = callback
        self.comparator = comparator
        self._total = 0.0

    def record(self, state: CouplingState, time: float) -> None:
        rate = 1.0e5 * state.current**2
        self._total += rate
        self.callback(time, rate)
        if self.comparator is not None:
            self.comparator.compare(time, rate)

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

    def __init__(
        self,
        callback: Callable[[float, float], None],
        comparator: Optional["RealTimeComparator"] = None,
    ) -> None:
        self.callback = callback
        self.comparator = comparator
        self._total = 0.0

    def record(self, state: CouplingState, time: float) -> None:
        power = abs(state.current * state.voltage) * 1.0e-3
        self._total += power
        self.callback(time, power)
        if self.comparator is not None:
            self.comparator.compare(time, power)

    @property
    def total_power(self) -> float:
        """Return accumulated X-ray emission estimate."""

        return self._total


class RealTimeComparator:
    """Hold experimental measurements and compare to simulation streams.

    Experimental values are ingested via :meth:`ingest` and are paired with
    simulated values using the timestamp provided to :meth:`compare`.  When
    both values for a given time are available the supplied ``callback`` is
    invoked with ``(time, sim_value, exp_value)``.
    """

    def __init__(self, callback: Callable[[float, float, float], None]) -> None:
        self.callback = callback
        self._experimental: Dict[float, float] = {}

    def ingest(self, time: float, value: float) -> None:
        """Provide an experimental measurement for later comparison."""

        self._experimental[time] = value

    def compare(self, time: float, sim_value: float) -> None:
        """Compare ``sim_value`` against any stored experimental datum."""

        if time in self._experimental:
            exp_value = self._experimental.pop(time)
            self.callback(time, sim_value, exp_value)


__all__ = ["NeutronYieldStreamer", "XRayEmissionStreamer", "RealTimeComparator"]
