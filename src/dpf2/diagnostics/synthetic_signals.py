"""Lightweight synthetic diagnostic signal generators.

Each helper operates on an iterable of :class:`~dpf2.core.bases.CouplingState`
objects representing the circuit/plasma coupling at successive time steps.
The functions are intentionally simple and serve as stand-ins for more
specialised diagnostics that would exist in a full application.
"""

from __future__ import annotations

from typing import Iterable, List
import math

from ..core.bases import CouplingState


def current_waveform(history: Iterable[CouplingState]) -> List[float]:
    """Return the circuit current for each time step."""

    return [float(state.current) for state in history]


def voltage_waveform(history: Iterable[CouplingState]) -> List[float]:
    """Return the capacitor voltage for each time step."""

    return [float(state.voltage) for state in history]


def rogowski_signal(history: Iterable[CouplingState], dt: float) -> List[float]:
    """Compute a synthetic Rogowski coil signal ``dI/dt``."""

    currents = current_waveform(history)
    if len(currents) < 2 or dt <= 0:
        return [0.0 for _ in currents]
    deriv = [(currents[i + 1] - currents[i]) / dt for i in range(len(currents) - 1)]
    deriv.append(deriv[-1])
    return deriv


def bdot_signal(history: Iterable[CouplingState], radius: float, dt: float) -> List[float]:
    """Generate a simple B-dot probe signal assuming axial field geometry."""

    mu0 = 4.0 * math.pi * 1e-7
    if radius <= 0:
        raise ValueError("radius must be positive")
    currents = current_waveform(history)
    B = [mu0 * I / (2.0 * math.pi * radius) for I in currents]
    if len(B) < 2 or dt <= 0:
        return [0.0 for _ in B]
    deriv = [(B[i + 1] - B[i]) / dt for i in range(len(B) - 1)]
    deriv.append(deriv[-1])
    return deriv


__all__ = [
    "current_waveform",
    "voltage_waveform",
    "rogowski_signal",
    "bdot_signal",
]

