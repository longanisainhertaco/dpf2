"""Lightweight synthetic diagnostic signal generators.

Each helper operates on an iterable of :class:`~dpf2.core.bases.CouplingState`
objects representing the circuit/plasma coupling at successive time steps.
The functions are intentionally simple and serve as stand-ins for more
specialised diagnostics that would exist in a full application.
"""

from __future__ import annotations

from typing import Callable, Iterable, List
import math

from ..core.bases import CouplingState


def _apply(
    values: List[float],
    response_fn: Callable[[float], float] | None,
    noise_fn: Callable[[float], float] | None,
) -> List[float]:
    if response_fn:
        values = [response_fn(v) for v in values]
    if noise_fn:
        values = [v + noise_fn(v) for v in values]
    return values


def current_waveform(
    history: Iterable[CouplingState],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Return the circuit current for each time step."""

    data = [float(state.current) for state in history]
    return _apply(data, response_fn, noise_fn)


def voltage_waveform(
    history: Iterable[CouplingState],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Return the capacitor voltage for each time step."""

    data = [float(state.voltage) for state in history]
    return _apply(data, response_fn, noise_fn)


def coupled_current_waveform(
    history: Iterable[CouplingState],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Return current including simple back-reaction term.

    The ``CouplingState.back_reaction`` value is interpreted as a voltage
    contribution which is scaled by unity for this synthetic diagnostic.
    """

    data = [float(state.current + state.back_reaction) for state in history]
    return _apply(data, response_fn, noise_fn)


def coupled_voltage_waveform(
    history: Iterable[CouplingState],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Return voltage including mutual inductance contribution."""

    data = [float(state.voltage + state.mutual_inductance) for state in history]
    return _apply(data, response_fn, noise_fn)


def rogowski_signal(
    history: Iterable[CouplingState],
    dt: float,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Compute a synthetic Rogowski coil signal ``dI/dt``."""

    currents = current_waveform(history)
    if len(currents) < 2 or dt <= 0:
        deriv = [0.0 for _ in currents]
    else:
        deriv = [(currents[i + 1] - currents[i]) / dt for i in range(len(currents) - 1)]
        deriv.append(deriv[-1])
    return _apply(deriv, response_fn, noise_fn)


def bdot_signal(
    history: Iterable[CouplingState],
    radius: float,
    dt: float,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Generate a simple B-dot probe signal assuming axial field geometry."""

    mu0 = 4.0 * math.pi * 1e-7
    if radius <= 0:
        raise ValueError("radius must be positive")
    currents = current_waveform(history)
    B = [mu0 * I / (2.0 * math.pi * radius) for I in currents]
    if len(B) < 2 or dt <= 0:
        deriv = [0.0 for _ in B]
    else:
        deriv = [(B[i + 1] - B[i]) / dt for i in range(len(B) - 1)]
        deriv.append(deriv[-1])
    return _apply(deriv, response_fn, noise_fn)


__all__ = [
    "current_waveform",
    "voltage_waveform",
    "coupled_current_waveform",
    "coupled_voltage_waveform",
    "rogowski_signal",
    "bdot_signal",
]

