"""Simple solver for a distributed series RLC circuit.

The real project contains a comprehensive circuit model.  For the unit tests in
this kata we implement only the behaviour that is required to demonstrate the
interaction between :class:`~dpf2.circuit.distributed.TransmissionLineSegment`
and :class:`~dpf2.circuit.distributed.TriggeredSwitch` objects.  The solver
supports time varying parameters and switch triggering albeit in a very small
subset of the features of the full application.

The module still exposes ``run_circuit_simulation`` from the original solver for
backwards compatibility.  The new :func:`solve_distributed_circuit` function is
used by the tests introduced in this exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ``numpy`` may be replaced by a light‑weight stub in the test environment but
# it provides the minimal functionality used below (``array``).
import numpy as np

from .circuit.distributed import TransmissionLineSegment, TriggeredSwitch, assemble_matrices

# Re-export for legacy tests
# ``circuit_solver`` is a heavy dependency in the full project.  The tests in
# this kata only require that ``run_circuit_simulation`` exists so we try to
# import it lazily and fall back to a stub when unavailable.
try:  # pragma: no cover - imported in real application
    from dpf2.circuit_solver import run_circuit_simulation  # type: ignore
except Exception:  # pragma: no cover - simplified test environment
    def run_circuit_simulation(*args, **kwargs):  # type: ignore
        raise RuntimeError("circuit_solver not available in minimal test environment")

__all__ = ["run_circuit_simulation", "solve_distributed_circuit", "DistributedRLCSolution"]


@dataclass
class DistributedRLCSolution:
    """Container returned by :func:`solve_distributed_circuit`."""

    t: np.ndarray
    current: np.ndarray
    voltage: np.ndarray


# ---------------------------------------------------------------------------
# Solver


def solve_distributed_circuit(
    segments: Sequence[TransmissionLineSegment],
    switches: Sequence[TriggeredSwitch] | None,
    V0: float,
    t_end: float,
    dt: float,
    I0: float = 0.0,
) -> DistributedRLCSolution:
    """Integrate a simple series RLC circuit composed of ``segments``.

    Parameters
    ----------
    segments, switches:
        Objects describing the circuit.  Only a very small subset of the real
        functionality is implemented; all components are assumed to be in
        series.
    V0:
        Initial capacitor voltage in Volts.
    t_end, dt:
        Simulation end time and step size in seconds.
    I0:
        Initial current (defaults to zero).
    """

    n_steps = int((t_end / dt)) + 1
    t = [i * dt for i in range(n_steps)]
    I = [0.0] * n_steps
    V = [0.0] * n_steps
    I[0] = I0
    V[0] = V0

    switches = list(switches or [])

    for k in range(1, n_steps):
        tk = t[k - 1]

        # Compute total R, L and C at the current time
        R_tot = L_tot = C_tot = 0.0
        for seg in segments:
            L, R, C = seg.totals(tk)
            L_tot += L
            R_tot += R
            C_tot += C
        for sw in switches:
            R_tot += sw.resistance(tk)
            L_tot += sw.L_parasitic
            C_tot += sw.C_parasitic

        # Explicit Euler integration of the series RLC equations
        dIdt = (V0 - V[k - 1] - R_tot * I[k - 1]) / L_tot
        dVdt = -I[k - 1] / C_tot
        I[k] = I[k - 1] + dIdt * dt
        V[k] = V[k - 1] + dVdt * dt

        for sw in switches:
            sw.update(tk + dt)

    return DistributedRLCSolution(t=np.array(t), current=np.array(I), voltage=np.array(V))
