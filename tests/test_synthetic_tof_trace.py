import math
import numpy as np

from dpf2.synthetic_diagnostics import synthetic_tof_trace
from dpf2.core.bases import CouplingState


def test_tof_alignment():
    history = [CouplingState(current=c, voltage=c) for c in [0.0, 1.0, 5.0, 1.0, 0.0]]
    dt = 1e-9
    times, counts = synthetic_tof_trace(history, dt, 1.0, energies_mev=[2.45])
    idx = next(i for i, v in enumerate(counts) if v > 0)
    tof_time = times[idx]
    m_n = 1.67492749804e-27
    E = 2.45 * 1.602176634e-13
    v = math.sqrt(2.0 * E / m_n)
    expected = 2 * dt + 1.0 / v
    assert abs(tof_time - expected) / expected < 0.1
