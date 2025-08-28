import numpy as np
import pytest

from dpf2.plasma_model import advance_plasma_with_circuit, advance_plasmas_with_circuit
from dpf2.physics.simple_plasma import ZeroDPlasma
from dpf2.core.bases import CouplingState
from dpf2.rlc_solver import solve_distributed_circuit
from dpf2.circuit.distributed import TransmissionLineSegment


def test_parallel_plasma_solver_matches_serial():
    plasmas = [ZeroDPlasma(lambda t, I, V: (0.0, 0.0)) for _ in range(3)]
    for p in plasmas:
        p.coil_radius = 0.1
    states = [{"radius": 0.05 + 0.01 * i} for i in range(3)]
    couplings = [CouplingState(current=1.0 + i, voltage=0.0, mutual_inductance=0.0) for i in range(3)]
    dt = 1e-6

    seq = [
        advance_plasma_with_circuit(p, s, c, dt)
        for p, s, c in zip(plasmas, states, couplings)
    ]
    par = advance_plasmas_with_circuit(plasmas, states, couplings, dt, n_threads=2)

    for a, b in zip(par, seq):
        assert a.mutual_inductance == pytest.approx(b.mutual_inductance)
        assert a.back_reaction == pytest.approx(b.back_reaction)


def test_parallel_circuit_solver_matches_serial():
    segs = [
        TransmissionLineSegment(0, 1, 1.0, 1e-6, 0.1, 0.0),
        TransmissionLineSegment(1, 2, 1.0, 1e-6, 0.1, 0.0),
    ]

    sol1 = solve_distributed_circuit(segs, [], V0=1.0, t_end=1e-6, dt=1e-7, n_threads=1)
    sol2 = solve_distributed_circuit(segs, [], V0=1.0, t_end=1e-6, dt=1e-7, n_threads=2)

    assert np.allclose(sol1.current, sol2.current)
    assert np.allclose(sol1.voltage, sol2.voltage)

