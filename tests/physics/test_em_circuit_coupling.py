import math

import numpy as np

from dpf2.physics.em_wave import FDTDSolver, C0
from dpf2.rlc_solver import solve_distributed_circuit
from dpf2.circuit.distributed import TransmissionLineSegment


def test_fdtd_plane_wave_propagation():
    length = 1.0
    n = 100
    dx = length / n
    solver = FDTDSolver([dx] * n)
    freq = 50e6
    w = 2.0 * math.pi * freq
    dt = dx / (2 * C0)
    t = 0.0
    steps = 200
    for _ in range(steps):
        voltage = math.sin(w * t)
        solver.step(None, dt, 0.0, voltage)
        t += dt
    x = 0.2
    expected = math.sin(w * (t - x / C0)) / dx
    ratio = solver.field_at(x) / expected if expected else 0.0
    assert abs(ratio - 1.0) < 0.75


def test_circuit_em_coupling_back_reaction():
    seg = TransmissionLineSegment(
        from_node=0,
        to_node=1,
        length=1.0,
        L_per_m=1e12,
        R_per_m=0.0,
        C_per_m=0.0,
    )
    em = FDTDSolver([1.0])
    dt = 1e-9
    res = solve_distributed_circuit([seg], [], V0=1.0, t_end=dt, dt=dt, em_solver=em)
    feedback = em.coupling_interface()
    assert np.isclose(res.current[-1], feedback.back_reaction)
