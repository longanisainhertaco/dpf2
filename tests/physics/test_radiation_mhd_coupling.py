
import numpy as np

from dpf2.physics.radiation_mhd import RadiationMHDSolver
from dpf2.circuit.distributed import TransmissionLineSegment
from dpf2.rlc_solver import solve_distributed_circuit


def test_frequency_domain_coupling_zero_back_reaction():
    seg = TransmissionLineSegment(
        from_node=0,
        to_node=1,
        length=1.0,
        L_per_m=1e-6,
        R_per_m=0.0,
        C_per_m=1e-10,
    )
    solver = RadiationMHDSolver()
    freq = 1e7
    dt = 1e-9
    t_end = 1e-8
    res = solve_distributed_circuit(
        [seg],
        [],
        V0=1.0,
        t_end=t_end,
        dt=dt,
        frequency=freq,
        em_solver=solver,
    )
    ref = solve_distributed_circuit([seg], [], V0=1.0, t_end=t_end, dt=dt, frequency=freq)
    assert np.allclose(res.current, ref.current)
    assert np.allclose(res.voltage, ref.voltage)
    assert solver.coupling_interface().back_reaction == 0.0

