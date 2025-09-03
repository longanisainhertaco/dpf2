"""Tests for the placeholder radiation-MHD solver and AMR support."""

import numpy as np

from dpf2.physics.radiation_mhd import RadiationMHDSolver
from dpf2.circuit.distributed import TransmissionLineSegment
from dpf2.rlc_solver import solve_distributed_circuit


def test_amr_refinement_creates_child_grid():
    solver = RadiationMHDSolver(refine_criterion=lambda arr: True)
    state = solver.allocate_state((4, 4, 4))
    assert not state.grid.children
    solver.step(state, 1e-6, 0.0, 0.0)
    assert state.grid.children and state.grid.children[0].level == 1


def test_circuit_coupling_back_reaction():
    seg = TransmissionLineSegment(0, 1, length=1.0, L_per_m=1e-6, R_per_m=0.0, C_per_m=0.0)
    solver = RadiationMHDSolver()
    dt = 1e-9
    res = solve_distributed_circuit([seg], [], V0=1.0, t_end=dt, dt=dt, em_solver=solver)
    iface = solver.coupling_interface()
    br = iface.back_reaction
    expected = dt * 1.0 / (seg.L_per_m * seg.length) + br
    assert np.isclose(res.current[-1], expected, rtol=1e-3)
    assert br != 0.0

