"""Tests for radiation-MHD solver coupling and conservation."""

import cmath
import math
import numpy as np

from dpf2.physics.radiation_mhd_solver import RadiationMHDSolver
from dpf2.circuit.distributed import TransmissionLineSegment
from dpf2.rlc_solver import solve_distributed_circuit


def _measure_amp_phase(vals, ref, t, freq):
    """Return amplitude and phase of ``vals`` relative to ``ref``."""
    amp = (max(vals) - min(vals)) / 2.0
    period = int((1.0 / freq) / (t[1] - t[0]))
    vals_period = vals[:period]
    ref_period = ref[:period]
    idx_val = max(range(len(vals_period)), key=lambda i: vals_period[i])
    idx_ref = max(range(len(ref_period)), key=lambda i: ref_period[i])
    delay = t[idx_val] - t[idx_ref]
    phase = -2.0 * math.pi * freq * delay
    return amp, phase


def test_wave_propagation():
    seg1 = TransmissionLineSegment(0, 1, length=1.0, L_per_m=2.5e-7, R_per_m=0.0, C_per_m=1e-10)
    seg2 = TransmissionLineSegment(1, 2, length=1.0, L_per_m=2.5e-7, R_per_m=0.0, C_per_m=1e-10)
    solver = RadiationMHDSolver()
    freq = 1e7
    dt = 1e-9
    res = solve_distributed_circuit([seg1, seg2], [], V0=1.0, t_end=2e-6, dt=dt, frequency=freq, em_solver=solver)

    node_vals = [row[1] for row in res.node_voltages]
    src_vals = [row[0] for row in res.node_voltages]
    amp_sol, phase_sol = _measure_amp_phase(node_vals, src_vals, res.t, freq)

    def seg_Y(seg):
        gamma = seg.propagation_constant(freq) * seg.length
        Z0 = seg.characteristic_impedance(freq)
        sinh_gl = cmath.sinh(gamma)
        cosh_gl = cmath.cosh(gamma)
        Y_self = (1.0 / Z0) * (cosh_gl / sinh_gl)
        Y_off = -(1.0 / Z0) * (1.0 / sinh_gl)
        return Y_self, Y_off

    Y = [[0j for _ in range(3)] for _ in range(3)]
    for seg in (seg1, seg2):
        Y_self, Y_off = seg_Y(seg)
        i, j = seg.from_node, seg.to_node
        Y[i][i] += Y_self
        Y[j][j] += Y_self
        Y[i][j] += Y_off
        Y[j][i] += Y_off
    V1 = (-Y[1][0] * 1.0) / Y[1][1]
    amp_exp = abs(V1)
    phase_exp = cmath.phase(V1)

    assert np.isclose(amp_sol, amp_exp, rtol=1e-2, atol=1e-6)
    assert np.isclose(phase_sol, phase_exp, rtol=1e-2, atol=1e-6)


def test_energy_conservation():
    solver = RadiationMHDSolver(two_temperature=True, use_gpu=False)
    state = solver.allocate_state((4, 1, 1))
    nx, ny, nz = state.density.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                state.density[i][j][k] = 1.0
                state.energy[i][j][k] = 2.0
                state.velocity[i][j][k][0] = 1.0
                state.magnetic[i][j][k][0] = 0.1
                if state.electron_temp is not None:
                    state.electron_temp[i][j][k] = 3.0

    def total_energy(st):
        nx, ny, nz = st.density.shape
        total = 0.0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    rho = st.density[i][j][k]
                    vx = st.velocity[i][j][k][0]
                    vy = st.velocity[i][j][k][1]
                    vz = st.velocity[i][j][k][2]
                    Bx = st.magnetic[i][j][k][0]
                    By = st.magnetic[i][j][k][1]
                    Bz = st.magnetic[i][j][k][2]
                    e = st.energy[i][j][k]
                    rad = st.electron_temp[i][j][k] if st.electron_temp is not None else 0.0
                    total += 0.5 * rho * (vx**2 + vy**2 + vz**2)
                    total += 0.5 * (Bx**2 + By**2 + Bz**2) + e + rad
        return total

    E0 = total_energy(state)
    for _ in range(5):
        solver.step(state, 1e-6, 0.0, 0.0)
    E1 = total_energy(state)
    assert np.isclose(E1, E0, rtol=1e-12)
