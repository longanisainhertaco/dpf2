"""Benchmarks for telegrapher equation phase delay in branched networks."""

import cmath

import numpy as np

from dpf2.circuit.distributed import TransmissionLineSegment
from dpf2.rlc_solver import solve_distributed_circuit


def _measure_amp_phase(vals, ref, t, freq):
    amp = (max(vals) - min(vals)) / 2.0
    period = int((1.0 / freq) / (t[1] - t[0]))
    vals_period = vals[:period]
    ref_period = ref[:period]
    idx_val = max(range(len(vals_period)), key=lambda i: vals_period[i])
    idx_ref = max(range(len(ref_period)), key=lambda i: ref_period[i])
    delay = t[idx_val] - t[idx_ref]
    phase = -2.0 * np.pi * freq * delay
    return amp, phase


def test_branch_phase_delay_matches_nodal_solution():
    seg1 = TransmissionLineSegment(0, 1, length=1.0, L_per_m=2.5e-7, R_per_m=0.0, C_per_m=1e-10)
    seg2 = TransmissionLineSegment(1, 2, length=1.0, L_per_m=2.5e-7, R_per_m=0.0, C_per_m=1e-10)
    seg3 = TransmissionLineSegment(1, 2, length=1.0, L_per_m=2.5e-7, R_per_m=0.0, C_per_m=1e-10)
    freq = 1e7
    res = solve_distributed_circuit([seg1, seg2, seg3], [], V0=1.0, t_end=2e-6, dt=1e-9, frequency=freq)

    # Analytic nodal solution
    def seg_Y(seg):
        gamma = seg.propagation_constant(freq) * seg.length
        Z0 = seg.characteristic_impedance(freq)
        sinh_gl = cmath.sinh(gamma)
        cosh_gl = cmath.cosh(gamma)
        Y_self = (1.0 / Z0) * (cosh_gl / sinh_gl)
        Y_off = -(1.0 / Z0) * (1.0 / sinh_gl)
        return Y_self, Y_off

    Y = np.zeros((3, 3)) + 0j
    for seg in [seg1, seg2, seg3]:
        Y_self, Y_off = seg_Y(seg)
        i, j = seg.from_node, seg.to_node
        Y[i, i] += Y_self
        Y[j, j] += Y_self
        Y[i, j] += Y_off
        Y[j, i] += Y_off

    Y11 = Y[1, 1]
    rhs = -Y[1, 0] * 1.0
    V1 = rhs / Y11
    amp_exp = abs(V1)
    phase_exp = cmath.phase(V1)

    node_vals = [row[1] for row in res.node_voltages]
    src_vals = [row[0] for row in res.node_voltages]
    amp_sol, phase_sol = _measure_amp_phase(node_vals, src_vals, res.t, freq)
    assert np.isclose(amp_sol, amp_exp, rtol=1e-2, atol=1e-6)
    assert np.isclose(phase_sol, phase_exp, rtol=1e-2, atol=1e-6)

