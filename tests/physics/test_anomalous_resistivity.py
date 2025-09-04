import numpy as np
import pytest

from dpf2.hall_mhd_solver import HallMHDSolver
from dpf2.physics.lower_hybrid_drift import LowerHybridDrift


def test_impedance_spike_from_lhdi_not_fixed():
    def fixed(J):
        return np.full(J.shape[:-1], 0.05)

    lhd = LowerHybridDrift(B=1.0, n_i=1e19, amplitude=0.2)
    solver = HallMHDSolver(
        anomalous_resistivity=fixed, lower_hybrid_drift=lhd.anomalous_resistivity
    )
    J = np.ones((1, 3))
    eta = solver.compute_anomalous_resistivity(J)
    assert solver.last_voltage_spike == pytest.approx(0.6)
    assert eta[0] == pytest.approx(0.25)

    solver.impedance_growth.append(solver.last_voltage_spike / 1.0)
    assert solver.impedance_growth[-1] == pytest.approx(0.6)


def test_fixed_resistivity_does_not_trigger_spike():
    def fixed(J):
        return np.full(J.shape[:-1], 0.05)

    solver = HallMHDSolver(anomalous_resistivity=fixed)
    J = np.ones((1, 3))
    solver.compute_anomalous_resistivity(J)
    assert solver.last_voltage_spike == 0.0

