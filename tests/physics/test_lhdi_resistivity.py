import numpy as np
import pytest

from dpf2.hall_mhd_solver import HallMHDSolver
from dpf2.physics.lower_hybrid_drift import LowerHybridDrift
from dpf2.physics.lhdi_resistivity import LHDIResistivity


def test_lhdi_resistivity_voltage_spike_scaling():
    lhd = LowerHybridDrift(B=1.0, n_i=1e19, amplitude=0.1)
    model = LHDIResistivity(lhd, scale=0.5)
    solver = HallMHDSolver(lower_hybrid_drift=model)
    J = np.ones((1, 3))
    eta = solver.compute_anomalous_resistivity(J)
    expected = 0.5 * lhd.power() / (abs(lhd.phase_velocity(1.0)) + 1e-12)
    assert eta[0] == pytest.approx(expected)
    assert solver.last_voltage_spike == pytest.approx(expected * 3)
    assert solver.last_Ez_surge == pytest.approx(expected)


def test_lhdi_resistivity_below_spitzer_fails():
    lhd = LowerHybridDrift(B=1.0, n_i=1e19, amplitude=0.0)
    model = LHDIResistivity(lhd)
    solver = HallMHDSolver(lower_hybrid_drift=model)
    J = np.ones((1, 3))
    with pytest.raises(RuntimeError):
        solver.compute_anomalous_resistivity(J)
