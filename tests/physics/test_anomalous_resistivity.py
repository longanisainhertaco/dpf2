import numpy as np
import pytest

from dpf2.hall_mhd_solver import HallMHDSolver
from dpf2.physics.lower_hybrid_drift import LowerHybridDrift
from dpf2.diagnostics.quality_dashboard import QualityDashboard


def test_impedance_spike_from_lhdi_not_fixed():
    def fixed(J):
        return np.full(J.shape[:-1], 0.05)

    lhd = LowerHybridDrift(B=1.0, n_i=1e19, amplitude=0.2)
    lhd.phase_velocity(1.0)
    solver = HallMHDSolver(anomalous_resistivity=fixed, lower_hybrid_drift=lhd)
    J = np.ones((1, 3))
    eta = solver.compute_anomalous_resistivity(J)
    assert solver.last_voltage_spike == pytest.approx(0.6)
    assert eta[0] == pytest.approx(0.25)
    assert solver.last_lh_power > 0.0
    assert solver.last_lh_phase_velocity > 0.0

    solver.impedance_growth.append(solver.last_voltage_spike / 1.0)
    assert solver.impedance_growth[-1] == pytest.approx(0.6)


def test_quality_dashboard_logs_lh_metrics(tmp_path):
    dash = QualityDashboard(output_dir=tmp_path)
    dash.log(
        step=1,
        dt=1.0,
        cell_size=1.0,
        ppc=1.0,
        cfl=0.1,
        lambda_D=0.2,
        lower_hybrid_power=2.0,
        lower_hybrid_phase_velocity=3.0,
        plasma_impedance=0.5,
    )
    entry = dash.history[-1]
    assert entry["plasma_impedance"] == pytest.approx(0.5)
    assert entry["lower_hybrid_power"] == pytest.approx(2.0)
    assert entry["lower_hybrid_phase_velocity"] == pytest.approx(3.0)


def test_fixed_resistivity_does_not_trigger_spike():
    def fixed(J):
        return np.full(J.shape[:-1], 0.05)

    solver = HallMHDSolver(anomalous_resistivity=fixed)
    J = np.ones((1, 3))
    solver.compute_anomalous_resistivity(J)
    assert solver.last_voltage_spike == 0.0

