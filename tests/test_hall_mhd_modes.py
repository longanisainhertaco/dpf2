import numpy as np

from dpf2.hall_mhd_solver import HallMHDSolver


def test_anomalous_resistivity_gating():
    solver = HallMHDSolver()
    J = np.ones((2, 2, 3))

    solver.anomalous_resistivity = lambda arr: (np.ones(arr.shape[:-1]), np.zeros_like(arr))
    solver.gate_anomalous_resistivity("anomalous_resistivity", False)
    eta_disabled = solver.compute_anomalous_resistivity(J)
    eta_disabled_data = getattr(eta_disabled, "data", eta_disabled)
    eta_disabled_data = getattr(eta_disabled_data, "data", eta_disabled_data)
    assert all(val == 0.0 for row in eta_disabled_data for val in row)

    solver.gate_anomalous_resistivity("anomalous_resistivity", True)
    eta_enabled = solver.compute_anomalous_resistivity(J)
    eta_enabled_data = getattr(eta_enabled, "data", eta_enabled)
    eta_enabled_data = getattr(eta_enabled_data, "data", eta_enabled_data)
    assert all(val == 1.0 for row in eta_enabled_data for val in row)


def test_mode_monitoring_growth_rate():
    solver = HallMHDSolver()
    base = np.ones((2, 8))
    solver._record_mode_metrics(base, dt=1.0)
    assert solver.last_mode_spectrum is not None

    amplified = 2.0 * base
    solver._record_mode_metrics(amplified, dt=0.5)
    assert solver.last_mode_growth is not None
    assert solver.mode_growth_history

