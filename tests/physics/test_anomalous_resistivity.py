from pathlib import Path
import numpy as np
import pytest
from dpf2.hall_mhd_solver import HallMHDSolver
from dpf2.validation_suite import load_pinch_dataset


@pytest.mark.parametrize("device", ["PF1000", "LLNL_MJOLNIR"])
def test_voltage_spike_matches_dataset(device):
    if not hasattr(np, "loadtxt"):
        import csv

        def _loadtxt(path, delimiter=",", skiprows=1):
            with open(path) as f:
                reader = csv.reader(f)
                rows = [row for row in reader][skiprows:]
                return np.array([[float(r[0]), float(r[1])] for r in rows])

        np.loadtxt = _loadtxt  # type: ignore[attr-defined]

    bench = load_pinch_dataset(Path(f"data/benchmarks/{device}"))
    _, voltage = bench["voltage"]
    expected = float(np.max(voltage))

    def model(J):
        return np.full(J.shape[:-1], expected / 3)

    solver = HallMHDSolver(anomalous_resistivity=model)
    J = np.ones((1, 3))
    eta = solver.compute_anomalous_resistivity(J)
    assert np.isclose(eta[0], expected / 3)
    assert np.isclose(solver.voltage_spikes[-1], expected)
    assert np.isclose(expected, np.max(voltage))
