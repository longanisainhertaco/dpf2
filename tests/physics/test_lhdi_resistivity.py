import sys
import pathlib
import csv
from pathlib import Path

sys.modules.pop("numpy", None)
sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "venv/lib/python3.12/site-packages"
    ),
)
import numpy as np

from dpf2.hall_mhd_solver import HallMHDSolver, spitzer_resistivity
from dpf2.physics.lower_hybrid_drift import LowerHybridDrift


def _peak_voltage() -> float:
    """Return maximum voltage from the LLNL MJOLNIR benchmark."""
    path = Path("data/benchmarks/LLNL_MJOLNIR/voltage.csv")
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        return max(float(row[1]) for row in reader)


def _make_solver(scale: float, floor: float) -> HallMHDSolver:
    lhd = LowerHybridDrift(B=1.0, n_i=1e19)
    solver = HallMHDSolver()
    solver.enable_spectral_resistivity(lhd, scale=scale, floor=floor)
    return solver


def test_voltage_spike_scaling_matches_experiment():
    expected = _peak_voltage()
    floor = float(spitzer_resistivity(1e20, 1e5, 1.0))
    J = np.ones((1, 3))
    power = float(np.linalg.norm(J, axis=-1)[0])
    mag = float(np.abs(J[..., 0]) + np.abs(J[..., 1]) + np.abs(J[..., 2]))
    scale = (expected / mag - floor) / power

    solver_full = _make_solver(scale, floor)
    solver_full.compute_anomalous_resistivity(J)
    spike_full = solver_full.last_voltage_spike

    solver_half = _make_solver(scale / 2, floor)
    solver_half.compute_anomalous_resistivity(J)
    spike_half = solver_half.last_voltage_spike

    assert np.isclose(spike_full, expected)
    assert np.isclose(spike_half - floor * mag, 0.5 * (spike_full - floor * mag))


def test_spitzer_floor_enforced():
    floor = float(spitzer_resistivity(1e20, 1e5, 1.0))
    J = np.ones((1, 3))
    mag = float(np.abs(J[..., 0]) + np.abs(J[..., 1]) + np.abs(J[..., 2]))
    solver = _make_solver(0.0, floor)
    eta = solver.compute_anomalous_resistivity(J)
    assert np.isclose(float(eta[0]), floor)
    assert np.isclose(solver.last_voltage_spike, floor * mag)
