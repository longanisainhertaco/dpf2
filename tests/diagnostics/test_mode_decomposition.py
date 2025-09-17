import math
import numpy as np

from dpf2.diagnostics.modes import (
    azimuthal_mode_spectrum,
    azimuthal_decomposition,
    growth_rate,
)
from dpf2.synthetic_diagnostics.modes import write_growth_rates
from dpf2.hall_mhd_solver import HallMHDSolver


def test_mode_spectrum_simple_cosine():
    theta = np.linspace(0.0, 2 * np.pi, 17)[:-1]
    r = np.linspace(0.0, 1.0, 4)
    field = np.zeros((len(r), len(theta)))
    # ``numpy_stub`` lacks ``cos`` – emulate via phase-shifted sine
    cos = lambda x: np.sin(x + np.pi / 2)
    for i in range(len(r)):
        field[i] = 0.5 * cos(theta) + 0.25 * cos(2 * theta)
        field[i] = field[i] + 1.0
    spec = azimuthal_mode_spectrum(field, axis=1)
    assert spec.shape[0] >= 3
    assert np.isclose(spec[0], 1.0, atol=1e-12)
    assert np.isclose(spec[1], 0.5, atol=1e-12)
    assert np.isclose(spec[2], 0.25, atol=1e-12)


def test_growth_rate():
    prev = np.array([1.0, 1.0, 1.0])
    curr = np.array([1.0, math.e, 0.5])
    dt = 1.0
    rates = growth_rate(prev, curr, dt)
    assert np.allclose(rates, [0.0, 1.0, math.log(0.5)])


def test_mode_decomposition_complex():
    theta = np.linspace(0.0, 2 * np.pi, 33)[:-1]
    cos = lambda x: np.sin(x + np.pi / 2)
    field = cos(theta) + 0.5 * cos(2 * theta)
    coeff = [complex(c) for c in azimuthal_decomposition(field)]
    assert len(coeff) >= 3
    assert abs(coeff[0].real - 0.0) < 1e-12
    assert abs(coeff[1].real - 1.0) < 1e-12
    assert abs(coeff[2].real - 0.5) < 1e-12
    assert all(abs(c.imag) < 1e-12 for c in coeff)


def test_write_growth_rates(tmp_path):
    times = [0.0, 1.0, 2.0]
    spectra = [
        np.array([1.0, 1.0]),
        np.array([1.0, math.e]),
        np.array([1.0, math.e**2]),
    ]
    path = write_growth_rates(times, spectra, tmp_path)
    assert path.exists()
    if hasattr(np, "loadtxt"):
        data = np.loadtxt(path, delimiter=",")
        col = data[:, 1]
        assert len(col) == 2
        assert all(abs(float(v) - 1.0) < 1e-12 for v in col)


def test_instability_thresholds_trigger():
    solver = HallMHDSolver(instability_thresholds={"sausage": 0.9, "kink": 0.4})
    theta = np.linspace(0.0, 2 * np.pi, 32)[:-1]
    cos = lambda x: math.sin(x + math.pi / 2)
    mag = np.array([1.0 + 0.5 * cos(float(t)) for t in theta])
    zeros = np.zeros(len(mag))
    J = np.stack((mag, zeros, zeros), axis=-1)
    solver.compute_anomalous_resistivity(J)
    assert solver.sausage_onset
    assert solver.kink_onset
