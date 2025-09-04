import numpy as np
import pytest

from dpf2.uq.calibration import dynesty_calibrate_waveform


def test_dynesty_calibrate_waveform():
    pytest.importorskip("dynesty")
    rng = np.random.default_rng(0)
    t_sim = np.linspace(0.0, 1.0, 50)
    current_sim = np.sin(2 * np.pi * t_sim)
    true_mass = 1.1
    true_current = 0.9
    t_data = t_sim
    current_true = true_current * np.interp(t_data, true_mass * t_sim, current_sim, left=0.0, right=0.0)
    current_data = current_true + rng.normal(0, 0.01, size=t_data.shape)

    samples = dynesty_calibrate_waveform(
        t_sim,
        current_sim,
        t_data,
        current_data,
        bounds={"mass_factor": (0.5, 2.0), "current_factor": (0.5, 2.0)},
        n_live=20,
        n_iter=200,
        sigma=0.01,
        seed=0,
    )
    mean_mass = float(np.mean(samples["mass_factor"]))
    mean_current = float(np.mean(samples["current_factor"]))
    assert abs(mean_mass - true_mass) < 0.2
    assert abs(mean_current - true_current) < 0.2
