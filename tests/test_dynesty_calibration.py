import numpy as np
import pytest

from dpf2.uq.calibration import dynesty_calibrate_mass_current


def test_dynesty_calibrate_mass_current():
    pytest.importorskip("dynesty")
    rng = np.random.default_rng(0)

    current_sim = np.linspace(0.0, 1.0, 5)
    tof_sim = np.array([1.0])

    true_mass = 1.2
    true_current = 0.8

    current_data = true_current * current_sim + rng.normal(0, 0.01, size=current_sim.shape)
    tof_data = true_mass * tof_sim + rng.normal(0, 0.01, size=tof_sim.shape)

    samples = dynesty_calibrate_mass_current(
        current_sim,
        current_data,
        tof_sim,
        tof_data,
        bounds={"mass_factor": (0.5, 2.0), "current_factor": (0.5, 2.0)},
        n_live=10,
        n_iter=100,
        sigma_current=0.01,
        sigma_tof=0.01,
        seed=0,
    )

    mean_mass = float(np.mean(samples["mass_factor"]))
    mean_current = float(np.mean(samples["current_factor"]))

    assert abs(mean_mass - true_mass) < 0.2
    assert abs(mean_current - true_current) < 0.2
