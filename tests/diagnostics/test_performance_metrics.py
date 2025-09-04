import math

import pytest

from dpf2.diagnostics.performance_metrics import compute_performance_metrics


def test_basic_metrics():
    metrics = compute_performance_metrics(
        1e8,
        rep_rate_hz=2.0,
        energy_out_j=100.0,
        energy_in_j=500.0,
        electrode_mass_g=100.0,
        erosion_per_shot_g=0.01,
    )
    assert metrics["yield_per_shot"] == pytest.approx(1e8)
    assert metrics["yield_per_hour"] == pytest.approx(1e8 * 2.0 * 3600.0)
    assert metrics["wall_plug_efficiency"] == pytest.approx(0.2)
    expected_life = (100.0 / 0.01) / 2.0 / 3600.0
    assert metrics["lifetime_hours"] == pytest.approx(expected_life)


def test_zero_and_infinite_cases():
    metrics = compute_performance_metrics(
        0.0,
        rep_rate_hz=0.0,
        energy_out_j=0.0,
        energy_in_j=0.0,
        electrode_mass_g=10.0,
        erosion_per_shot_g=0.0,
    )
    assert metrics["wall_plug_efficiency"] == 0.0
    assert math.isinf(metrics["lifetime_hours"])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        compute_performance_metrics(
            1.0,
            rep_rate_hz=-1.0,
            energy_out_j=1.0,
            energy_in_j=1.0,
            electrode_mass_g=1.0,
            erosion_per_shot_g=1.0,
        )
    with pytest.raises(ValueError):
        compute_performance_metrics(
            1.0,
            rep_rate_hz=1.0,
            energy_out_j=-1.0,
            energy_in_j=1.0,
            electrode_mass_g=1.0,
            erosion_per_shot_g=1.0,
        )
