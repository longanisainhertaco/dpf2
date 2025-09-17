"""Regression tests for the snowplow pinch model."""

from __future__ import annotations

import numpy as np

from dpf2.physics.inductance import dynamic_inductance
from dpf2.pinch_models import SnowplowPinchModel


def _synthetic_waveform(num: int = 400) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0e-6, num)
    I = 1.8e6 * np.sin(np.pi * t / t[-1]) ** 2
    return t, I


def test_snowplow_transitions_and_radius_collapse() -> None:
    model = SnowplowPinchModel()
    t, I = _synthetic_waveform()
    result = model.run(t, I)

    assert result.radius[0] == model.geometry.cathode_radius
    assert result.axial_position[-1] >= model.geometry.anode_length * 0.95
    assert result.radius[-1] < model.geometry.cathode_radius * 0.7
    assert np.all(np.diff(result.rayleigh_taylor_growth) >= -1e-6)
    assert result.neutron_yield > 0.0


def test_snowplow_inductance_consistency() -> None:
    model = SnowplowPinchModel()
    t, I = _synthetic_waveform()
    result = model.run(t, I)
    Lp = dynamic_inductance(result.axial_position, result.radius, model.geometry)
    energy_expected = 0.5 * Lp * I**2
    assert np.allclose(result.energy, energy_expected, rtol=1e-6, atol=1e-9)
