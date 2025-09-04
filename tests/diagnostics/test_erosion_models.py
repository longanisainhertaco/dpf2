import math

import pytest

from dpf2.diagnostics import estimate_lifetime_sputtering


def test_estimate_lifetime_sputtering():
    lifetime = estimate_lifetime_sputtering(
        1e15,
        electrode_area_cm2=10.0,
        electrode_thickness_cm=0.1,
        material_density_g_cm3=8.0,
        atomic_mass_g_mol=63.546,
        rep_rate_hz=1.0,
    )
    avog = 6.02214076e23
    mass_per_shot = 1e15 * 10.0 * 63.546 / avog
    total_mass = 10.0 * 0.1 * 8.0
    expected = total_mass / mass_per_shot / 3600.0
    assert lifetime == pytest.approx(expected)


def test_zero_sputtering_rate():
    life = estimate_lifetime_sputtering(
        0.0,
        electrode_area_cm2=1.0,
        electrode_thickness_cm=1.0,
        material_density_g_cm3=1.0,
        atomic_mass_g_mol=1.0,
        rep_rate_hz=1.0,
    )
    assert math.isinf(life)
