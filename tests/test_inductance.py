"""Unit tests for the geometry-aware inductance helpers."""

from __future__ import annotations

import math

try:  # pragma: no cover - SciPy optional during tests
    from scipy.constants import mu_0
except Exception:  # pragma: no cover
    mu_0 = 4e-7 * math.pi

from dpf2.physics.inductance import (
    CoaxialGeometry,
    axial_inductance,
    dynamic_inductance,
    dynamic_inductance_with_derivatives,
    radial_inductance,
)


def test_axial_inductance_matches_coaxial_expression() -> None:
    geom = CoaxialGeometry(anode_radius=0.01, cathode_radius=0.05, anode_length=0.2)
    z = 0.12
    expected = (
        mu_0 / (2.0 * math.pi) * z * math.log(geom.cathode_radius / geom.anode_radius)
    )
    assert math.isclose(axial_inductance(z, geom), expected, rel_tol=1e-12)


def test_radial_inductance_matches_log_scaling() -> None:
    geom = CoaxialGeometry(
        anode_radius=0.01,
        cathode_radius=0.05,
        anode_length=0.2,
        pinch_length=0.03,
    )
    r = 0.02
    expected = (
        mu_0 * geom.pinch_span / (2.0 * math.pi) * math.log(geom.cathode_radius / r)
    )
    assert math.isclose(radial_inductance(r, geom), expected, rel_tol=1e-12)


def test_dynamic_inductance_combines_terms() -> None:
    geom = CoaxialGeometry(
        anode_radius=0.01,
        cathode_radius=0.05,
        anode_length=0.2,
        insulator_length=0.01,
        pinch_length=0.03,
        end_correction=5e-9,
    )
    z = 0.15
    r = 0.018
    total = dynamic_inductance(z, r, geom)
    expected = (
        geom.end_correction
        + geom.insulator_inductance
        + axial_inductance(z, geom)
        + radial_inductance(r, geom)
    )
    assert math.isclose(total, expected, rel_tol=1e-12)


def test_dynamic_inductance_derivatives() -> None:
    geom = CoaxialGeometry(anode_radius=0.01, cathode_radius=0.05, anode_length=0.2)
    z = 0.1
    r = 0.03
    L, dL_dz, dL_dr = dynamic_inductance_with_derivatives(z, r, geom)
    assert math.isclose(L, dynamic_inductance(z, r, geom), rel_tol=1e-12)
    expected_dz = (
        mu_0 / (2.0 * math.pi) * math.log(geom.cathode_radius / geom.anode_radius)
    )
    assert math.isclose(dL_dz, expected_dz, rel_tol=1e-12)
    expected_dr = -geom.radial_gradient_scale / r
    assert math.isclose(dL_dr, expected_dr, rel_tol=1e-12)
