import pytest

from dpf2.materials import (
    Species,
    sigmund_yield,
    yamamura_yield,
    impurity_source_terms,
)


def test_sigmund_yield_threshold():
    d = Species("D", Z=1, mass_u=2.0)
    cu = Species("Cu", Z=29, mass_u=63.5)
    y_low = sigmund_yield(d, cu, 10.0)
    assert y_low == 0.0
    y_high = sigmund_yield(d, cu, 100.0)
    assert y_high > 0.0
    assert pytest.approx(y_high, rel=1e-6) == 0.0065810636


def test_yamamura_angle_scaling():
    d = Species("D", Z=1, mass_u=2.0)
    cu = Species("Cu", Z=29, mass_u=63.5)
    y = yamamura_yield(d, cu, 100.0, 45.0)
    assert pytest.approx(y, rel=1e-6) == 0.0075262530


def test_impurity_source_terms():
    cu = Species("Cu", Z=29, mass_u=63.5)
    flux = impurity_source_terms(1e20, 0.01, cu)
    assert flux["Cu"] == pytest.approx(1e18)
