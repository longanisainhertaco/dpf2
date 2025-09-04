import pytest

from dpf2.materials import MaterialLibrary


def test_resistivity_vs_frequency():
    copper = MaterialLibrary.get("copper")
    r1 = copper.resistivity_at(1e5)
    r4 = copper.resistivity_at(4e5)
    assert pytest.approx(r4, rel=1e-12) == r1 * 2


def test_surface_conditioning():
    quartz = MaterialLibrary.get("quartz")
    assert pytest.approx(quartz.conditioned_field(100.0), rel=1e-12) == 120.0
