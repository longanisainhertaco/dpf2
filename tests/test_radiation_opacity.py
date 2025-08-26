import numpy as np
import pytest

# Skip tests if optional radiation dependencies are missing
pytest.importorskip("amrex")
pytest.importorskip("adios2")

from dpf2.simulation.radiation_model import RadiationModel


def _make_model(model: str, params: dict) -> RadiationModel:
    rm = RadiationModel.__new__(RadiationModel)
    rm.opacity_model = model
    rm.opacity_params = params
    return rm


def test_constant_opacity():
    rm = _make_model("constant", {"constant_opacity": 2.5})
    assert np.allclose(rm._compute_opacity(Te=0.0, ne=0.0, Z=0.0), 2.5)


def test_temperature_dependent_opacity():
    rm = _make_model(
        "temperature_dependent", {"base": 1.0, "alpha": 0.5, "beta": 2.0}
    )
    Te = 3.0
    expected = 1.0 + 0.5 * Te ** 2.0
    assert np.allclose(rm._compute_opacity(Te=Te, ne=0.0, Z=0.0), expected)

