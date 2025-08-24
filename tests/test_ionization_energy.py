import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ionization", Path(__file__).resolve().parents[1] / "dpf2" / "ionization.py"
)
ionization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ionization)  # type: ignore[attr-defined]

from dpf2.physics_models import PhysicsModels


def test_equilibrium_root_solver():
    n_total = 1e20  # m^-3
    T = 2e4  # K
    ne = ionization.equilibrium_electron_density(n_total, T)
    # check bounds and near-zero derivative
    assert 0.0 <= ne <= n_total
    assert abs(ionization.collisional_radiative_rhs(ne, n_total, T)) < 1e-3 * n_total


def test_physics_model_energy_sink_update():
    model = PhysicsModels.with_defaults()
    n_total = 1e20
    T = 1e4
    ne0 = 0.1 * n_total
    dt = 1e-9
    ne1 = model.update_electron_density(ne0, n_total, T, dt)
    assert ne1 != ne0
    sink = model.ionization_energy_sink(ne1, n_total, T)
    assert sink >= 0.0
