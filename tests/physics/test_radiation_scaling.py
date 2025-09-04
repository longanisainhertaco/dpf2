import numpy as np

from dpf2.mesh import Mesh3D
from dpf2.physics import HallMHD
from dpf2.radiation import bremsstrahlung_power


def _state(model: HallMHD, rho: float, T: float):
    """Construct a conservative state with given density and temperature."""

    p = rho * T
    E = p / (model.gamma - 1.0)
    return np.array([rho, 0.0, 0.0, 0.0, E, 0.0, 0.0, 0.0, 0.0])


def test_bremsstrahlung_scaling() -> None:
    model = HallMHD(radiation_model=bremsstrahlung_power)
    mesh = Mesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1)
    dt = 1.0

    def evolve(state):
        U = np.array([state, state.copy()])
        return model.ctu_update(U, mesh, dt)[0, 4]

    base = _state(model, 1.0, 1.0)
    loss_base = base[4] - evolve(base)

    dense = _state(model, 2.0, 1.0)
    loss_dense = dense[4] - evolve(dense)
    assert np.isclose(loss_dense / loss_base, 4.0)

    hot = _state(model, 1.0, 4.0)
    loss_hot = hot[4] - evolve(hot)
    assert np.isclose(loss_hot / loss_base, 2.0)
