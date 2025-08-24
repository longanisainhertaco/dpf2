import numpy as np

from dpf2.mesh import Mesh3D
from dpf2.physics import HallMHD


def test_hall_mhd_ctu_update_on_mesh3d():
    mesh = Mesh3D(0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1)
    model = HallMHD()

    left = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    U = np.vstack([
        model.conservative_variables(left),
        model.conservative_variables(right),
    ])

    dt = 0.1 * mesh.dx / max(
        model.max_speed(U[0], "x"), model.max_speed(U[1], "x")
    )
    U_new = model.ctu_update(U, mesh, dt, periodic=True)

    assert U_new.shape == U.shape
    initial_mass = sum(cell[0] for cell in U)
    final_mass = sum(cell[0] for cell in U_new)
    assert np.isclose(initial_mass, final_mass)
