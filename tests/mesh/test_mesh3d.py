import numpy as np

import numpy as np

from dpf2.mesh import Mesh3D, apply_bc


def test_cell_indexing_and_centers():
    mesh = Mesh3D(0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2, 2, 2)
    first = mesh.cell(0, 0, 0)
    last = mesh.cell(1, 1, 1)
    assert (first.x_index, first.y_index, first.z_index) == (0, 0, 0)
    assert np.isclose(first.x_center, 0.5)
    assert np.isclose(first.y_center, 0.5)
    assert np.isclose(first.z_center, 0.5)
    assert (last.x_index, last.y_index, last.z_index) == (1, 1, 1)
    assert np.isclose(last.x_center, 1.5)
    assert np.isclose(last.y_center, 1.5)
    assert np.isclose(last.z_center, 1.5)


def test_neighbor_retrieval():
    mesh = Mesh3D(0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 3, 3, 3)
    neighbors = set(mesh.get_neighbors(1, 1, 1))
    expected = {
        (0, 1, 1),
        (2, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (1, 1, 0),
        (1, 1, 2),
    }
    assert neighbors == expected

    boundary = set(mesh.get_neighbors(0, 0, 0))
    assert boundary == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}


def test_metric_calculations():
    mesh = Mesh3D(0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 2, 4, 3)
    vol = mesh.cell_volume()
    ax, ay, az = mesh.face_areas()
    assert np.isclose(mesh.dx, 0.5)
    assert np.isclose(mesh.dy, 0.5)
    assert np.isclose(mesh.dz, 1.0)
    assert np.isclose(vol, 0.25)
    assert np.isclose(ax, 0.5)  # dy * dz
    assert np.isclose(ay, 0.5)  # dx * dz
    assert np.isclose(az, 0.25)  # dx * dy


def test_periodic_boundary_application():
    mesh = Mesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1)
    g = 1
    field = [
        [[0.0 for _ in range(mesh.nz + 2 * g)] for _ in range(mesh.ny + 2 * g)]
        for _ in range(mesh.nx + 2 * g)
    ]
    field[g][g][g] = 1.0
    field[g + 1][g][g] = 2.0

    apply_bc(field, "periodic", axis=0, side="low", ghosts=g)
    apply_bc(field, "periodic", axis=0, side="high", ghosts=g)

    assert field[0][g][g] == 2.0
    assert field[-1][g][g] == 1.0
