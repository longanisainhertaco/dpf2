import numpy as np

from dpf2.mesh import Mesh3D


def test_cell_indexing_and_centers():
    mesh = Mesh3D(0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2, 2, 2)
    first = mesh.cells[0]
    last = mesh.cells[-1]
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
