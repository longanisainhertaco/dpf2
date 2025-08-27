import math

from dpf2.mesh import Mesh3D, apply_bc


def test_curved_surface_and_interpolation():
    mesh = Mesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1)
    surface_fn = lambda x, y: 1 + 0.1 * math.sin(math.pi * x) * math.sin(math.pi * y)
    surface = mesh.map_curved_boundary(surface_fn)
    expected_surface = surface_fn(0.5, 0.5)
    assert len(surface) == 1 and len(surface[0]) == 1
    assert abs(surface[0][0] - expected_surface) <= 1e-8

    g = 1
    field = [
        [
            [0.0 for _ in range(mesh.nz + 2 * g)] for _ in range(mesh.ny + 2 * g)
        ]
        for _ in range(mesh.nx + 2 * g)
    ]
    field[g][g][g] = 10.0

    value_fn = lambda x, y, z: 20.0
    mesh.interpolate_ghost_cells(
        field, axis=2, side="high", ghosts=g, surface=surface_fn, value=value_fn
    )

    z_c = 0.5 * (mesh.z[-2] + mesh.z[-1])
    z_surf = surface_fn(0.5, 0.5)
    expected = 10.0 + (20.0 - 10.0) * mesh.dz / (z_surf - z_c)
    assert abs(field[g][g][g + 1] - expected) <= 1e-8


def test_curved_surface_interpolation_x():
    mesh = Mesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1)
    surface_fn = lambda y, z: 1 + 0.1 * math.sin(math.pi * y) * math.sin(math.pi * z)
    g = 1
    field = [
        [
            [0.0 for _ in range(mesh.nz + 2 * g)] for _ in range(mesh.ny + 2 * g)
        ]
        for _ in range(mesh.nx + 2 * g)
    ]
    field[g][g][g] = 10.0
    value_fn = lambda x, y, z: 20.0
    mesh.interpolate_ghost_cells(
        field, axis=0, side="high", ghosts=g, surface=surface_fn, value=value_fn
    )
    x_c = 0.5 * (mesh.x[-2] + mesh.x[-1])
    x_surf = surface_fn(0.5, 0.5)
    expected = 10.0 + (20.0 - 10.0) * mesh.dx / (x_surf - x_c)
    assert abs(field[g + 1][g][g] - expected) <= 1e-8


def test_curved_surface_interpolation_y():
    mesh = Mesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1)
    surface_fn = lambda x, z: -0.1 * math.sin(math.pi * x) * math.sin(math.pi * z)
    g = 1
    field = [
        [
            [0.0 for _ in range(mesh.nz + 2 * g)] for _ in range(mesh.ny + 2 * g)
        ]
        for _ in range(mesh.nx + 2 * g)
    ]
    field[g][g][g] = 10.0
    value_fn = lambda x, y, z: 20.0
    mesh.interpolate_ghost_cells(
        field, axis=1, side="low", ghosts=g, surface=surface_fn, value=value_fn
    )
    y_c = 0.5 * (mesh.y[0] + mesh.y[1])
    y_surf = surface_fn(0.5, 0.5)
    expected = 10.0 + (20.0 - 10.0) * mesh.dy / (y_c - y_surf)
    assert abs(field[g][g - 1][g] - expected) <= 1e-8


def test_reflective_and_absorbing_bc():
    mesh = Mesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1)
    g = 1
    field = [
        [
            [0.0 for _ in range(mesh.nz + 2 * g)] for _ in range(mesh.ny + 2 * g)
        ]
        for _ in range(mesh.nx + 2 * g)
    ]
    field[1][1][1] = 5.0

    apply_bc(field, "reflective", axis=0, side="low", ghosts=g)
    apply_bc(field, "reflective", axis=0, side="high", ghosts=g)
    assert field[0][1][1] == field[2][1][1] == 5.0

    apply_bc(field, "absorbing", axis=1, side="low", ghosts=g)
    assert field[1][0][1] == 0.0
