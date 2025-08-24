import numpy as np

from dpf2.mesh import Mesh3D, apply_bc
from dpf2.physics.mhd import ResistiveMHD


def test_solver_update_with_periodic_boundaries():
    mesh = Mesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1)
    model = ResistiveMHD()

    left = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    U_left = model.conservative_variables(left)
    U_right = model.conservative_variables(right)

    g = 1
    state = [
        [
            [[0.0 for _ in range(len(model.equations))] for _ in range(mesh.nz + 2 * g)]
            for _ in range(mesh.ny + 2 * g)
        ]
        for _ in range(mesh.nx + 2 * g)
    ]
    state[1][1][1] = list(U_left)
    state[2][1][1] = list(U_right)

    for var in range(len(model.equations)):
        field = [
            [
                [state[i][j][k][var] for k in range(mesh.nz + 2 * g)]
                for j in range(mesh.ny + 2 * g)
            ]
            for i in range(mesh.nx + 2 * g)
        ]
        apply_bc(field, "periodic", axis=0, side="low", ghosts=g)
        apply_bc(field, "periodic", axis=0, side="high", ghosts=g)
        for i in range(mesh.nx + 2 * g):
            for j in range(mesh.ny + 2 * g):
                for k in range(mesh.nz + 2 * g):
                    state[i][j][k][var] = field[i][j][k]

    F_L = model.flux_function(state[2][1][1], "x")
    F_R = model.flux_function(state[3][1][1], "x")
    smax = max(
        model.max_speed(state[2][1][1], "x"),
        model.max_speed(state[3][1][1], "x"),
    )
    dt = 0.1 * mesh.dx / smax
    flux = [
        0.5 * (fl + fr) - 0.5 * smax * (sr - sl)
        for fl, fr, sr, sl in zip(F_L, F_R, state[3][1][1], state[2][1][1])
    ]
    state[2][1][1] = [u - dt / mesh.dx * f for u, f in zip(state[2][1][1], flux)]
    state[1][1][1] = [u + dt / mesh.dx * f for u, f in zip(state[1][1][1], flux)]

    total_mass = state[1][1][1][0] + state[2][1][1][0]
    assert np.isclose(total_mass, U_left[0] + U_right[0])

    dt_geo = model.stable_timestep(state[1][1][1], mesh, cfl=1.0)
    vol = mesh.cell_volume()
    areas = mesh.face_areas()
    speeds = [model.max_speed(state[1][1][1], d) for d in ["x", "y", "z"]]
    expected = min(vol / (a * v) if v > 0 else np.inf for a, v in zip(areas, speeds))
    assert np.isclose(dt_geo, expected)

