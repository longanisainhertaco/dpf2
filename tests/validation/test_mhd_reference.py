import json
from pathlib import Path
import numpy as np

from dpf2.physics import ResistiveMHD
from dpf2.mesh import Mesh3D


def test_shock_tube_matches_reference():
    ref_path = Path(__file__).resolve().parents[2] / "ReferenceMaterial/mhd_shock_tube.json"
    rho_ref = np.array(json.loads(ref_path.read_text())["rho"])
    gamma = 1.4
    model = ResistiveMHD(gamma=gamma)
    n = len(rho_ref)
    mesh = Mesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n, 1, 1)

    left = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    U = np.zeros((n, len(model.equations)))
    for i in range(n):
        prim = left if i < n // 2 else right
        U[i] = model.conservative_variables(prim)

    dx = mesh.dx
    t = 0.0
    t_end = 0.1
    while t < t_end:
        max_speeds = [model.max_speed(state, "x") for state in U]
        dt = 0.4 * dx / max(max_speeds)
        if t + dt > t_end:
            dt = t_end - t
        fluxes = np.zeros((n + 1, len(model.equations)))
        for i in range(n - 1):
            F_L = model.flux_function(U[i], "x")
            F_R = model.flux_function(U[i + 1], "x")
            smax = max(model.max_speed(U[i], "x"), model.max_speed(U[i + 1], "x"))
            fluxes[i + 1] = 0.5 * (F_L + F_R) - 0.5 * smax * (U[i + 1] - U[i])
        fluxes[0] = model.flux_function(U[0], "x")
        fluxes[-1] = model.flux_function(U[-1], "x")
        for i in range(n):
            U[i] -= dt / dx * (fluxes[i + 1] - fluxes[i])
        t += dt

    rho_num = U[:, 0]
    l1 = np.mean(np.abs(rho_num - rho_ref))
    assert l1 < 0.1
