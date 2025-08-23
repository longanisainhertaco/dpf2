import importlib.util
import pathlib
import sys
import numpy as np

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "dpf2" / "solvers" / "muscl_hancock.py"
spec = importlib.util.spec_from_file_location("muscl_hancock", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
MUSCLHancock = module.MUSCLHancock

GAMMA = 1.4


def primitive_to_conservative(rho, u, p, gamma=GAMMA):
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.stack([rho, rho * u, E], axis=-1)


def run_sod(n, t_end=0.2, cfl=0.8):
    solver = MUSCLHancock()
    x = (np.arange(n) + 0.5) / n
    dx = 1.0 / n

    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(x < 0.5, 1.0, 0.1)
    U = primitive_to_conservative(rho, u, p)

    t = 0.0
    while t < t_end:
        rho_p, u_p, p_p = solver._primitive(U)
        c = np.sqrt(GAMMA * p_p / rho_p)
        dt = cfl * dx / np.max(np.abs(u_p) + c)
        if t + dt > t_end:
            dt = t_end - t

        U_L, U_R = solver.reconstruct(U)
        U_Lh, U_Rh = solver.evolve_half_step(U_L, U_R, dt, dx)
        F = solver.compute_fluxes(U_Lh, U_Rh)
        U = U - dt / dx * (F[1:] - F[:-1])
        t += dt
    return x, U


def interpolate(U_ref, x_ref, x):
    out = np.empty((x.size, U_ref.shape[1]))
    for i in range(U_ref.shape[1]):
        out[:, i] = np.interp(x, x_ref, U_ref[:, i])
    return out


def test_sod_second_order():
    x_ref, U_ref = run_sod(400)
    x1, U1 = run_sod(50)
    x2, U2 = run_sod(100)

    U_ref_1 = interpolate(U_ref, x_ref, x1)
    U_ref_2 = interpolate(U_ref, x_ref, x2)

    err1 = np.mean(np.abs(U1 - U_ref_1), axis=0)
    err2 = np.mean(np.abs(U2 - U_ref_2), axis=0)

    # Expect notable error reduction when doubling resolution
    ratio = err1[0] / err2[0]
    assert ratio > 1.8
