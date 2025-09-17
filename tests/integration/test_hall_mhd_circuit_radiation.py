import pytest

# Require SciPy constants used by the Hall MHD solver
pytest.importorskip("scipy")

import numpy as np
from dpf2.hall_mhd_solver import HallMHDSolver, MHDState
from dpf2.core.circuit import RLCCircuitSolver
from dpf2.radiation.multigroup import MultiGroupDiffusion


def test_hall_mhd_circuit_radiation_energy_flux_conservation():
    """Run a coupled Hall-MHD, circuit and radiation scenario and verify budgets."""
    shape = (2, 2, 2)
    rho = np.ones(shape)
    v = np.zeros(shape + (3,))
    mom = rho[..., None] * v
    B = np.zeros(shape + (3,))
    B[..., 0] = 0.1
    B[..., 1] = -0.2
    B[..., 2] = 0.05
    p = 1.0
    gamma = 5.0 / 3.0
    kinetic = 0.5 * np.sum(v**2, axis=-1)
    magnetic = 0.5 * np.sum(B[0, 0, 0] ** 2)
    energy = p / (gamma - 1.0) + kinetic + magnetic
    energy = np.full(shape, energy)
    state = MHDState(rho=rho, mom=mom, energy=energy, B=B)

    radiation = MultiGroupDiffusion([0.01, 0.02], c=1.0)
    circuit = RLCCircuitSolver(L_ext=1.0, R_ext=0.0, C_ext=1.0, V0=1.0)
    solver = HallMHDSolver(radiation=radiation, circuit=circuit)

    dt = 1e-3
    steps = 5
    current = circuit.currents[-1]
    voltage = circuit.voltages[-1]

    def total_energy() -> float:
        plasma = float(np.sum(state.energy))
        rad = sum(float(np.sum(g)) for g in radiation.energy)
        circ = 0.5 * circuit.L_ext * current**2 + 0.5 * circuit.C_ext * voltage**2
        return plasma + rad + circ

    energy0 = total_energy()
    flux0 = np.sum(state.B, axis=(0, 1, 2))

    for _ in range(steps):
        state = solver.step(state, dt, current, voltage)
        current = circuit.currents[-1]
        voltage = circuit.voltages[-1]

    energy = total_energy()
    flux = np.sum(state.B, axis=(0, 1, 2))

    assert np.isclose(
        energy, energy0, rtol=1e-6
    ), f"Energy drifted by {energy - energy0:.3e}"
    assert np.allclose(
        flux, flux0, rtol=1e-6, atol=1e-12
    ), f"Flux drifted by {flux - flux0}"
