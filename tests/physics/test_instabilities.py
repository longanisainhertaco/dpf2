import numpy as np
import pytest

from dpf2.physics import (
    LowerHybridDrift,
    MZeroInstability,
    PhysicalPICDriver,
    SimplePIC,
)
from dpf2.hall_mhd_solver import HallMHDSolver, MHDState

mu_0 = 4e-7 * np.pi
pi = np.pi


def _load_waveform(device: str, field: str):
    values = []
    with open(f"data/benchmarks/{device}/{field}.csv") as f:
        next(f)
        for line in f:
            values.append(float(line.split(",")[1]))
    return np.array(values)


def _state_with_gradient(shape):
    rho = np.ones(shape)
    mom = np.zeros(shape + (3,))
    B = np.zeros(shape + (3,))
    x = np.linspace(0.0, 1.0, shape[0])
    for i, val in enumerate(x):
        for j in range(shape[1]):
            for k in range(shape[2]):
                B[i, j, k, 2] = val
    energy = np.ones(shape)
    return MHDState(rho=rho, mom=mom, energy=energy, B=B)


def test_lower_hybrid_grid_evolution():
    model = LowerHybridDrift(B=1.0, n_i=1e19)
    k = np.zeros((2, 2, 2)) + 0.1
    amp0 = np.zeros_like(k) + 1e-3
    evolved = model.evolve(amp0, k, dt=1.0)
    rates = model.growth_rate(k)
    expected = amp0 * np.exp(np.clip(rates, -50.0, 50.0))
    assert np.allclose(evolved, expected)


def test_m0_instability_pf1000_evolution():
    current_1d = _load_waveform("PF1000", "current")
    radius_1d = _load_waveform("PF1000", "radius") / 100.0
    n = len(current_1d)
    current = np.zeros((n, 1, 1))
    radius = np.zeros((n, 1, 1))
    for i in range(n):
        current[i, 0, 0] = current_1d[i]
        radius[i, 0, 0] = radius_1d[i]
    density = np.zeros((n, 1, 1)) + 1e-3
    instab = MZeroInstability(current=current, radius=radius, density=density)
    amp0 = np.zeros((n, 1, 1)) + 1e-3
    evolved = instab.evolve(amp0, dt=1.0)
    rates = np.abs(mu_0 * current / (2 * pi * radius)) / np.sqrt(mu_0 * density)
    expected = amp0 * np.exp(np.clip(rates, -50.0, 50.0))
    assert np.allclose(evolved, expected)


def test_m0_instability_mjolnir_evolution():
    current_1d = _load_waveform("LLNL_MJOLNIR", "current")
    radius_1d = _load_waveform("LLNL_MJOLNIR", "radius") / 100.0
    n = len(current_1d)
    current = np.zeros((n, 1, 1))
    radius = np.zeros((n, 1, 1))
    for i in range(n):
        current[i, 0, 0] = current_1d[i]
        radius[i, 0, 0] = radius_1d[i]
    density = np.zeros((n, 1, 1)) + 1e-3
    instab = MZeroInstability(current=current, radius=radius, density=density)
    amp0 = np.zeros((n, 1, 1)) + 1e-3
    evolved = instab.evolve(amp0, dt=1.0)
    rates = np.abs(mu_0 * current / (2 * pi * radius)) / np.sqrt(mu_0 * density)
    expected = amp0 * np.exp(np.clip(rates, -50.0, 50.0))
    assert np.allclose(evolved, expected)


@pytest.mark.skipif(not hasattr(np, "roll"), reason="requires full NumPy")
def test_hall_mhd_instability_coupling():
    shape = (4, 4, 4)
    state = _state_with_gradient(shape)
    instab = MZeroInstability(current=1e5, radius=0.01, density=1e-3)
    amp_field = instab.evolve(np.full(shape, 1e-3), dt=1.0)
    solver = HallMHDSolver(anomalous_resistivity=instab.anomalous_resistivity)
    solver.step(state, dt=0.1)
    J = solver.last_J
    expected = amp_field[..., None] * J
    assert np.allclose(solver.last_E, expected)


def test_pic_driver_beam_anisotropy():
    pic = SimplePIC(
        charge=1.0,
        mass=1.0,
        length=1.0,
        positions=[0.0, 0.1, -0.1],
        velocities=[1e5, 1e5, 1.2e5],
    )
    driver = PhysicalPICDriver(pic)
    state = MHDState(
        rho=np.ones((1, 1, 1)),
        mom=np.zeros((1, 1, 1, 3)),
        energy=np.ones((1, 1, 1)),
        B=np.zeros((1, 1, 1, 3)),
    )
    driver.step(state, current=1e5, dt=1e-9)
    pos, vel = driver.exchange_particles()
    anis = np.var(vel[:, 2]) / (np.var(vel[:, 0]) + np.var(vel[:, 1]) + 1e-30)
    assert anis > 1e6
