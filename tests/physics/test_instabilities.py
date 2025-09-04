import csv
import numpy as np
import math

mu_0 = 4e-7 * np.pi
pi = np.pi

from dpf2.physics import HallMHD, LowerHybridDrift, MZeroInstability


def _load_pf1000(field: str):
    with open(f"data/benchmarks/PF1000/{field}.csv") as f:
        next(f)  # skip header
        return np.array([float(line.split(",")[1]) for line in f])


def test_lower_hybrid_grid_evolution():
    model = LowerHybridDrift(B=1.0, n_i=1e19)
    k = np.linspace(0.0, 1.0, 5)
    amp0 = np.zeros_like(k) + 1e-3
    evolved = model.evolve(amp0, k, dt=1.0)
    rates = model.growth_rate(k)
    expected = [a * math.exp(max(min(r, 50.0), -50.0)) for a, r in zip(amp0, rates)]
    assert np.allclose(evolved, expected)


def test_m0_instability_pf1000_evolution():
    current = _load_pf1000('current')
    radius = _load_pf1000('radius') / 100.0  # cm -> m
    density = np.zeros_like(current) + 1e-3
    instab = MZeroInstability(current=current, radius=radius, density=density)
    amp0 = np.zeros_like(current) + 1e-3
    evolved = instab.evolve(amp0, dt=1.0)
    rates = np.array([
        abs(mu_0 * c / (2 * pi * r)) / math.sqrt(mu_0 * d)
        for c, r, d in zip(current, radius, density)
    ])
    expected = amp0 * np.exp(rates)
    assert np.allclose(evolved, expected)


def test_hall_mhd_instability_coupling():
    instab = MZeroInstability(current=1e5, radius=0.01, density=1e-3)
    amp = instab.evolve(1e-3, dt=1.0)
    model = HallMHD()
    model.step(np.zeros(9), dt=1.0, current=0.0, instability_amp=amp)
    assert model.back_emf == amp
    assert model.beam_velocity > 0.0
