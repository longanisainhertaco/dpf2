from pathlib import Path
import csv
import numpy as np
from scipy.constants import mu_0, pi

from dpf2.physics import LowerHybridDrift, MZeroInstability


def test_lower_hybrid_frequency_and_growth():
    model = LowerHybridDrift(B=1.0, n_i=1e19)
    freq = model.frequency()
    growth = model.growth_rate(0.5)
    assert freq > 0
    assert growth > 0


def test_m0_instability_growth():
    instab = MZeroInstability(current=1e5, radius=0.01, density=1e-3)
    rate = instab.growth_rate()
    assert rate > 0


def _load_pf1000(field: str) -> np.ndarray:
    base = Path("data/benchmarks/PF1000")
    with open(base / f"{field}.csv") as f:
        reader = csv.DictReader(f)
        return np.array([float(row["value"]) for row in reader])


def test_m0_instability_pf1000_growth_rates():
    current = _load_pf1000("current")
    radius = _load_pf1000("radius") / 100.0  # cm -> m
    density = np.full_like(current, 1e-3)
    instab = MZeroInstability(current=current, radius=radius, density=density)
    rates = instab.growth_rate()
    expected = np.abs(mu_0 * current / (2 * pi * radius)) / np.sqrt(mu_0 * density)
    assert np.allclose(rates, expected)
    assert np.all(rates >= 0)
