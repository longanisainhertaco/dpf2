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
