from dpf2.diagnostics import ez_beam_correlation


def test_ez_beam_correlation_basic():
    ez = [0.0, 1.0, 2.0, 3.0]
    ion = [0.0, 2.0, 4.0, 6.0]
    electron = [3.0, 2.0, 1.0, 0.0]
    corr = ez_beam_correlation(ez, ion, electron)
    assert corr["ion"] > 0.9
    assert corr["electron"] < -0.9
