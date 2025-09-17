import numpy as np
import pytest

from diagnostics.instabilities import analyze_instabilities, fft_m_modes


def test_m_mode_growth_and_alert():
    n_theta = 32
    dt = 1.0
    times = np.arange(8)
    series = []
    for ti in times:
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        field = 1.0 + np.exp(0.2 * ti) * np.cos(theta)
        series.append(field)
    ez_series = times**2
    result = analyze_instabilities(
        series,
        dt,
        ez_series=ez_series,
        thresholds={"sausage": 0.1, "kink": 0.1},
    )
    assert result["growth_rates"]["m1"] == pytest.approx(0.2, rel=0.1)
    assert abs(result["growth_rates"]["m0"]) < 1e-3
    assert "kink" in result["alerts"]
    assert "beam_onset_time" in result


def test_fft_m_mode_amplitudes():
    n_theta = 8
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    field = 2 + 3 * np.cos(theta)
    modes = fft_m_modes(field)
    # m=0 mode amplitude should be 2
    assert modes[0] == pytest.approx(2.0, rel=1e-6)
    # m=1 mode amplitude should be 3/2 due to FFT normalisation
    assert modes[1] == pytest.approx(1.5, rel=1e-6)
