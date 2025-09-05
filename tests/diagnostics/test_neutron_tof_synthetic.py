import math

from dpf2.diagnostics.neutron.tof_synthetic import (
    synthetic_tof_from_iv,
    cross_correlation_with_iv,
)


def _iv_history():
    current = [0.0, 1.0, 5.0, 1.0, 0.0]
    voltage = [0.0, 1.0, 5.0, 1.0, 0.0]
    return current, voltage


def test_tof_generation():
    current, voltage = _iv_history()
    dt = 1e-9
    distance = 1.0
    energies = [2.45]
    times, counts = synthetic_tof_from_iv(current, voltage, dt, distance, energies)
    idx = next(i for i, v in enumerate(counts) if v > 0)
    tof_time = times[idx]
    m_n = 1.67492749804e-27
    E = energies[0] * 1.602176634e-13
    v = math.sqrt(2.0 * E / m_n)
    expected = 2 * dt + 1.0 / v
    assert abs(tof_time - expected) / expected < 0.1


def test_correlation_and_alignment():
    current, voltage = _iv_history()
    dt = 1e-9
    distance = 1.0
    energies = [2.45]
    offset = 3e-9
    # Generate without alignment
    _, counts = synthetic_tof_from_iv(
        current, voltage, dt, distance, energies, time_offset=offset
    )
    lags, corr = cross_correlation_with_iv(counts, current, voltage, dt)
    best_lag = lags[corr.index(max(corr))]
    m_n = 1.67492749804e-27
    E = energies[0] * 1.602176634e-13
    v = math.sqrt(2.0 * E / m_n)
    expected = offset + 1.0 / v
    assert abs(best_lag - expected) / expected < 0.1

    # Now align peaks and verify the correlation maximum is near zero
    _, aligned = synthetic_tof_from_iv(
        current, voltage, dt, distance, energies, time_offset=offset, align_peaks=True
    )
    lags2, corr2 = cross_correlation_with_iv(aligned, current, voltage, dt)
    best_lag2 = lags2[corr2.index(max(corr2))]
    assert abs(best_lag2) < dt
