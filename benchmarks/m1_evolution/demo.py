"""Reduced 3D benchmark demonstrating m=1 mode evolution.

This script generates a small synthetic data set where an m=1 kink mode
grows exponentially. The :mod:`diagnostics.instabilities` module is used
to extract growth rates and trigger instability alerts.
"""
from __future__ import annotations

import numpy as np

from diagnostics.instabilities import analyze_instabilities


def main() -> None:
    n_theta = 32
    n_z = 4
    n_r = 4
    dt = 1.0
    times = np.arange(10)
    series = []
    for ti in times:
        theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
        base = np.ones((n_z, n_r, n_theta))
        m1 = np.cos(theta)[None, None, :] * np.exp(0.3 * ti)
        series.append(base + m1)
    ez_series = np.sin(0.1 * times) + 0.5 * times
    result = analyze_instabilities(series, dt, ez_series)
    print("Growth rates:", result["growth_rates"])
    print("Alerts:", result["alerts"])
    if "beam_onset_time" in result:
        print("Beam onset time:", result["beam_onset_time"])


if __name__ == "__main__":
    main()
