from __future__ import annotations

from typing import Callable, Sequence


def interferometer_phase_shift(
    electron_density: Sequence[float],
    path_lengths: Sequence[float],
    wavelength: float,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> float:
    """Compute optical phase shift from line-integrated electron density.

    Parameters
    ----------
    electron_density:
        Electron density along the interferometer line of sight in m^-3.
    path_lengths:
        Path length segments corresponding to each density sample in meters.
    wavelength:
        Probe wavelength in meters.
    response_fn, noise_fn:
        Optional callables applied to the computed phase shift. ``response_fn``
        is evaluated first and ``noise_fn`` should return a noise contribution
        which is added to the response-corrected value.

    Returns
    -------
    float
        The accumulated phase shift in radians.
    """
    if len(electron_density) != len(path_lengths):
        raise ValueError("electron_density and path_lengths must be the same length")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    r_e = 2.8179403262e-15  # Classical electron radius in meters
    line_integral = 0.0
    for ne, dl in zip(electron_density, path_lengths):
        line_integral += float(ne) * float(dl)
    phase = -r_e * wavelength * line_integral
    if response_fn:
        phase = response_fn(phase)
    if noise_fn:
        phase += noise_fn(phase)
    return phase


__all__ = ["interferometer_phase_shift"]
