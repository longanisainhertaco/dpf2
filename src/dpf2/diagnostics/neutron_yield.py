from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import h5py_stub as h5py  # type: ignore


def compute_neutron_yield(reaction_rate: Sequence[float], dt: float) -> float:
    """Compute total neutron yield from a reaction rate history.

    Parameters
    ----------
    reaction_rate:
        One dimensional array containing the neutron production rate in
        neutrons/second for each time sample.
    dt:
        Time step between samples in seconds.

    Returns
    -------
    float
        Integrated neutron yield over the provided history.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    total = 0.0
    for val in reaction_rate:
        total += float(val)
    return total * dt


def compute_beam_target_yield(
    ion_distribution: Sequence[float],
    target_density: Sequence[float],
    cross_section: Sequence[float],
    dt: float,
) -> float:
    """Compute beam-target neutron yield from diagnostic inputs."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if not (
        len(ion_distribution) == len(target_density) == len(cross_section)
    ):
        raise ValueError("all inputs must have the same length")
    rate = [i * n * s for i, n, s in zip(ion_distribution, target_density, cross_section)]
    return compute_neutron_yield(rate, dt)


def compute_thermonuclear_yield(
    reactivity: Sequence[float], ion_density: Sequence[float], dt: float
) -> float:
    """Compute thermonuclear neutron yield from ion density and reactivity."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if len(reactivity) != len(ion_density):
        raise ValueError("reactivity and ion_density must be same length")
    rate = [r * n ** 2 for r, n in zip(reactivity, ion_density)]
    return compute_neutron_yield(rate, dt)


def save_anisotropic_spectrum_hdf5(
    path: str | Path,
    energies: Sequence[float],
    angles: Sequence[float],
    spectrum: Sequence[Sequence[float]],
) -> None:
    """Save anisotropic neutron spectrum in an HDF5 file."""
    spec_arr = np.asarray(spectrum, dtype=float)
    if spec_arr.shape != (len(angles), len(energies)):
        raise ValueError("spectrum shape must be (n_angles, n_energies)")
    with h5py.File(path, "w") as fh:
        e_ds = fh.create_dataset("energy_MeV", data=np.asarray(energies, dtype=float))
        e_ds.data = list(e_ds.data)
        a_ds = fh.create_dataset("angle_deg", data=np.asarray(angles, dtype=float))
        a_ds.data = list(a_ds.data)
        s_ds = fh.create_dataset("spectrum", data=spec_arr)
        s_ds.data = [list(row) for row in spec_arr]


__all__ = [
    "compute_neutron_yield",
    "compute_beam_target_yield",
    "compute_thermonuclear_yield",
    "save_anisotropic_spectrum_hdf5",
]
