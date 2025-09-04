from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

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
    detector_names: Sequence[str] | None = None,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
    openpmd: bool = False,
) -> None:
    """Save anisotropic neutron spectrum in an HDF5 file with per-detector datasets.

    Parameters
    ----------
    path:
        Output file location.
    energies, angles, spectrum:
        Spectral information for each detector angle.
    detector_names:
        Optional list naming each detector.
    response_fn, noise_fn:
        Optional callables applied to the spectral data.  ``response_fn`` is
        applied first to each value while ``noise_fn`` should return a noise
        contribution for the already response-corrected value which will be
        added to it.
    openpmd:
        If ``True`` the file will include a minimal openPMD structure with
        datasets stored under ``/data/0`` and standard attributes.
    """

    spec_arr = [[float(v) for v in row] for row in spectrum]
    if len(spec_arr) != len(angles) or any(len(row) != len(energies) for row in spec_arr):
        raise ValueError("spectrum shape must be (n_angles, n_energies)")
    if detector_names is not None and len(detector_names) != len(angles):
        raise ValueError("detector_names must match number of angles")

    with h5py.File(path, "w") as fh:
        base = fh
        if openpmd:
            fh.attrs.update(
                {
                    "openPMD": "1.1.0",
                    "basePath": "/data/%T/",
                    "iterationEncoding": "groupBased",
                    "iterationFormat": "%T",
                    "software": "dpf2",
                }
            )
            base = fh.require_group("data/0")
        e_ds = base.create_dataset("energy_MeV", data=[float(e) for e in energies])
        e_ds.data = list(e_ds.data)
        e_ds.attrs["unitSI"] = 1.0
        a_ds = base.create_dataset("angle_deg", data=[float(a) for a in angles])
        a_ds.data = list(a_ds.data)
        a_ds.attrs["unitSI"] = 1.0
        grp = base.require_group("detectors")
        for i, row in enumerate(spec_arr):
            processed = [response_fn(v) if response_fn else v for v in row]
            if noise_fn:
                processed = [v + noise_fn(v) for v in processed]
            name = detector_names[i] if detector_names else f"detector_{i}"
            ds = grp.create_dataset(name, data=processed)
            ds.data = list(processed)
            ds.attrs["unitSI"] = 1.0


__all__ = [
    "compute_neutron_yield",
    "compute_beam_target_yield",
    "compute_thermonuclear_yield",
    "save_anisotropic_spectrum_hdf5",
]
