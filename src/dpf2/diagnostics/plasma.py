from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import h5py_stub as h5py  # type: ignore


def save_density_temperature_map_hdf5(
    path: str | Path,
    density: Sequence[Sequence[float]],
    temperature: Sequence[Sequence[float]],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
    openpmd: bool = False,
) -> None:
    """Save 2D density and temperature maps to an HDF5 file.

    Parameters
    ----------
    path:
        Output file location.
    density, temperature:
        Two dimensional arrays with identical shape.
    response_fn, noise_fn:
        Optional callables applied to each value. ``response_fn`` is evaluated
        first and ``noise_fn`` should return a noise contribution which is then
        added to the response-corrected value.
    openpmd:
        If ``True`` the file will include a minimal openPMD structure with
        datasets stored under ``/data/0`` and standard attributes.
    """

    dens_arr = [[float(v) for v in row] for row in density]
    temp_arr = [[float(v) for v in row] for row in temperature]
    if len(dens_arr) != len(temp_arr) or any(len(d) != len(t) for d, t in zip(dens_arr, temp_arr)):
        raise ValueError("density and temperature must have the same shape")

    def _process(arr: Sequence[Sequence[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for row in arr:
            proc = [response_fn(v) if response_fn else v for v in row]
            if noise_fn:
                proc = [v + noise_fn(v) for v in proc]
            out.append([float(v) for v in proc])
        return out

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
        dens_ds = base.create_dataset("density", data=_process(dens_arr))
        dens_ds.data = [row[:] for row in dens_ds.data]
        dens_ds.attrs["unitSI"] = 1.0
        temp_ds = base.create_dataset("temperature", data=_process(temp_arr))
        temp_ds.data = [row[:] for row in temp_ds.data]
        temp_ds.attrs["unitSI"] = 1.0


def compute_eedf(energies: Sequence[float], bin_edges: Sequence[float]) -> tuple[list[float], list[float]]:
    """Compute an electron energy distribution function (EEDF).

    Parameters
    ----------
    energies:
        Sequence of particle energies (eV).
    bin_edges:
        Monotonic sequence defining the histogram bin edges.

    Returns
    -------
    tuple of (bin_centers, counts)
        Histogram counts for each energy bin and the corresponding bin centers.
    """
    if len(bin_edges) < 2:
        raise ValueError("at least two bin edges required")
    counts = [0 for _ in range(len(bin_edges) - 1)]
    for e in energies:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= e < bin_edges[i + 1]:
                counts[i] += 1
                break
    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(len(bin_edges) - 1)]
    return centers, counts


def save_eedf_hdf5(
    path: str | Path,
    energies: Sequence[float],
    distribution: Sequence[float],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
    openpmd: bool = False,
) -> None:
    """Save an EEDF to an HDF5 file."""

    eng_arr = [float(e) for e in energies]
    dist_arr = [float(v) for v in distribution]
    if len(eng_arr) != len(dist_arr):
        raise ValueError("energies and distribution must have the same length")

    processed = [response_fn(v) if response_fn else v for v in dist_arr]
    if noise_fn:
        processed = [v + noise_fn(v) for v in processed]

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
        e_ds = base.create_dataset("energy_eV", data=eng_arr)
        e_ds.data = list(e_ds.data)
        e_ds.attrs["unitSI"] = 1.0
        d_ds = base.create_dataset("distribution", data=processed)
        d_ds.data = list(d_ds.data)
        d_ds.attrs["unitSI"] = 1.0


__all__ = [
    "save_density_temperature_map_hdf5",
    "compute_eedf",
    "save_eedf_hdf5",
]
