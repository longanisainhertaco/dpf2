from __future__ import annotations

from math import pi, sqrt
from pathlib import Path
from typing import Callable, Sequence

import h5py_stub as h5py  # type: ignore

try:  # pragma: no cover - SciPy optional
    from scipy.constants import mu_0, k as k_B, m_p
except Exception:  # pragma: no cover - fallback values
    mu_0 = 4e-7 * pi
    k_B = 1.380649e-23
    m_p = 1.67262192369e-27


def bennett_radius(I: float, n: float, T: float) -> float:
    """Return the Bennett pinch radius.

    Parameters
    ----------
    I:
        Pinch current in amperes.
    n:
        Number density in m^-3.
    T:
        Plasma temperature in kelvin.
    """

    if n <= 0 or T <= 0:
        raise ValueError("n and T must be positive")
    return sqrt(2 * k_B * T / (mu_0 * n)) / abs(I)


def plasma_beta(n: float, T: float, B: float) -> float:
    """Compute plasma beta for a cell."""

    if B == 0:
        # In very early phases of a simulation the magnetic field can be zero
        # which would otherwise raise an exception and halt diagnostics.  In
        # that regime the beta parameter is formally infinite, so we return a
        # large value instead of raising.
        return float("inf")
    pressure = n * k_B * T
    return 2 * mu_0 * pressure / (B * B)


def alfven_mach_number(v: float, B: float, n: float) -> float:
    """Return the Alfven Mach number for a cell."""

    rho = n * m_p
    v_a = B / sqrt(mu_0 * rho)
    if v_a == 0:
        return float("inf")
    return v / v_a


def magnetic_reynolds_number(v: float, L: float, sigma: float) -> float:
    """Compute the magnetic Reynolds number."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return mu_0 * sigma * v * L


def lundquist_number(B: float, n: float, L: float, sigma: float) -> float:
    """Compute the Lundquist number."""

    rho = n * m_p
    v_a = B / sqrt(mu_0 * rho)
    return mu_0 * sigma * v_a * L


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
    "bennett_radius",
    "plasma_beta",
    "alfven_mach_number",
    "magnetic_reynolds_number",
    "lundquist_number",
    "save_density_temperature_map_hdf5",
    "compute_eedf",
    "save_eedf_hdf5",
]
