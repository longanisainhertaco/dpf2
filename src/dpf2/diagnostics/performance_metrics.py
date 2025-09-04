from __future__ import annotations

"""Utilities for computing simple key performance indicators.

The production code only needs a very small subset of KPIs in order to feed
other parts of the UI.  The implementation here intentionally keeps the
interface extremely small and returns a dictionary so that the front-end can
consume it without having to understand any additional classes.
"""

from typing import Dict
from pathlib import Path
import csv

try:  # pragma: no cover - matplotlib optional
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib may be absent
    plt = None

import h5py


AVOGADRO = 6.022_140_76e23  # mol^-1


def compute_performance_metrics(
    yield_per_shot: float,
    *,
    rep_rate_hz: float,
    energy_out_j: float,
    energy_in_j: float,
    electrode_mass_g: float,
    erosion_per_shot_g: float,
) -> Dict[str, float]:
    """Compute basic performance KPIs for a DPF system.

    Parameters
    ----------
    yield_per_shot:
        Neutron yield produced per individual shot.
    rep_rate_hz:
        Firing repetition rate in Hertz.
    energy_out_j:
        Output energy per shot in joules.
    energy_in_j:
        Input energy per shot in joules.
    electrode_mass_g:
        Mass of the sacrificial electrode material in grams available before
        replacement is required.
    erosion_per_shot_g:
        Electrode mass lost per shot in grams.

    Returns
    -------
    dict
        Mapping containing ``yield_per_shot``, ``yield_per_hour``,
        ``wall_plug_efficiency`` and ``lifetime_hours``.
    """
    if rep_rate_hz < 0:
        raise ValueError("rep_rate_hz must be non-negative")
    if energy_in_j < 0 or energy_out_j < 0:
        raise ValueError("energies must be non-negative")
    if electrode_mass_g < 0:
        raise ValueError("electrode_mass_g must be non-negative")
    if erosion_per_shot_g < 0:
        raise ValueError("erosion_per_shot_g must be non-negative")

    # How many neutrons are produced per hour.  ``rep_rate_hz`` is the firing
    # rate, so we convert to hours by multiplying by ``3600``.
    yield_hour = yield_per_shot * rep_rate_hz * 3600.0

    # The wall plug efficiency is defined as the ratio of output to input
    # energy.  If the input energy is zero the efficiency is defined to be zero
    # rather than raising an error in order to keep the API convenient for the
    # front end.
    wall = energy_out_j / energy_in_j if energy_in_j > 0 else 0.0

    # ``erosion_per_shot_g`` can be zero when erosion has not been measured.
    # In such cases the lifetime is effectively unbounded.
    if erosion_per_shot_g == 0 or rep_rate_hz == 0:
        lifetime = float("inf")
    else:
        shots = electrode_mass_g / erosion_per_shot_g
        lifetime = shots / rep_rate_hz / 3600.0

    return {
        "yield_per_shot": float(yield_per_shot),
        "yield_per_hour": float(yield_hour),
        "wall_plug_efficiency": float(wall),
        "lifetime_hours": float(lifetime),
    }


def estimate_lifetime_sputtering(
    sputtering_rate_atoms_per_cm2: float,
    *,
    electrode_area_cm2: float,
    electrode_thickness_cm: float,
    material_density_g_cm3: float,
    atomic_mass_g_mol: float,
    rep_rate_hz: float,
) -> float:
    """Estimate electrode lifetime from a simple sputtering model.

    Parameters
    ----------
    sputtering_rate_atoms_per_cm2:
        Number of atoms ejected per square centimetre for each shot.
    electrode_area_cm2:
        Exposed surface area of the electrode.
    electrode_thickness_cm:
        Thickness of material available before replacement is required.
    material_density_g_cm3:
        Density of the electrode material.
    atomic_mass_g_mol:
        Atomic mass of the electrode material in grams per mole.
    rep_rate_hz:
        Shot repetition rate.

    Returns
    -------
    float
        Estimated lifetime in hours.  ``inf`` is returned when no erosion occurs
        or the repetition rate is zero.
    """

    if sputtering_rate_atoms_per_cm2 < 0:
        raise ValueError("sputtering_rate_atoms_per_cm2 must be non-negative")
    if electrode_area_cm2 < 0:
        raise ValueError("electrode_area_cm2 must be non-negative")
    if electrode_thickness_cm < 0:
        raise ValueError("electrode_thickness_cm must be non-negative")
    if material_density_g_cm3 < 0:
        raise ValueError("material_density_g_cm3 must be non-negative")
    if atomic_mass_g_mol <= 0:
        raise ValueError("atomic_mass_g_mol must be positive")
    if rep_rate_hz < 0:
        raise ValueError("rep_rate_hz must be non-negative")

    mass_per_shot = (
        sputtering_rate_atoms_per_cm2
        * electrode_area_cm2
        * atomic_mass_g_mol
        / AVOGADRO
    )
    total_mass = electrode_area_cm2 * electrode_thickness_cm * material_density_g_cm3

    if mass_per_shot == 0 or rep_rate_hz == 0:
        return float("inf")

    shots = total_mass / mass_per_shot
    return shots / rep_rate_hz / 3600.0


def export_performance_metrics(metrics: Dict[str, float], output_dir: Path) -> None:
    """Save KPI data to CSV, HDF5 and generate basic visualisations.

    The following files are created inside ``output_dir``:

    ``performance_metrics.csv``
        Table of key/value pairs.
    ``performance_metrics.h5``
        HDF5 file with one dataset per KPI.
    ``summary.md``
        Markdown table summarising the metrics.
    ``performance_metrics.png``
        Bar chart of the KPI values (if :mod:`matplotlib` is available).
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "performance_metrics.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key, val in metrics.items():
            writer.writerow([key, float(val)])

    h5_path = output_dir / "performance_metrics.h5"
    with h5py.File(h5_path, "w") as h5:
        for key, val in metrics.items():
            ds = h5.create_dataset(key, data=[float(val)])
            # ``h5py_stub`` stores raw memoryviews which are not picklable. Replace
            # them with plain lists when possible so that the file can be closed
            # without error.
            try:  # pragma: no cover - defensive for real h5py
                ds.data = [float(val)]  # type: ignore[attr-defined]
            except Exception:
                pass

    md_lines = ["| KPI | Value |", "| --- | --- |"]
    for key, val in metrics.items():
        md_lines.append(f"| {key} | {val} |")
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    if plt is None:  # pragma: no cover - optional dependency
        return

    keys = list(metrics.keys())
    values = [metrics[k] for k in keys]
    plt.figure()
    plt.bar(keys, values)
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_metrics.png")
    plt.close()
