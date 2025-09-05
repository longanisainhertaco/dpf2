"""Utilities for neutron benchmark reference data and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_REFERENCE_DIRS = {
    "pf-1000": "PF-1000",
    "pf1000": "PF-1000",
    "mjolnir": "MJOLNIR",
}


def _data_path(name: str) -> Path:
    """Return the path to the reference data file for ``name``."""
    root = Path(__file__).resolve().parents[2]
    directory = _REFERENCE_DIRS.get(name.lower())
    if directory is None:
        raise ValueError(f"Unknown benchmark '{name}'")
    return root / "benchmarks" / directory / "reference.csv"


def load_reference(name: str) -> pd.DataFrame:
    """Load reference current trace for a benchmark device."""
    path = _data_path(name)
    return pd.read_csv(path)


def load_pf1000_reference() -> pd.DataFrame:
    """Load PF-1000 reference data."""
    return load_reference("pf-1000")


def load_mjolnir_reference() -> pd.DataFrame:
    """Load MJOLNIR reference data."""
    return load_reference("mjolnir")


def within_pass_band(
    data: Iterable[float],
    reference: Iterable[float],
    band: float | Iterable[float],
) -> np.ndarray:
    """Return boolean array indicating whether points lie within band.

    Parameters
    ----------
    data, reference:
        Measured and reference values.
    band:
        Relative pass band expressed as a fraction. If a scalar is provided
        the same band is applied to all points. When a reference value is
        zero the band is interpreted as an absolute tolerance.
    """
    data_arr = np.asarray(data, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    band_arr = np.asarray(band, dtype=float)
    if band_arr.size == 1:
        band_arr = np.full_like(ref_arr, band_arr)
    tol = band_arr * np.where(ref_arr != 0, np.abs(ref_arr), 1.0)
    return np.abs(data_arr - ref_arr) <= tol


def evaluate_pass_fail(
    data: Iterable[float],
    reference: Iterable[float],
    band: float | Iterable[float],
) -> bool:
    """Return ``True`` if all points fall within the pass band."""
    return bool(np.all(within_pass_band(data, reference, band)))
