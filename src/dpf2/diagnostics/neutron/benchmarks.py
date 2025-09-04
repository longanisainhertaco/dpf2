"""Benchmark comparison helpers for neutron diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import json


def _load_expected(benchmark: str) -> Tuple[float, float]:
    """Return (expected_yield, tolerance) for *benchmark*."""

    base = (
        Path(__file__).resolve().parents[4]
        / "benchmarks"
        / benchmark
        / "expected.json"
    )
    with base.open("r", encoding="utf8") as fh:
        data = json.load(fh)
    exp = float(data.get("neutron_yield", [0.0, 0.0])[-1])
    tol = float(data.get("tolerance", {}).get("neutron_yield", 0.0))
    return exp, tol


def compare_with_benchmark(
    yield_value: float, benchmark: str, pass_band: float | None = None
) -> Tuple[bool, float]:
    """Compare ``yield_value`` against reference data for *benchmark*.

    Parameters
    ----------
    yield_value:
        Simulated neutron yield to be compared.
    benchmark:
        Name of the benchmark directory under ``benchmarks/`` (e.g., ``pf_1000``
        or ``mjolnir``).
    pass_band:
        Optional absolute tolerance overriding the value stored in the
        benchmark file.

    Returns
    -------
    ``(passed, difference)`` where ``passed`` indicates whether the simulated
    yield lies within ``±pass_band`` of the reference value and ``difference`` is
    the signed difference ``yield_value - reference``.
    """

    expected, tol = _load_expected(benchmark)
    if pass_band is not None:
        tol = float(pass_band)
    diff = float(yield_value) - expected
    return (abs(diff) <= tol, diff)


__all__ = ["compare_with_benchmark"]
