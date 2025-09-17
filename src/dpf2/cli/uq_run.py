"""Command line interface for running UQ calibration routines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from ..uq import calibrate_waveform


def _load_waveform(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load two-column ``time,current`` waveform data."""
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"waveform file {path!s} must have at least two columns")
    return arr[:, 0], arr[:, 1]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``uq_run`` CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim", help="CSV file with simulated time,current waveform")
    parser.add_argument("data", help="CSV file with measured time,current waveform")
    parser.add_argument(
        "--method",
        choices=["emcee", "dynesty"],
        default="emcee",
        help="Calibration backend to use",
    )
    parser.add_argument(
        "--output",
        default="uq_results.npz",
        help="Path to store or read calibration samples",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Summarise an existing results file instead of running calibration",
    )

    args = parser.parse_args(argv)
    out_path = Path(args.output)

    if args.summary:
        if not out_path.exists():
            raise SystemExit(f"results file {out_path} does not exist")
        data = np.load(out_path)
        for name in data.files:
            arr = data[name]
            print(f"{name}: mean={arr.mean():.3f} std={arr.std():.3f}")
        return 0

    t_sim, i_sim = _load_waveform(args.sim)
    t_data, i_data = _load_waveform(args.data)
    samples = calibrate_waveform(t_sim, i_sim, t_data, i_data, method=args.method)
    np.savez(out_path, **samples)
    print(f"Saved results to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
