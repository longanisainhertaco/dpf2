"""Command line utilities for running uncertainty quantification analyses."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from ..uq.analysis import sobol_indices
from ..uq.calibration import bayes_factor, posterior_summary
from ..uq import calibrate_waveform


def _load_waveform(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"waveform file {path!s} must have at least two columns")
    return arr[:, 0], arr[:, 1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Calibration sub-command -------------------------------------------------
    cal = sub.add_parser("calibrate", help="Calibrate waveform scaling factors")
    cal.add_argument("sim", help="CSV file with simulated time,current waveform")
    cal.add_argument("data", help="CSV file with measured time,current waveform")
    cal.add_argument(
        "--method",
        choices=["emcee", "dynesty"],
        default="emcee",
        help="Calibration backend to use",
    )
    cal.add_argument(
        "--output",
        default="uq_results.npz",
        help="Path to store or read calibration samples",
    )

    # Summary sub-command -----------------------------------------------------
    summ = sub.add_parser("summary", help="Summarise posterior samples from an NPZ file")
    summ.add_argument("file", help="NPZ file produced by the calibrate command")

    # Bayes factor sub-command -----------------------------------------------
    bf = sub.add_parser("bayes", help="Compute Bayes factor from two log-evidences")
    bf.add_argument("logz1", type=float)
    bf.add_argument("logz2", type=float)

    # Sensitivity sub-command -------------------------------------------------
    sens = sub.add_parser("sensitivity", help="Compute Sobol indices from arrays")
    sens.add_argument("samples", help="NPZ file containing 'samples' and 'values'")
    sens.add_argument("--names", nargs="+", required=True, help="Parameter names")

    args = parser.parse_args(argv)

    if args.cmd == "calibrate":
        out_path = Path(args.output)
        t_sim, i_sim = _load_waveform(args.sim)
        t_data, i_data = _load_waveform(args.data)
        samples = calibrate_waveform(t_sim, i_sim, t_data, i_data, method=args.method)
        np.savez(out_path, **samples)
        print(f"Saved results to {out_path}")
        return 0

    if args.cmd == "summary":
        data = np.load(args.file)
        samples = {name: data[name] for name in data.files}
        stats = posterior_summary(samples)
        for name, s in stats.items():
            print(f"{name}: mean={s['mean']:.3f} std={s['std']:.3f}")
        return 0

    if args.cmd == "bayes":
        bf_val = bayes_factor(args.logz1, args.logz2)
        print(f"Bayes factor: {bf_val:.3f}")
        return 0

    if args.cmd == "sensitivity":
        data = np.load(args.samples)
        samples = data["samples"]
        values = data["values"]
        indices = sobol_indices(samples, values, args.names)
        for name, val in indices.items():
            print(f"{name}: {val:.3f}")
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

