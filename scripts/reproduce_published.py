"""Reproduce published device benchmarks using bundled data.

This helper script loads a DPF configuration, runs the
:class:`~dpf2.simulation_engine.SimulationEngine`, and overlays the
resulting observables with experimental traces. Validation scores are
printed to standard output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dpf2.cli.validate import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to DPF configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PF1000",
        help="Dataset identifier (default: PF1000)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("validation"),
        help="Directory to store overlay plots",
    )
    args = parser.parse_args()
    run_validation(args.config, args.dataset, outdir=args.outdir)


if __name__ == "__main__":  # pragma: no cover
    main()

