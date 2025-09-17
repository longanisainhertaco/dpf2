#!/usr/bin/env python3
"""Compute simple KPIs and export them for the web UI.

This command line utility wraps :func:`dpf2.diagnostics.compute_performance_metrics`
and writes summary tables, plots and data files into the specified output
directory.  The default location matches the layout expected by the web UI
(``ui/performance_metrics``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dpf2.diagnostics import (
    compute_performance_metrics,
    export_performance_metrics,
)


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yield_per_shot", type=float, help="Neutron yield per shot")
    parser.add_argument("rep_rate_hz", type=float, help="Repetition rate in Hz")
    parser.add_argument("energy_out_j", type=float, help="Output energy per shot (J)")
    parser.add_argument("energy_in_j", type=float, help="Input energy per shot (J)")
    parser.add_argument(
        "electrode_mass_g",
        type=float,
        help="Available electrode mass before replacement (g)",
    )
    parser.add_argument(
        "erosion_per_shot_g", type=float, help="Electrode mass lost per shot (g)"
    )
    parser.add_argument(
        "--output",
        default="ui/performance_metrics",
        type=Path,
        help="Directory where output files will be written",
    )
    args = parser.parse_args()

    metrics = compute_performance_metrics(
        args.yield_per_shot,
        rep_rate_hz=args.rep_rate_hz,
        energy_out_j=args.energy_out_j,
        energy_in_j=args.energy_in_j,
        electrode_mass_g=args.electrode_mass_g,
        erosion_per_shot_g=args.erosion_per_shot_g,
    )
    export_performance_metrics(metrics, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
