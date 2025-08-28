"""Run parameter sweeps for the DPF simulation engine.

The script distributes independent runs across MPI ranks when
:mod:`mpi4py` is available.  Each rank evaluates a subset of the
parameter values and stores results as ``.npz`` files containing the time
and current arrays.  The script focuses on circuit parameters but can be
extended for other configuration fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from dpf2.dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine

try:  # pragma: no cover - MPI optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Base configuration file")
    parser.add_argument("--param", type=str, required=True, help="Name of circuit parameter to vary")
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        required=True,
        help="List of parameter values to explore",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("sweep_results"),
        help="Directory where result files are written",
    )
    args = parser.parse_args()

    cfg = DPFConfig.from_file(args.config)
    args.outdir.mkdir(exist_ok=True)

    comm = MPI.COMM_WORLD if MPI else None
    rank = comm.Get_rank() if comm else 0
    size = comm.size if comm else 1

    values: Sequence[float] = args.values
    for i, val in enumerate(values):
        if i % size != rank:
            continue
        setattr(cfg.circuit_config, args.param, val)
        engine = SimulationEngine(cfg, comm=None)
        res = engine.run()
        out = args.outdir / f"{args.param}_{val:.3g}.npz"
        np.savez(out, time=res.time, current=res.current)

    if comm:
        comm.Barrier()


if __name__ == "__main__":  # pragma: no cover
    main()
