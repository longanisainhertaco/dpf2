"""Launch parameter sweep jobs using :class:`~dpf2.hpc.JobManager`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from dpf2.hpc import JobManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Base configuration file"
    )
    parser.add_argument(
        "--param", type=str, required=True, help="Parameter name to vary"
    )
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        required=True,
        help="List of parameter values",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="mpi",
        choices=["mpi", "slurm", "awsbatch"],
        help="Job scheduler backend",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("sweep_results"),
        help="Directory for outputs",
    )
    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True)
    jm = JobManager(args.scheduler)

    job_files: list[Path] = []
    for val in args.values:
        script = args.outdir / f"job_{args.param}_{val:.3g}.sh"
        cmd = (
            f"python scripts/parameter_sweep.py --config {args.config} --param {args.param} "
            f"--values {val} --outdir {args.outdir}"
        )
        script.write_text(f"#!/bin/bash\n{cmd}\n")
        script.chmod(0o755)
        jm.submit(str(script))
        job_files.append(args.outdir / f"{args.param}_{val:.3g}.npz")

    for res in job_files:
        if res.exists():
            data = np.load(res)
            print(f"{res.stem}: peak current {data['current'].max():.3e} A")


if __name__ == "__main__":  # pragma: no cover
    main()
