"""Generate a simple batch submission script for parameter sweeps."""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Configuration file used by the sweep")
    parser.add_argument("--param", required=True, help="Parameter name passed to parameter_sweep.py")
    parser.add_argument(
        "--values", nargs="+", required=True, help="Values supplied to the sweep script"
    )
    parser.add_argument("--outfile", type=Path, default=Path("submit.sh"), help="Output script path")
    parser.add_argument("--nprocs", type=int, default=1, help="Number of MPI ranks")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=0, help="GPUs per node")
    parser.add_argument("--dependency", help="SLURM dependency specification")
    parser.add_argument("--job-name", default="dpf2-sweep", help="Job name")
    parser.add_argument("--output", default="slurm-%j.out", help="SLURM output file")
    parser.add_argument("--stage", help="Directory to stage results after completion")
    args = parser.parse_args()

    cmd = (
        f"python scripts/parameter_sweep.py --config {args.config} --param {args.param} "
        f"--values {' '.join(args.values)}"
    )

    script = textwrap.dedent(
        f"""
        #!/bin/bash
        #SBATCH -J {args.job_name}
        #SBATCH -N {args.nodes}
        #SBATCH -n {args.nprocs}
        #SBATCH -o {args.output}
        """
    )
    if args.gpus:
        script += f"#SBATCH --gpus={args.gpus}\n"
    if args.dependency:
        script += f"#SBATCH --dependency={args.dependency}\n"
    script += f"\n{cmd}\n"
    if args.stage:
        script += textwrap.dedent(
            f"""
            mkdir -p {args.stage}
            rsync -a results/ {args.stage}/
            """
        )
    args.outfile.write_text(script)
    print(f"Wrote batch script to {args.outfile}")


if __name__ == "__main__":  # pragma: no cover
    main()
