#!/bin/bash
# Example SLURM job script for a CPU-only cluster
#SBATCH -J dpf2_cpu_example
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o dpf2_cpu_%j.log

module load python mpi
srun python examples/run_simulation.py --config examples/config.json
