#!/bin/bash
#SBATCH --job-name=dpf2-bench
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out

module load singularity
srun singularity exec containers/dpf2.sif python scripts/run_benchmarks.py
