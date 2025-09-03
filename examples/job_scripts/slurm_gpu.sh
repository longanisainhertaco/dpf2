#!/bin/bash
# Example SLURM job script for a GPU-enabled cluster
#SBATCH -J dpf2_gpu_example
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=1
#SBATCH -o dpf2_gpu_%j.log

module load python cuda mpi
srun python examples/run_simulation.py --config examples/config.json --use-gpu
