#!/bin/bash
#SBATCH -J dpf2_scaling
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH -o scaling_%j.log

module load apptainer

# Run the scaling benchmark inside the container
apptainer run dpf2.sif python /opt/dpf2/scripts/benchmark_scaling.py --max-workers 4 --outdir results
