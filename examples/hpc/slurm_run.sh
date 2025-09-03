#!/bin/bash
# Initial run of a DPF2 simulation on a SLURM cluster
#SBATCH -J dpf2_run
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Stage input data
tmp=${SLURM_TMPDIR:-$(mktemp -d)}
cp $SLURM_SUBMIT_DIR/examples/config.json "$tmp/"
cd "$tmp"

# Execute the simulation
srun python $SLURM_SUBMIT_DIR/examples/run_simulation.py --config config.json --output result.npz

# Collect output data
mkdir -p $SLURM_SUBMIT_DIR/results
cp result.npz $SLURM_SUBMIT_DIR/results/
