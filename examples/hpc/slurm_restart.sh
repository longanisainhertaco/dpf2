#!/bin/bash
# Restart a DPF2 simulation using a checkpoint with data staging
# Submit with a dependency on the initial job:
#   sbatch --dependency=afterok:<jobid> slurm_restart.sh
#SBATCH -J dpf2_restart
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Stage checkpoint and configuration into scratch
tmp=${SLURM_TMPDIR:-$(mktemp -d)}
cp $SLURM_SUBMIT_DIR/examples/config.json "$tmp/"
cp $SLURM_SUBMIT_DIR/results/result.npz "$tmp/"
cd "$tmp"

# Restart the simulation from the checkpoint
echo "Restarting from result.npz"
srun python $SLURM_SUBMIT_DIR/examples/run_simulation.py --config config.json --restart result.npz --output restart_result.npz

# Collect output data back to the submission directory
mkdir -p $SLURM_SUBMIT_DIR/results
cp restart_result.npz $SLURM_SUBMIT_DIR/results/
