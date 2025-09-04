#!/bin/bash
# Restart a DPF2 simulation using a checkpoint with data staging
# Submit with a dependency on the initial job:
#   sbatch --dependency=afterok:<jobid> slurm_restart.sh
#SBATCH -J dpf2_restart
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Stage checkpoint and configuration into scratch. If ``DPF_RESTART`` is set
# (for example via :class:`~dpf2.hpc.JobManager.restart`), use that path,
# otherwise fall back to the previous run's output.
tmp=${SLURM_TMPDIR:-$(mktemp -d)}
cp $SLURM_SUBMIT_DIR/examples/config.json "$tmp/"
checkpoint=${DPF_RESTART:-$SLURM_SUBMIT_DIR/results/result.npz}
cp "$checkpoint" "$tmp/"
cd "$tmp"

# Restart the simulation from the checkpoint
chk_file=$(basename "$checkpoint")
echo "Restarting from $chk_file"
srun python $SLURM_SUBMIT_DIR/examples/run_simulation.py --config config.json --restart "$chk_file" --output restart_result.npz

# Collect output data back to the submission directory
mkdir -p $SLURM_SUBMIT_DIR/results
cp restart_result.npz $SLURM_SUBMIT_DIR/results/
