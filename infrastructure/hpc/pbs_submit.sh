#!/bin/bash
#PBS -N dpf2_scaling
#PBS -l select=1:ncpus=4
#PBS -l walltime=00:10:00
#PBS -o $PBS_O_WORKDIR/scaling_$PBS_JOBID.log

module load apptainer

cd "$PBS_O_WORKDIR"
# Run the scaling benchmark inside the container
apptainer run dpf2.sif python /opt/dpf2/scripts/benchmark_scaling.py --max-workers 4 --outdir results
