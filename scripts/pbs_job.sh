#!/bin/bash
#PBS -N dpf2-bench
#PBS -l select=1:ncpus=4:mpiprocs=4
#PBS -l walltime=00:05:00
#PBS -j oe

module load singularity
mpirun singularity exec containers/dpf2.sif python scripts/run_benchmarks.py
