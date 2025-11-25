# DPF2 Singularity Container

This directory contains the Singularity/Apptainer definition for the DPF2 project.
Build the image using:

```bash
singularity build dpf2.sif dpf2.def
```

The definition installs system build tools and Python dependencies from `requirements.txt`.

## Running on HPC / at scale

The resulting `dpf2.sif` image is portable to common schedulers.  For quick
multi-node sweeps:

```bash
mpirun -np 8 singularity exec dpf2.sif dpf2 batch pipeline \
  --config my_config.json --parameter charging_voltage --linspace 12e3:18e3:5
```

For Slurm clusters, enqueue a job with CPU and GPU variants using the same
container:

```bash
sbatch --ntasks=16 --gres=gpu:1 <<'EOF'
#!/bin/bash
module load singularity
singularity exec --nv dpf2.sif dpf2 simulate --config my_config.json --output out_gpu
EOF
```

See `SCALING.md` in this directory for tips on MPI ranks, threading, and batch
orchestrations using the new CLI pipeline helper.
