# HPC Usage

DPF2 provides a minimal ``JobManager`` to help launch simulations on
clusters.  When running on shared systems, enable the ``--lab-mode`` flag
to capture metadata for reproducibility and auditing.

The job manager automatically stages the ``run_manifest.json`` file with
other outputs.  Passing the manifest path via ``restart`` allows a job to
resume from recorded metadata.

## Example SLURM batch script

```bash
#!/bin/bash
#SBATCH -J dpf2_example
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 00:05:00
#SBATCH -o run/%x-%j.log

module load python

# Execute the simulation and record a manifest with git hash and RNG seeds
# The manifest is written next to other outputs in the run directory
python -m dpf2.cli --lab-mode simulate --config config.json --output run
```

After the job completes the ``run`` directory will contain simulation outputs
alongside ``run_manifest.json`` (and ``run_manifest.h5`` when ``h5py`` is installed).
The manifest records the git commit, random seeds, environment details and
particles-per-cell setting, allowing each run to be audited and reproduced.

## Restarting from a manifest

To resume a previous run provide the path to the existing manifest via
``--restart``. The job manager will stage the manifest automatically.

```bash
#!/bin/bash
#SBATCH -J dpf2_restart
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 00:05:00
#SBATCH -o run2/%x-%j.log

module load python

# Restart from a recorded manifest and write outputs to a new directory
python -m dpf2.cli --lab-mode simulate \
    --config config.json \
    --output run2 \
    --restart run/run_manifest.json
```

