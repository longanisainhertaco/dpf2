# HPC Usage

DPF2 provides a minimal ``JobManager`` to help launch simulations on
clusters.  When running on shared systems, enable the ``--lab-mode`` flag
to capture metadata for reproducibility and auditing.

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
alongside ``manifest.json`` (and ``manifest.h5`` when ``h5py`` is installed).
The manifest records the git commit, random seeds and particles-per-cell setting,
allowing each run to be audited and reproduced.

