# HPC Deployment

This guide outlines how to build the Singularity container used for DPF2
simulations and how to submit jobs to a cluster using the bundled utilities.

## Building the container

The project ships with a `Singularity.def` definition that pins all runtime
dependencies and installs an entrypoint script which records the current git
commit and environment details.  Build the image with:

```bash
singularity build -F dpf2.sif infrastructure/Singularity.def
```

Running the resulting `dpf2.sif` image will print the code hash and platform
information before executing the `dpf2` CLI.

## Submitting jobs

Use the :class:`dpf2.hpc.JobManager` helper to launch batch jobs on the
cluster.  Each submission creates an `run_manifest.h5` file containing the
code version, run configuration and container hash for reproducibility.

```python
from dpf2.hpc import JobManager

jm = JobManager("slurm")
jm.submit(
    "run.sh",
    config={"shot": 1},
    container_hash="sha256:1234...",
)
```

The manifest is staged with other outputs and may be used to audit or reproduce
simulation runs on the cluster.

