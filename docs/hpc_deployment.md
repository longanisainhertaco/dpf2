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
cluster.  Each submission creates a `run_manifest.h5` file capturing the git
commit, run configuration and container image hash.  When no hash is supplied
explicitly, the manager looks for a `SINGULARITY_CONTAINER` environment variable
and records the SHA256 digest of that image for later auditing.

The manifest is created regardless of whether configuration details or a
container hash are provided so that every run leaves an auditable record of the
code revision and runtime environment.

```python
from dpf2.hpc import JobManager

jm = JobManager("slurm")
jm.submit(
    "run.sh",
    config={"shot": 1},
    # container_hash="sha256:1234...",  # optional
)
```

The manifest is staged with other outputs and may be inspected with tools like
`h5dump run_manifest.h5` or from Python using `h5py` to reproduce and audit
simulation runs on the cluster.

