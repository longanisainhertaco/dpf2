# Reproducibility Workflow

DPF2 records detailed provenance for every run. When CLI commands are invoked
with the `--lab-mode` flag, a `run_manifest.json` is written alongside the
outputs. The manifest captures:

- git commit hash of the code
- full configuration used for the run
- SHA256 hash of the configuration inputs
- random number generator seeds for Python and NumPy
- optional diagnostic summaries for each execution
- basic environment information such as Python version, platform and exported
  environment variables

These details allow results to be precisely reproduced even when runs are moved
between systems.

The recorded metadata follows the FAIR and FAIR4RS principles, ensuring that
results are Findable, Accessible, Interoperable and Reusable by embedding
machine-readable hashes, seeds and diagnostics directly within each
``run_manifest.json``.

When launching jobs through :class:`dpf2.hpc.JobManager`, provide the manifest
path via the ``manifest`` argument. The manager will stage the manifest file
with other outputs and accepts a manifest path in ``restart`` to resume a run
using recorded metadata.

