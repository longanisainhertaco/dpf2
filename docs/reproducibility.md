# Reproducibility Workflow

DPF2 provides tooling to capture provenance for each simulation run.

## CLI Metadata

Running simulations through the command line interface writes metadata into
every generated HDF5 output.  Each file records:

- the git commit hash of the codebase at run time,
- the full configuration used for the run, and
- random number generator seeds used by Python and NumPy.

This information allows exact reproduction of results from a single HDF5
output file.

## Run Manifests

When executing in lab mode or other batch-oriented commands, a manifest is
stored in each run directory.  The manifest is available as both
`run_manifest.json` and `run_manifest.h5` and mirrors the metadata embedded in the
HDF5 outputs.

Manifests include paths to configuration files, computed seeds and additional
provenance details such as maximum particles-per-cell or any warnings
encountered during the run.

These files enable auditing of large parameter sweeps or shot ensembles.

