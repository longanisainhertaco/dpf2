# Validation Workflow

This document describes how to validate the synthetic diagnostics in this
repository. Benchmark definitions and expected diagnostic outputs are stored in
`tests/benchmarks`.

## Running the Benchmarks

1. Install the package dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the benchmark tests:
   ```bash
   pytest tests/benchmarks/test_diagnostic_baselines.py
   ```
   The tests compute neutron yield, X-ray spectra, and scope traces for the
   provided cases and compare the results against the stored baselines. These
   tests run in continuous integration to ensure diagnostic calculations remain
   consistent with the reference outputs.

## Updating Baselines

To update or add a benchmark, edit or create the JSON files in
`tests/benchmarks` and ensure the expected outputs match the new results. Commit
these baseline files together with any code changes.
