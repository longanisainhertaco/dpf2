# Lab-mode UI quickstart

This directory documents the lab-mode controls surfaced in the web UI. The
panels are designed for rapid, shot-to-shot experimentation while keeping
project assets organised.

## Shot-to-shot jitter panel

* **Enable Jitter** – toggles stochastic sampling for pressure and trigger
  timing before each shot.
* **Switch Jitter (ns)** – offsets the trigger timing; positive values delay the
  discharge, negative values (where allowed) pull the trigger earlier.
* **Pressure Jitter (%)** – scales the fill pressure for each shot. Use small
  percentages to mimic day-to-day lab drift; larger values for stress tests.
* **Gas Puff Timing (ns)** – coordinates the gas puff relative to the switch,
  mirroring the `gas_puff_timing_ns` field in configurations.

Each shot emits a `run_manifest.json` capturing seeds, jitter samples, and the
container hash when lab mode is enabled server-side.

## Project management cards

* Import/export configuration sets as JSON to compare sweeps or archive
  favourite setups.
* Use the selection checkboxes to overlay yield/efficiency curves and annotate
  diagnostics notes per project.
* Comparison cards surface the best-performing shot and the parameter value
  that achieved it, keeping optimisation notes next to the physics visuals.

## Reproducibility tips

* Always download the manifest bundle after running a batch; it contains
  `bundle_manifest.json` plus individual run manifests for offline analysis.
* Pair UI experiments with CLI sweeps (`dpf2 batch sweep ...`) to validate
  trends. Both paths now include wall-plug efficiency and yield-per-hour in the
  computed metrics.
