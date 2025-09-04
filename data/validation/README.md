# Validation Datasets

This directory contains small example datasets used by the validation suite.
Each dataset lives in its own subdirectory and follows the same layout:

* `current.csv` – discharge current waveform
* `voltage.csv` – bank voltage waveform
* `neutron_yield.csv` – time‐resolved neutron yield
* `scaling.json` – simple scaling law parameters

The CSV files are two columns with headers `time,value` where time is in
microseconds and values are in arbitrary units scaled for the examples.  The
`scaling.json` files contain coefficients used by tests of empirical scaling
laws.

Two devices are bundled:

* **PF1000** – representative of the PF‑1000 plasma focus experiment.
* **MJOLNIR** – representative of the MJOLNIR device at LLNL.

These datasets are intentionally minimal and are intended only for unit tests
and examples; they are not full experimental records.
