# Reference Traces

This directory contains example, normalized diagnostic traces for several dense plasma focus devices:

- **PF1000** (Institute of Plasma Physics and Laser Microfusion, Poland)
- **NX2** (Nanyang Technological University, Singapore)
- **UNU** (University of New South Wales / IAEA UNU/ICTP PFF)

A machine-readable overview of the available shots is provided in `dataset_manifest.json`. Consumers can access this information programmatically via the `load_dataset_manifest` utility in `dpf2.io.datasets`.

The data are illustrative and derived from publicly available descriptions of these experiments. Values have been normalized and do not represent exact experimental measurements.

> **Note**
> Real HDF5 traces are not distributed in this repository. Files ending in `.h5` are plain-text placeholders included so downstream code can locate expected paths. Replace them with actual data when running locally.

## Licensing

These files are provided for educational and testing purposes. When using data from the referenced facilities in research or publications, consult the original sources and respect any licensing or data usage restrictions that may apply.
