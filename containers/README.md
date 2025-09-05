# DPF2 Singularity Container

This directory contains the Singularity/Apptainer definition for the DPF2 project.
Build the image using:

```bash
singularity build dpf2.sif dpf2.def
```

The definition installs system build tools and Python dependencies from `requirements.txt`.
