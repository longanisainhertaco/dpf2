# Performance Environment

This document records the build and runtime settings used for scalability
measurements and parallel I/O benchmarks.

## Build Settings

| Component | Setting |
|-----------|--------|
| Compile flags | `-O3 -march=native -ffast-math` |
| MPI | OpenMPI 4.1 with `OMPI_MCA_btl=^openib` |
| HDF5 | 1.14 built with `--enable-parallel`, runtime `HDF5_USE_FILE_LOCKING=FALSE` |

## Container Images

Singularity definition files are provided in `infrastructure/singularity/`.
The SHA256 digest of each definition file is recorded below.

| Image | Definition | SHA256 |
|-------|------------|--------|
| Base | `infrastructure/singularity/base.def` | ce9fb3a3d1adf976dbf73df849af8732cf8d5f2ab5cf4d504031a20947bd9fe4 |
| MPI | `infrastructure/singularity/mpi.def` | 679199e3fb9eb91422e32a00285a9151292e529660efa91de71e5f3ae86a633c |

These hashes can be used to verify the container images produced in the CI
pipeline.
