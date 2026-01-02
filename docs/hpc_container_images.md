# Containerised HPC images and scaling benchmarks

We publish two container targets to keep solver performance reproducible on
clusters:

* **`containers/hpc-mpi.Dockerfile`** – built with OpenMPI, UCX and HDF5, tuned
  for multi-node launches under Slurm. Each image advertises its git hash via
  the `CONTAINER_HASH` label so manifests record exactly which bits were run.
* **`containers/hpc-gpu.Dockerfile`** – adds CUDA-aware MPI and NCCL bindings for
  GPU partitions. The image ships the benchmarking harness under
  `scripts/run_scaling_suite.py`.

### Running benchmarks

1. Build or pull the image then launch the scaling sweep:
   ```bash
   srun -n 4 --mpi=pmix_v4 ./scripts/run_scaling_suite.py --profile mpi --output scaling_results
   ```
2. The suite writes wall-clock timings, solver throughput, and memory footprint
   into `scaling_results/summary.json` plus a `run_manifest.json` that captures
   code hash, container hash, RNG seeds, CPU/GPU model and MPI/HDF5 versions.
3. Plots are emitted to `docs/images/scaling` so they surface automatically on
   the performance dashboard.
4. When invoked with `--artifact-doi`, the manifest also stores SHA256 hashes
   of `strong.png`, `weak.png`, and `hdf5_io.png` alongside the DOI for the
   published benchmark artefacts.

### Reproducibility metadata

Every batch run appends its manifest to `scaling_results/batch_manifest.json` so
campaigns can be re-launched verbatim. Include relevant datasets using the
`--datasets` flag to record DOIs and SHA256 hashes alongside the timings.

### Deployment steps

* **Slurm + CPUs:**
  ```bash
  srun -n 8 --ntasks-per-node=4 docker://ghcr.io/dpf2/hpc-mpi:latest dpf2 simulate -c config.json -o out --lab-mode
  ```
* **Slurm + GPUs:**
  ```bash
  srun -n 4 --gpus-per-task=1 docker://ghcr.io/dpf2/hpc-gpu:latest dpf2 simulate -c config.json -o out --lab-mode --emit-openpmd
  ```
* **Kubernetes:** wrap the image with the Helm chart in `containers/charts/dpf2`
  and set `labMode=true` to force manifest logging for every driver pod.

For end-to-end provenance, pair these runs with the new `dpf2 project sweep`
command to orchestrate parameter studies while logging manifests at each sweep
point.
