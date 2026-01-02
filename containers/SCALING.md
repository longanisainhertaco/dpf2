# Containerised scaling guidance

This document captures the most common patterns for running DPF2 in
containerised HPC environments.

## Selecting a launcher

* **MPI/Singularity**: `mpirun -np <ranks> singularity exec dpf2.sif dpf2 batch pipeline ...`
* **Slurm**: use `sbatch`/`srun` with `singularity exec` inside the script.
* **Kubernetes**: wrap the same command in a `Job` and mount a persistent
  volume for outputs.

## Threading and GPU utilisation

* Set `OMP_NUM_THREADS` to the number of cores per rank to avoid oversubscription.
* Use `--gres=gpu:N` with Slurm or `--nv` with Singularity to expose GPUs.
* The CLI `batch pipeline` command emits manifests suitable for aggregating
  results from multiple ranks or pods.

## Parametric sweeps at scale

1. Build the container locally or on a login node: `singularity build dpf2.sif dpf2.def`.
2. Launch a coarse sweep: `mpirun -np 16 singularity exec dpf2.sif dpf2 batch sweep --parameter charging_voltage --linspace 12e3:18e3:8`.
3. Refine near the optimum with `dpf2 batch pipeline --parameter charging_voltage --linspace 14e3:16e3:5`.

## Optimisation and resilience

* Use checkpoint emitting (`--emit-checkpoints`) to make long optimisations restartable.
* Persist manifests on a shared filesystem so that downstream analytics can pick up partial sweeps.
* Prefer immutable container tags for reproducibility across clusters.

## Provenance for scaling artefacts

Running `python scripts/bench_scaling.py --max-procs 8 --outdir scaling_results` now
writes a `run_manifest.json` that records the container hash, git revision, MPI/HDF5
versions, and SHA256 hashes of the generated scaling plots. Attach a DOI to the
artefacts with `--artifact-doi` to make the performance envelope citeable when
publishing container images or benchmark suites.
