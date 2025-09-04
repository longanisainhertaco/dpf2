# HPC Job Submission Examples

This directory contains small templates for running the DPF2 example
simulation on a SLURM cluster. The scripts showcase how the
``JobManager`` can stage input data, control GPU placement and restart a
simulation from a checkpoint. The examples also showcase the ``--lab-mode``
flag which records a manifest of code hash, random seeds, configuration
paths and particle-per-cell settings for reproducibility.

* ``slurm_run.sh`` – stages the configuration file into a temporary
  directory and launches ``dpf2 simulate`` with ``--lab-mode`` using
  ``srun``. The ``CUDA_VISIBLE_DEVICES`` environment variable may be set by
  the :class:`~dpf2.hpc.JobManager` to pin the job to specific GPUs. A
  ``manifest.json`` and ``manifest.h5`` files are written inside the run
  directory capturing reproducibility metadata.

* ``slurm_restart.sh`` – similar to ``slurm_run.sh`` but demonstrates how
  to restart from a checkpoint while still capturing a manifest with
  ``--lab-mode``. The script checks the ``DPF_RESTART`` environment
  variable, which is automatically exported by
  :meth:`JobManager.restart <dpf2.hpc.manager.JobManager.restart>`.

These templates are intentionally minimal and are meant to be adapted to
local cluster policies.

