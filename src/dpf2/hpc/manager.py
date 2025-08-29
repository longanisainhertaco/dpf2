"""Simple job manager for scheduling simulations on HPC resources.

This module provides a :class:`JobManager` that abstracts submission of
simulation runs to different schedulers such as MPI via ``mpirun``,
SLURM clusters and AWS Batch.  The implementation intentionally focuses on
small scale examples and does not aim to expose the full feature set of the
underlying schedulers.
"""
from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Any, Dict


@dataclass
class JobManager:
    """Dispatch simulation jobs to different backends."""

    scheduler: str = "slurm"

    def submit(self, job_script: str, **kwargs: Any) -> Any:
        """Submit ``job_script`` to the configured scheduler.

        Parameters
        ----------
        job_script:
            Path to a submission script or executable.
        **kwargs:
            Additional scheduler specific keyword arguments.

        Returns
        -------
        Any
            Scheduler specific return object.
        """

        if self.scheduler == "slurm":
            cmd = ["sbatch", job_script]
            return subprocess.run(cmd, capture_output=True, text=True, check=False)
        if self.scheduler == "awsbatch":
            try:
                import boto3  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("boto3 is required for AWS Batch submissions") from exc
            batch = boto3.client("batch")
            params: Dict[str, Any] = {
                "jobName": kwargs.get("job_name", "dpf2-job"),
                "jobQueue": kwargs["job_queue"],
                "jobDefinition": kwargs["job_definition"],
            }
            if "parameters" in kwargs:
                params["parameters"] = kwargs["parameters"]
            return batch.submit_job(**params)
        if self.scheduler == "mpi":
            cmd = ["mpirun", job_script]
            return subprocess.run(cmd, capture_output=True, text=True, check=False)
        raise ValueError(f"Unsupported scheduler: {self.scheduler}")


__all__ = ["JobManager"]
