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
from typing import Any, Dict, Iterable


@dataclass
class JobManager:
    """Dispatch simulation jobs to different backends."""

    scheduler: str = "slurm"

    def _extend_cmd(self, cmd: list[str], opts: Dict[str, Any], flag_map: Dict[str, Iterable[str]]) -> None:
        """Append CLI options from ``opts`` to ``cmd`` using ``flag_map``.

        Parameters
        ----------
        cmd:
            Mutable command list to extend.
        opts:
            User supplied options.
        flag_map:
            Mapping of option keys to one or more flags understood by the
            underlying scheduler command.
        """

        for key, flags in flag_map.items():
            if key in opts and opts[key] is not None:
                value = str(opts[key])
                if isinstance(flags, str):
                    flags = [flags]
                cmd.extend(list(flags) + [value])

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
            cmd = ["sbatch"]
            self._extend_cmd(
                cmd,
                kwargs,
                {
                    "nodes": "-N",
                    "gpus": "--gpus",
                    "dependency": "--dependency",
                    "output": "-o",
                    "error": "-e",
                },
            )
            cmd.append(job_script)
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
            cmd = ["mpirun"]
            self._extend_cmd(
                cmd,
                kwargs,
                {
                    "nprocs": "-n",
                },
            )
            # Domain decomposition parameters, passed as --decomp-x 2 etc.
            decomp: Dict[str, Any] | None = kwargs.get("decomp")
            if decomp:
                for axis, cnt in decomp.items():
                    cmd.extend([f"--decomp-{axis}", str(cnt)])
            if "restart" in kwargs:
                cmd.extend(["--restart", str(kwargs["restart"])])
            cmd.append(job_script)
            return subprocess.run(cmd, capture_output=True, text=True, check=False)
        raise ValueError(f"Unsupported scheduler: {self.scheduler}")

    # ------------------------------------------------------------------
    def restart(self, job_script: str, checkpoint: str, **kwargs: Any) -> Any:
        """Convenience wrapper to restart a job from ``checkpoint``.

        The checkpoint path is passed through to the underlying scheduler
        submission so that job scripts can act accordingly.
        """

        kwargs = dict(kwargs)
        kwargs["restart"] = checkpoint
        return self.submit(job_script, **kwargs)


__all__ = ["JobManager"]
