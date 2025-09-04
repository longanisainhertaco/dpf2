"""Simple job manager for scheduling simulations on HPC resources.

This module provides a :class:`JobManager` that abstracts submission of
simulation runs to different schedulers such as MPI via ``mpirun``,
SLURM clusters and AWS Batch.  The implementation intentionally focuses on
small scale examples and does not aim to expose the full feature set of the
underlying schedulers.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import tempfile
from typing import Any, Dict, Iterable, Mapping


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

    def _write_temp_file(self, content: str, suffix: str = "") -> str:
        """Write ``content`` to a temporary file and return its path."""

        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        return path

    def _wrap_staging(
        self,
        job_script: str,
        stage_in: Mapping[str, str] | None,
        stage_out: Mapping[str, str] | None,
    ) -> str:
        """Create a wrapper script that performs data staging.

        Parameters
        ----------
        job_script:
            Path to the actual job script.
        stage_in, stage_out:
            Optional mappings of ``source -> destination`` that should be
            copied before/after executing ``job_script``.
        """

        if not stage_in and not stage_out:
            return job_script

        lines = ["#!/bin/bash"]
        if stage_in:
            for src, dst in stage_in.items():
                lines.append(f"cp -r {src} {dst}")
        lines.append(job_script)
        if stage_out:
            for src, dst in stage_out.items():
                lines.append(f"cp -r {src} {dst}")

        fd, path = tempfile.mkstemp(suffix=".sh")
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.chmod(path, 0o755)
        return path

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

        stage_in: Mapping[str, str] | None = kwargs.pop("stage_in", None)
        stage_out: Mapping[str, str] | None = kwargs.pop("stage_out", None)
        job_script = self._wrap_staging(job_script, stage_in, stage_out)

        # Script level arguments and restart handling
        script_args: list[str] = list(kwargs.pop("script_args", []))
        restart = kwargs.pop("restart", None)
        if restart is not None:
            script_args.extend(["--restart", str(restart)])

        if self.scheduler == "slurm":
            gpus = kwargs.pop("gpus", None)
            gpu_type = kwargs.pop("gpu_type", None)
            deps = kwargs.pop("dependencies", None)
            dep_type = kwargs.pop("dependency_type", "afterok")
            gpu_affinity = kwargs.pop("gpu_affinity", None)

            cmd = ["sbatch"]
            self._extend_cmd(
                cmd,
                kwargs,
                {
                    "nodes": "-N",
                    "nodelist": "--nodelist",
                    "ntasks_per_node": "--ntasks-per-node",
                    "output": "-o",
                    "error": "-e",
                },
            )
            if gpu_type is not None:
                count = gpus if gpus is not None else 1
                cmd.extend(["--gres", f"gpu:{gpu_type}:{count}"])
            elif gpus is not None:
                cmd.extend(["--gpus", str(gpus)])
            if deps is not None:
                if isinstance(deps, (list, tuple)):
                    dep_str = f"{dep_type}:{':'.join(str(d) for d in deps)}"
                else:
                    dep_str = str(deps)
                cmd.extend(["--dependency", dep_str])
            cmd.append(job_script)
            cmd.extend(script_args)
            env = os.environ.copy()
            if gpu_affinity is not None:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_affinity)
            if restart is not None:
                # Allow job scripts to locate the checkpoint for staging
                env["DPF_RESTART"] = str(restart)
            return subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if self.scheduler == "awsbatch":
            try:
                import boto3  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("boto3 is required for AWS Batch submissions") from exc
            batch = boto3.client("batch")
            gpus = kwargs.pop("gpus", None)
            gpu_type = kwargs.pop("gpu_type", None)
            deps = kwargs.pop("dependencies", None)
            gpu_affinity = kwargs.pop("gpu_affinity", None)

            params: Dict[str, Any] = {
                "jobName": kwargs.get("job_name", "dpf2-job"),
                "jobQueue": kwargs["job_queue"],
                "jobDefinition": kwargs["job_definition"],
            }
            if "parameters" in kwargs:
                params["parameters"] = kwargs["parameters"]
            if deps is not None:
                dep_list = deps if isinstance(deps, (list, tuple)) else [deps]
                params["dependsOn"] = [{"jobId": str(d)} for d in dep_list]
            if gpus is not None or gpu_type is not None:
                rr = {"value": str(gpus if gpus is not None else 1), "type": "GPU"}
                params.setdefault("containerOverrides", {})["resourceRequirements"] = [rr]
            if restart is not None or gpu_affinity is not None:
                env_list = params.setdefault("containerOverrides", {}).setdefault("environment", [])
                if restart is not None:
                    env_list.append({"name": "DPF_RESTART", "value": str(restart)})
                if gpu_affinity is not None:
                    env_list.append({"name": "CUDA_VISIBLE_DEVICES", "value": ",".join(str(g) for g in gpu_affinity)})
            if script_args:
                params.setdefault("containerOverrides", {})["command"] = [job_script] + script_args
            return batch.submit_job(**params)
        if self.scheduler == "mpi":
            gpus = kwargs.pop("gpus", None)
            gpu_type = kwargs.pop("gpu_type", None)  # currently unused
            deps = kwargs.pop("dependencies", None)
            if deps is not None:
                raise ValueError("Dependencies are not supported for MPI scheduler")

            hosts = kwargs.pop("hosts", None)
            host_gpus = kwargs.pop("host_gpus", None)
            node_topology = kwargs.pop("node_topology", None)
            decomp: Dict[str, Any] | None = kwargs.pop("decomp", None)
            gpu_affinity = kwargs.pop("gpu_affinity", None)

            if node_topology is not None:
                if host_gpus is not None:
                    raise ValueError("Specify either node_topology or host_gpus")
                host_gpus = node_topology

            cmd = ["mpirun"]
            self._extend_cmd(
                cmd,
                kwargs,
                {
                    "nprocs": "-n",
                },
            )

            env = os.environ.copy()

            # Generate hostfile and GPU mapping if requested
            if host_gpus is not None:
                host_lines = []
                gpu_map_lines = []
                rank = 0
                for host, gpu_list in host_gpus.items():
                    host_lines.append(f"{host} slots={len(gpu_list)}")
                    for gpu in gpu_list:
                        gpu_map_lines.append(f"{rank} {host} {gpu}")
                        rank += 1
                hostfile = self._write_temp_file("\n".join(host_lines) + "\n", suffix=".hosts")
                cmd.extend(["--hostfile", hostfile])
                gpu_map_file = self._write_temp_file("\n".join(gpu_map_lines) + "\n", suffix=".gpus")
                env["DPF_GPU_MAP"] = gpu_map_file
            elif hosts is not None:
                hostfile = self._write_temp_file("\n".join(hosts) + "\n", suffix=".hosts")
                cmd.extend(["--hostfile", hostfile])

            script_cmd = [job_script] + script_args
            if decomp:
                for axis, cnt in decomp.items():
                    script_cmd.extend([f"--decomp-{axis}", str(cnt)])

            if gpu_affinity is not None:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_affinity)
            elif gpus is not None and host_gpus is None:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(int(gpus)))
            if restart is not None:
                env["DPF_RESTART"] = str(restart)

            cmd.extend(script_cmd)
            return subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
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
