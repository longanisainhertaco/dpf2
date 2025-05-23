from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, ConfigDict, Field, root_validator

# ---------------------------------------------------------------------------
# Compatibility helpers mirroring pydantic.v2 model_validator

def model_validator(*, mode: str = "after"):
    def decorator(func):
        if mode == "after":
            def wrapper(cls, values):
                inst = cls.construct(**values)
                result = func(cls, inst)
                return result.__dict__ if isinstance(result, cls) else values
            return root_validator(pre=False, skip_on_failure=True, allow_reuse=True)(wrapper)
        else:
            def wrapper(cls, values):
                out = func(values)
                return out if out is not None else values
            return root_validator(pre=True, skip_on_failure=True, allow_reuse=True)(wrapper)
    return decorator

if not hasattr(BaseModel, "model_validate"):
    BaseModel.model_validate = classmethod(lambda cls, d, **_: cls.parse_obj(d))
if not hasattr(BaseModel, "model_dump"):
    BaseModel.model_dump = BaseModel.dict
if not hasattr(BaseModel, "model_dump_json"):
    BaseModel.model_dump_json = BaseModel.json
if not hasattr(BaseModel, "model_copy"):
    BaseModel.model_copy = BaseModel.copy

from core_schema import ConfigSectionBase, to_camel_case


class ParallelSettings(ConfigSectionBase):
    """Parallel execution and hardware configuration."""

    config_section_id: ClassVar[Literal["parallel"]] = "parallel"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    mpi_ranks: int = Field(..., ge=1, metadata={"units": "ranks", "group": "MPI"})
    gpu_backend: Literal["CUDA", "HIP", "None"] = Field(..., alias="gpuBackend", metadata={"group": "GPU"})
    use_multithreading: bool = True
    decomposition_strategy: Literal["slices", "blocks", "auto", "manual"] = "auto"
    amr_refinement_criteria: Literal["gradient", "current_density", "manual"] = "gradient"

    threads_per_rank: Optional[int] = Field(None, metadata={"units": "threads"})
    bind_to_core: Optional[bool] = Field(True)
    gpu_device_ids: Optional[List[int]] = Field(default_factory=list)
    total_available_gpus: Optional[int] = None
    gpu_partition_strategy: Literal["1-per-rank", "shared", "auto", "round_robin", "affinity_map"] = "auto"
    gpu_node_affinity: Optional[Dict[int, List[int]]] = None
    tile_size: Optional[Tuple[int, int, int]] = Field(None, metadata={"units": "cells"})
    buffer_pool_size_MB: Optional[int] = Field(None, metadata={"units": "MB"})
    pinned_memory: bool = True

    max_allowed_cores: Optional[int] = Field(None, metadata={"units": "logical cores"})
    decomposition_failure_policy: Literal["abort", "warn", "auto_reduce"] = "abort"
    mesh_dependent_partitioning: bool = False
    disable_hardware_autodetection: bool = Field(False)
    gpu_fallback_mode: Literal["abort", "cpu_only", "retry"] = "abort"
    gpu_usage_fraction_per_rank: Optional[float] = Field(None, ge=0.0, le=1.0)

    scheduler_context: Literal["slurm", "pbs", "none"] = "none"
    launcher_type: Literal["mpirun", "srun", "jsrun", "auto"] = "auto"
    load_balancing_strategy: Literal["static", "dynamic", "hybrid"] = "dynamic"
    subdomain_ordering: Literal["Z-Morton", "Hilbert", "Linear"] = "Z-Morton"
    verbosity_level: Optional[int] = Field(1, ge=0, le=5)

    parallel_config_hash: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "ParallelSettings":
        return cls(
            mpi_ranks=1,
            gpu_backend="None",
            use_multithreading=True,
            decomposition_strategy="auto",
            amr_refinement_criteria="gradient",
            threads_per_rank=1,
            bind_to_core=True,
        )

    def resolve_defaults(self) -> "ParallelSettings":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [name for name, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {name: (field.json_schema_extra or field.metadata or {}) for name, field in cls.model_fields.items()}

    def normalize_units(self, unit_map: Optional[Dict[str, float]]) -> "ParallelSettings":
        if unit_map is None:
            return self
        update: Dict[str, Any] = {}
        if self.buffer_pool_size_MB is not None and "MB" in unit_map:
            update["buffer_pool_size_MB"] = int(self.buffer_pool_size_MB * unit_map["MB"])
        if self.tile_size is not None and "cells" in unit_map:
            scale = unit_map["cells"]
            update["tile_size"] = tuple(int(x * scale) for x in self.tile_size)
        return self.model_copy(update=update)

    def summarize(self) -> str:
        parts = [
            f"Parallelism: {self.mpi_ranks} MPI ranks, {self.threads_per_rank or 1} threads/rank, GPU = {self.gpu_backend} (devices: {self.gpu_device_ids})",
            f"Decomposition: {self.decomposition_strategy} (AMR = {self.amr_refinement_criteria}), Fallback = {self.decomposition_failure_policy}",
            f"Bind to core: {self.bind_to_core}, Scheduler = {self.scheduler_context}, Launcher = {self.launcher_type}",
            f"Memory: pinned = {self.pinned_memory}, buffer pool = {self.buffer_pool_size_MB} MB",
            f"Tile size: {list(self.tile_size) if self.tile_size else None}, Partition strategy: {self.gpu_partition_strategy}",
        ]
        warnings_list: List[str] = []
        if (
            self.max_allowed_cores is not None
            and (self.threads_per_rank or 1) * self.mpi_ranks > self.max_allowed_cores
        ):
            warnings_list.append(
                f"\u26a0 Threads/rank \xd7 MPI ranks exceeds max allowed cores ({self.max_allowed_cores})"
            )
        if self.gpu_backend == "HIP" and self.pinned_memory:
            warnings_list.append("\u26a0 HIP backend does not support pinned memory")
        if warnings_list:
            parts.extend(warnings_list)
        return "\n".join(parts)

    def hash_parallel_config(self) -> str:
        data = {
            "mpi_ranks": self.mpi_ranks,
            "gpu_backend": self.gpu_backend,
            "use_multithreading": self.use_multithreading,
            "threads_per_rank": self.threads_per_rank,
            "gpu_device_ids": self.gpu_device_ids,
            "decomposition_strategy": self.decomposition_strategy,
            "amr_refinement_criteria": self.amr_refinement_criteria,
            "gpu_partition_strategy": self.gpu_partition_strategy,
            "tile_size": self.tile_size,
            "pinned_memory": self.pinned_memory,
            "load_balancing_strategy": self.load_balancing_strategy,
            "scheduler_context": self.scheduler_context,
            "launcher_type": self.launcher_type,
            "mesh_dependent_partitioning": self.mesh_dependent_partitioning,
            "gpu_usage_fraction_per_rank": self.gpu_usage_fraction_per_rank,
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "ParallelSettings") -> "ParallelSettings":
        if values.gpu_backend != "None" and not values.gpu_device_ids:
            raise ValueError("gpu_device_ids required when GPU backend is enabled")
        if values.gpu_backend == "HIP" and values.pinned_memory:
            warnings.warn("HIP backend does not support pinned memory")
        if values.threads_per_rank and values.threads_per_rank > 1 and not values.use_multithreading:
            warnings.warn("threads_per_rank > 1 but multithreading disabled")
        if (
            values.decomposition_strategy == "manual"
            and values.amr_refinement_criteria != "manual"
        ):
            raise ValueError("manual decomposition requires manual AMR refinement criteria")
        if values.tile_size is not None:
            if len(values.tile_size) != 3 or any(t <= 0 for t in values.tile_size):
                raise ValueError("tile_size must be three positive integers")
        values = values.model_copy(update={"parallel_config_hash": values.hash_parallel_config()})
        return values


__all__ = ["ParallelSettings"]
