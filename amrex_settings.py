from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, ConfigDict, Field, root_validator

# ---------------------------------------------------------------------------
# Compatibility helpers mirroring pydantic v2 model_validator

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

# Local imports ---------------------------------------------------------------
from core_schema import ConfigSectionBase, to_camel_case


class ElectrodeGeometry(ConfigSectionBase):
    """Electrode mesh and geometry configuration."""

    config_section_id: ClassVar[Literal["electrode_geometry"]] = "electrode_geometry"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    cathode_type: Literal["bars", "ring", "custom"]
    cathode_bar_count: Optional[int] = Field(None, ge=2)
    cathode_gap_degrees: Optional[float] = None
    anode_shape: Literal["cylinder", "cone", "knife", "custom"]
    knife_edge_enabled: bool
    emitter_field_enhancement: Optional[float] = Field(None, ge=0.0)
    mesh_file: Optional[Path] = None
    mesh_file_units: Optional[Literal["cm", "m"]] = "cm"
    material_tagging_enabled: bool = False

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "ElectrodeGeometry":
        return cls(
            cathode_type="bars",
            cathode_bar_count=16,
            cathode_gap_degrees=15.0,
            anode_shape="cylinder",
            knife_edge_enabled=False,
        )

    def resolve_defaults(self) -> "ElectrodeGeometry":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {
            name: (field.json_schema_extra or field.metadata or {})
            for name, field in cls.model_fields.items()
        }

    def normalize_units(self, spatial_units: Literal["cm", "m"]) -> "ElectrodeGeometry":
        if self.mesh_file_units and spatial_units != self.mesh_file_units:
            return self.model_copy(update={"mesh_file_units": spatial_units})
        return self

    def summarize(self) -> str:
        bar = f"{self.cathode_type}({self.cathode_bar_count})" if self.cathode_bar_count else self.cathode_type
        return f"{bar}, {self.anode_shape} anode"


class AmrexSettings(ConfigSectionBase):
    """Configuration schema for AMReX solver settings."""

    config_section_id: ClassVar[Literal["amrex"]] = "amrex"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Core solver parameters
    amr_levels: int = Field(..., ge=1, le=5)
    amr_coarsening_ratio: int = Field(2, ge=2, le=4)
    integrator: Literal["RK2", "RK4", "Godunov", "Euler"] = "RK4"
    stencil_order: int = Field(2, ge=2, le=6)
    solver_tolerance: float = Field(1e-8)
    embedded_boundary: bool = Field(False)
    embedded_boundary_extrapolation: Optional[Literal["none", "linear", "parabolic"]] = "none"
    flux_limiter_enabled: bool = Field(False)
    numerical_damping_factor: Optional[float] = Field(None, ge=0.0, le=1.0)
    enabled_field_solvers: List[Literal["Poisson", "HLLC", "Diffusion"]] = Field(default_factory=list)
    gradient_method: Optional[Literal["PLM", "PPM", "WENO", "linear"]] = "PLM"
    interpolation_order: Optional[int] = Field(2, ge=1, le=5)

    # Tile and mesh controls
    tile_size_override: Optional[Tuple[int, int, int]] = None
    coarse_block_size: Optional[Tuple[int, int, int]] = None
    max_grid_size: Optional[int] = Field(None, ge=8)

    # Electrode and material settings
    electrode_geometry: ElectrodeGeometry
    electrode_material: Literal["Cu", "W", "Al", "Mo"] = "Cu"
    erosion_mechanisms_enabled: List[Literal["thermal", "sputtering"]] = Field(default_factory=list)
    material_properties_override: Optional[
        Dict[
            Literal[
                "thermal_conductivity",
                "resistivity",
                "emissivity",
                "yield_strength",
            ],
            float,
        ]
    ] = None

    amrex_config_hash: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "AmrexSettings":
        return cls(
            amr_levels=1,
            amr_coarsening_ratio=2,
            integrator="RK4",
            stencil_order=2,
            electrode_geometry=ElectrodeGeometry.with_defaults(),
        )

    def resolve_defaults(self) -> "AmrexSettings":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {
            name: (field.json_schema_extra or field.metadata or {})
            for name, field in cls.model_fields.items()
        }

    def normalize_units(self, spatial_units: Literal["cm", "m"]) -> "AmrexSettings":
        eg = self.electrode_geometry.normalize_units(spatial_units)
        return self.model_copy(update={"electrode_geometry": eg})

    def summarize(self) -> str:
        eb_state = "on" if self.embedded_boundary else "off"
        extrap = self.embedded_boundary_extrapolation or "none"
        tile = list(self.tile_size_override) if self.tile_size_override else None
        block = list(self.coarse_block_size) if self.coarse_block_size else None
        erosion = (
            " + ".join(self.erosion_mechanisms_enabled)
            if self.erosion_mechanisms_enabled
            else "none"
        )
        parts = [
            f"AMReX: {self.amr_levels} levels, {self.integrator}, stencil={self.stencil_order}, tol={self.solver_tolerance}",
            f"EB: {eb_state}, extrapolation: {extrap}, flux limiter: {'ON' if self.flux_limiter_enabled else 'OFF'}",
            f"Tile: {tile}, Block: {block}, GridMax: {self.max_grid_size}",
            f"Electrodes: {self.electrode_geometry.summarize()}, erosion = {erosion}",
        ]
        return "\n".join(parts)

    def hash_amrex_config(self) -> str:
        data = self.model_dump(exclude={"amrex_config_hash"}, by_alias=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "AmrexSettings") -> "AmrexSettings":
        if (
            values.tile_size_override is not None
            and values.max_grid_size is not None
            and any(t > values.max_grid_size for t in values.tile_size_override)
        ):
            raise ValueError("tile_size_override cannot exceed max_grid_size")

        if values.tile_size_override is not None:
            if len(values.tile_size_override) != 3 or any(t <= 0 for t in values.tile_size_override):
                raise ValueError("tile_size_override must be three positive integers")
        if values.coarse_block_size is not None:
            if len(values.coarse_block_size) != 3 or any(b <= 0 for b in values.coarse_block_size):
                raise ValueError("coarse_block_size must be three positive integers")
        if values.tile_size_override and values.coarse_block_size:
            for t, b in zip(values.tile_size_override, values.coarse_block_size):
                if b % t != 0:
                    raise ValueError("coarse_block_size must be divisible by tile_size_override")
        if values.coarse_block_size:
            for b in values.coarse_block_size:
                if b % values.amr_coarsening_ratio != 0:
                    raise ValueError("coarse_block_size must be divisible by amr_coarsening_ratio")
        if len(set(values.enabled_field_solvers)) != len(values.enabled_field_solvers):
            raise ValueError("enabled_field_solvers contains duplicates")
        allowed_keys = {"thermal_conductivity", "resistivity", "emissivity", "yield_strength"}
        if values.material_properties_override:
            for k in values.material_properties_override:
                if k not in allowed_keys:
                    raise ValueError("material_properties_override contains invalid key")
        if not values.embedded_boundary and values.embedded_boundary_extrapolation not in (None, "none"):
            raise ValueError("embedded_boundary_extrapolation requires embedded_boundary")
        values = values.model_copy(update={"amrex_config_hash": values.hash_amrex_config()})
        return values


__all__ = ["AmrexSettings", "ElectrodeGeometry"]
