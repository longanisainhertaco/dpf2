from __future__ import annotations

import hashlib
import json
import warnings
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


class GridResolution(ConfigSectionBase):
    """Domain and grid resolution configuration."""

    config_section_id: ClassVar[Literal["grid"]] = "grid"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Core grid parameters
    nx: int = Field(..., ge=1, metadata={"units": "cells"})
    ny: int = Field(..., ge=1, metadata={"units": "cells"})
    nz: int = Field(..., ge=1, metadata={"units": "cells"})

    x_min: float = Field(..., alias="xMin", metadata={"units": "cm"})
    x_max: float = Field(..., alias="xMax", metadata={"units": "cm"})
    y_min: float = Field(..., alias="yMin", metadata={"units": "cm"})
    y_max: float = Field(..., alias="yMax", metadata={"units": "cm"})
    z_min: float = Field(..., alias="zMin", metadata={"units": "cm"})
    z_max: float = Field(..., alias="zMax", metadata={"units": "cm"})

    # Optional controls -------------------------------------------------
    symmetry_axis: Optional[Literal["x", "y", "z", "none"]] = "none"
    grid_centering: Optional[Literal["cell", "face", "node"]] = "cell"
    padding_cells: Optional[
        Dict[Literal["xLow", "xHigh", "yLow", "yHigh", "zLow", "zHigh"], int]
    ] = Field(default_factory=dict)
    domain_refinement_level: Optional[int] = Field(0, ge=0, le=5)
    nonuniform_axis_scaling: Optional[Dict[str, Literal["uniform", "log", "exp"]]] = Field(
        default_factory=dict
    )
    grid_config_hash: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls, geometry: str) -> "GridResolution":
        if geometry == "2D_RZ":
            return cls(
                nx=128,
                ny=1,
                nz=256,
                x_min=0.0,
                x_max=5.0,
                y_min=0.0,
                y_max=0.0,
                z_min=-2.5,
                z_max=2.5,
                symmetry_axis="y",
                grid_centering="cell",
            )
        return cls(
            nx=128,
            ny=128,
            nz=128,
            x_min=-0.5,
            x_max=0.5,
            y_min=-0.5,
            y_max=0.5,
            z_min=-0.5,
            z_max=0.5,
            symmetry_axis="none",
            grid_centering="cell",
        )

    def resolve_defaults(self, geometry: str) -> "GridResolution":
        data = self.model_dump()
        defaults = self.with_defaults(geometry).model_dump()
        defaults.update({k: v for k, v in data.items() if v is not None})
        return self.model_validate(defaults, context={"geometry": geometry})

    @classmethod
    def required_fields(cls) -> List[str]:
        return [name for name, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {name: (field.json_schema_extra or field.metadata or {}) for name, field in cls.model_fields.items()}

    def normalize_units(self, spatial_units: Literal["cm", "m"]) -> "GridResolution":
        scale = 0.01 if spatial_units == "cm" else 1.0
        update = {
            "x_min": self.x_min * scale,
            "x_max": self.x_max * scale,
            "y_min": self.y_min * scale,
            "y_max": self.y_max * scale,
            "z_min": self.z_min * scale,
            "z_max": self.z_max * scale,
        }
        return self.model_copy(update=update)

    def summarize(self) -> str:
        pad = ", ".join(f"{k}={v}" for k, v in self.padding_cells.items()) or "none"
        hash_short = self.grid_config_hash[:6] if self.grid_config_hash else "none"
        return (
            f"Grid: {self.nx}\u00d7{self.ny}\u00d7{self.nz}, Domain: [{self.x_min},{self.x_max}]\u00d7"
            f"[{self.y_min},{self.y_max}]\u00d7[{self.z_min},{self.z_max}] cm\n"
            f"Symmetry: {self.symmetry_axis}, Refinement: {self.domain_refinement_level}, Padding: {pad}\n"
            f"Centering: {self.grid_centering}, Hash: {hash_short}"
        )

    # Helper methods ----------------------------------------------------
    def grid_summary_string(self) -> str:
        return (
            f"{self.nx}\u00d7{self.ny}\u00d7{self.nz} @ [{self.x_min},{self.x_max}]\u00d7"
            f"[{self.z_min},{self.z_max}] cm"
        )

    def total_cells(self) -> int:
        return self.nx * self.ny * self.nz

    def cell_sizes(self) -> Tuple[float, float, float]:
        return (
            (self.x_max - self.x_min) / self.nx,
            max((self.y_max - self.y_min) / max(1, self.ny), 1e-8),
            (self.z_max - self.z_min) / self.nz,
        )

    def hash_grid_config(self) -> str:
        data = self.model_dump(exclude={"grid_config_hash"}, by_alias=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "GridResolution") -> "GridResolution":
        geometry = getattr(values, "_context", {}).get("geometry")
        eps = 1e-8

        if geometry == "2D_RZ":
            if values.ny != 1:
                raise ValueError("ny must be 1 for 2D_RZ geometry")
            if abs(values.y_min) > eps or abs(values.y_max) > eps:
                raise ValueError("y_min and y_max must be 0.0 for 2D_RZ geometry")
            if values.symmetry_axis != "y":
                raise ValueError("symmetry_axis must be 'y' for 2D_RZ geometry")

        if values.x_max - values.x_min <= eps:
            raise ValueError("x_max must be > x_min")
        if values.z_max - values.z_min <= eps:
            raise ValueError("z_max must be > z_min")
        if geometry != "2D_RZ" and values.y_max - values.y_min <= eps:
            raise ValueError("y_max must be > y_min")

        valid_faces = {"xLow", "xHigh", "yLow", "yHigh", "zLow", "zHigh"}
        for face, count in values.padding_cells.items():
            if face not in valid_faces or count < 0:
                raise ValueError("padding_cells contains invalid entries")

        lengths = [
            values.x_max - values.x_min,
            max(values.y_max - values.y_min, 1e-8),
            values.z_max - values.z_min,
        ]
        for i, L1 in enumerate(lengths):
            for j, L2 in enumerate(lengths):
                if i != j and L2 > eps and L1 / L2 > 20.0:
                    warnings.warn("Domain aspect ratio exceeds 20:1")
                    break

        if values.nonuniform_axis_scaling:
            allowed = {"uniform", "log", "exp"}
            for ax, mode in values.nonuniform_axis_scaling.items():
                if ax not in {"x", "y", "z"}:
                    raise ValueError("nonuniform_axis_scaling keys must be 'x', 'y', or 'z'")
                if mode not in allowed:
                    raise ValueError("nonuniform_axis_scaling values must be 'uniform', 'log', or 'exp'")

        if values.grid_centering == "node" and (values.nx % 2 == 0 or values.nz % 2 == 0):
            warnings.warn("Node-centered grids typically use odd nx and nz")

        values = values.model_copy(update={"grid_config_hash": values.hash_grid_config()})
        return values

    # ------------------------------------------------------------------
    @classmethod
    def model_validate(cls, data: Any, **kwargs) -> "GridResolution":
        context = kwargs.get("context") or {}
        obj = super().model_validate(data)
        object.__setattr__(obj, "_context", context)
        obj = cls.check_rules(obj)
        return obj


__all__ = ["GridResolution"]

