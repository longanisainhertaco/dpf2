"""Core configuration schemas for DPF simulations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Type aliases
TimeVoltageProfile = List[Tuple[float, float]]
DetectorConfig = Dict[str, Any]
ConfigOverride = Dict[str, Union[float, str]]
FieldUnit = str

# ---------------------------------------------------------------------------
# Helper for camelCase aliasing

def to_camel_case(string: str) -> str:
    parts = string.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

# ---------------------------------------------------------------------------
# Enums

class GeometryType(str, Enum):
    """Defines valid simulation geometries."""

    RZ_2D = "2D_RZ"
    XYZ_3D = "3D_Cartesian"

class ModeType(str, Enum):
    """Defines valid DPF solver modes."""

    FLUID = "fluid"
    PIC = "PIC"
    HYBRID = "hybrid"

class UnitsSystem(str, Enum):
    """Defines the base unit system used for internal normalization."""

    SI = "SI"
    CGS = "cgs"
    LAB = "lab"

class ValidationPolicy(str, Enum):
    """Controls schema enforcement mode on load/override."""

    STRICT = "strict"
    WARN = "warn"
    SILENT = "silent"

# ---------------------------------------------------------------------------
# Base configuration section

class ConfigSectionBase(BaseModel):
    """Base class for all configuration sections."""

    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    config_section_id: ClassVar[str]

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
        frozen=True,
    )

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "ConfigSectionBase":
        return cls()  # type: ignore[call-arg]

    def required_fields(self) -> List[str]:
        return [name for name, field in self.model_fields.items() if field.is_required()]

    def resolve_defaults(self) -> "ConfigSectionBase":
        data = self.model_dump()
        return self.model_validate(data)

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: field.json_schema_extra or {}
            for name, field in self.model_fields.items()
        }

# ---------------------------------------------------------------------------
# Root configuration model

class DPFConfig(BaseModel):
    """Root configuration object."""

    # Required submodels (placeholders for actual definitions)
    simulation: ConfigSectionBase
    grid: ConfigSectionBase
    initial: ConfigSectionBase
    physics: ConfigSectionBase
    circuit: ConfigSectionBase
    amrex: ConfigSectionBase
    warpx: ConfigSectionBase
    diagnostics: ConfigSectionBase
    variability: ConfigSectionBase
    benchmark: ConfigSectionBase
    boundary: ConfigSectionBase
    parallel: ConfigSectionBase
    metadata: ConfigSectionBase
    advanced: ConfigSectionBase
    units: ConfigSectionBase

    run_uuid: str = Field(..., alias="runUuid")
    schema_version: str = Field(..., alias="schemaVersion")
    created_at: datetime = Field(..., alias="createdAt")
    config_hash: Optional[str] = None
    restart_hash: Optional[str] = None
    run_lineage: Optional[List[str]] = None

    on_validation_error: ValidationPolicy = ValidationPolicy.STRICT
    validation_output_path: Optional[Path] = None

    schema_migrations: ClassVar[Dict[str, Any]] = {}

    model_config = ConfigDict(
        extra="allow",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "DPFConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        if p.suffix in {".yaml", ".yml"}:
            import yaml
            data = yaml.safe_load(p.read_text())
        else:
            import json
            data = json.loads(p.read_text())
        return cls.model_validate(data)

    def to_file(self, path: Union[str, Path], format: Union[str, None] = "json") -> None:
        p = Path(path)
        if format == "yaml":
            import yaml
            p.write_text(yaml.safe_dump(self.model_dump(by_alias=True)))
        else:
            import json
            p.write_text(json.dumps(self.model_dump(by_alias=True), indent=2))

    def override(self, key_path: str, value: Any) -> "DPFConfig":
        parts = key_path.split(".")
        data = self.model_dump()
        ref = data
        for p in parts[:-1]:
            ref = ref.setdefault(p, {})
        ref[parts[-1]] = value
        return self.model_validate(data)

    def validate(self) -> "DPFConfig":
        return self.model_validate(self.model_dump())

    def unit_scale(self) -> float:
        return 1.0

    def duration(self) -> float:
        sim = self.simulation
        start = getattr(sim, "time_start", 0.0)
        end = getattr(sim, "time_end", 0.0)
        return end - start

    def schema_export(self, path: Union[str, Path], format: str = "json") -> None:
        p = Path(path)
        if format == "yaml":
            import yaml
            p.write_text(yaml.safe_dump(self.model_json_schema()))
        else:
            import json
            p.write_text(json.dumps(self.model_json_schema(), indent=2))

    def normalize_units(self, base_units: UnitsSystem) -> None:
        pass

    def summarize(self) -> str:
        return f"DPFConfig(schema_version={self.schema_version}, run_uuid={self.run_uuid})"

    @classmethod
    def validate_and_fill(cls, raw: Union[str, Path, Dict[str, Any]]) -> "DPFConfig":
        if isinstance(raw, (str, Path)):
            return cls.from_file(raw)
        return cls.model_validate(raw)

    def resolve_defaults(self) -> "DPFConfig":
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [name for name, field in self.model_fields.items() if field.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: field.json_schema_extra or {}
            for name, field in self.model_fields.items()
        }

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_global_cross_dependencies(cls, values: "DPFConfig") -> "DPFConfig":
        sim = values.simulation
        grid = values.grid
        physics = values.physics
        diag = values.diagnostics
        if getattr(sim, "geometry", None) == GeometryType.RZ_2D.value:
            if getattr(grid, "ny", 1) != 1:
                raise ValueError("ny must be 1 for 2D_RZ geometry")
        if getattr(sim, "mode", None) == ModeType.PIC.value:
            if not getattr(physics, "neutral_fluid_enabled", False):
                raise ValueError("neutral_fluid_enabled must be True in PIC mode")
        if getattr(diag, "neutron_fluence_map", False):
            gas_type = getattr(values.initial, "gas_type", "")
            if gas_type not in {"D2", "DT"}:
                raise ValueError("neutron diagnostics require D2 or DT gas")
        if getattr(values.circuit, "switching_model", "") == "multi-bank":
            if getattr(values.circuit, "switch_feedback_delay_ns", None) is None:
                raise ValueError("switch_feedback_delay_ns is required for multi-bank model")
        return values

# ---------------------------------------------------------------------------
# Example test stubs

def test_round_trip():
    cfg = DPFConfig.validate_and_fill({})
    assert cfg

# ---------------------------------------------------------------------------
# Example usage

if __name__ == "__main__":
    cfg = DPFConfig.validate_and_fill({})
    cfg.validate()
    cfg.normalize_units(UnitsSystem.SI)
    print(cfg.summarize())
    cfg.schema_export("dpf_schema.json")

# ---------------------------------------------------------------------------
__all__ = [
    "ConfigSectionBase",
    "DPFConfig",
    "TimeVoltageProfile",
    "DetectorConfig",
    "ConfigOverride",
    "FieldUnit",
    "GeometryType",
    "ModeType",
    "UnitsSystem",
    "ValidationPolicy",
]
