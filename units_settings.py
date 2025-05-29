from __future__ import annotations

import hashlib
import json
from typing import Any, ClassVar, Dict, Optional, List, Literal

from pydantic import BaseModel, ConfigDict, Field, root_validator


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


class UnitsSettings(ConfigSectionBase):
    """Unit system configuration for DPF simulations."""

    config_section_id: ClassVar[Literal["units"]] = "units"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        allow_population_by_field_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Core unit specification
    base_units: Literal["SI", "CGS", "lab"] = Field(
        ..., description="System of reference units for simulation"
    )
    spatial_units: Literal["m", "cm"] = Field(
        ..., alias="spatialUnits", description="Length units for input/output"
    )
    temporal_units: Literal["s", "ms", "us", "ns"] = Field(
        ..., alias="temporalUnits", description="Time units for input/output"
    )

    # Conversion and output flags
    auto_convert_on_load: bool = Field(
        True, alias="autoConvertOnLoad", description="Normalize YAML/JSON input values to internal units"
    )
    normalize_to_internal: bool = Field(
        True, alias="normalizeToInternal", description="Convert all runtime values to solver base units"
    )
    unit_validation_policy: Literal["strict", "warn", "ignore"] = Field(
        "strict", alias="unitValidationPolicy", description="How to handle invalid or unknown units"
    )

    # Output & UI scaling
    preferred_output_units: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        alias="preferredOutputUnits",
        description="Field → units map for output formatting",
    )
    preferred_output_units_file: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        alias="preferredOutputUnitsFile",
        description="Field → units map for file/disk export",
    )

    # Input assumptions
    input_assumed_units: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        alias="inputAssumedUnits",
        description="Field → assumed units for raw config input",
    )

    # Manual overrides
    unit_resolution_overrides: Optional[
        Dict[Literal["pressure", "temperature", "rho", "J", "T", "E", "B"], float]
    ] = Field(
        default_factory=dict,
        alias="unitResolutionOverrides",
        description="Per-field manual unit scalers to override conversion logic (e.g., pressure: 1e5)",
    )

    # Hash of full config
    units_config_hash: Optional[str] = Field(
        None, alias="unitsConfigHash", description="Stable hash of all unit config values"
    )

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "UnitsSettings":
        return cls.model_validate(
            {
                "base_units": "SI",
                "spatial_units": "m",
                "temporal_units": "ns",
            }
        )

    def resolve_defaults(self) -> "UnitsSettings":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [name for name, fld in cls.model_fields.items() if fld.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {name: (field.json_schema_extra or field.metadata or {}) for name, field in cls.model_fields.items()}

    def normalize_units(self) -> Dict[str, float]:
        """Return unit scaling map based on current settings."""
        base_len = "m" if self.base_units == "SI" else "cm"
        base_time = "ns" if self.base_units == "lab" else "s"
        length_scale = 1.0
        time_scale = 1.0
        if self.spatial_units != base_len:
            conv = {("cm", "m"): 0.01, ("m", "cm"): 100.0}
            length_scale = conv.get((self.spatial_units, base_len), 1.0)
        if self.temporal_units != base_time:
            conv_t = {
                ("s", "ns"): 1e9,
                ("ms", "ns"): 1e6,
                ("us", "ns"): 1e3,
                ("ns", "s"): 1e-9,
                ("ms", "s"): 1e-3,
                ("us", "s"): 1e-6,
            }
            time_scale = conv_t.get((self.temporal_units, base_time), 1.0)
        unit_map: Dict[str, float] = {"spatial": length_scale, "temporal": time_scale}
        if self.unit_resolution_overrides:
            unit_map.update(self.unit_resolution_overrides)
        if self.input_assumed_units:
            for key in self.input_assumed_units:
                unit_map.setdefault(key, 1.0)
        object.__setattr__(self, "units_config_hash", self.hash_units_config())
        return unit_map

    def hash_units_config(self) -> str:
        data = self.model_dump(exclude={"units_config_hash"}, by_alias=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def summarize(self) -> str:
        hash_short = self.units_config_hash[:6] if self.units_config_hash else "none"
        return (
            f"Base units: {self.base_units} | Spatial: {self.spatial_units} | Temporal: {self.temporal_units}\n"
            f"Convert on load: {'ON' if self.auto_convert_on_load else 'OFF'} | Normalize: {'ON' if self.normalize_to_internal else 'OFF'} | Validation: {self.unit_validation_policy}\n"
            f"Output: {self.preferred_output_units} | File: {self.preferred_output_units_file}\n"
            f"Assumed input: {self.input_assumed_units} | Hash: {hash_short}"
        )

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "UnitsSettings") -> "UnitsSettings":
        if values.base_units == "SI":
            if values.spatial_units != "m" or values.temporal_units not in {"s", "ns"}:
                raise ValueError("SI base units require spatial=m and temporal=s or ns")
        if values.base_units == "lab":
            if values.spatial_units != "cm" or values.temporal_units != "ns":
                raise ValueError("lab base units require spatial=cm and temporal=ns")
        allowed_keys = {"pressure", "temperature", "rho", "J", "T", "E", "B"}
        if values.unit_resolution_overrides:
            for k, v in values.unit_resolution_overrides.items():
                if k not in allowed_keys or v <= 0:
                    raise ValueError("invalid unit_resolution_overrides entry")
        known_units = {"m", "cm", "s", "ms", "us", "ns", "kA", "A", "MA", "V", "kV", "J", "keV", "eV", "T", "kV/cm"}
        for mapping in [values.preferred_output_units, values.preferred_output_units_file]:
            if mapping:
                for u in mapping.values():
                    if not any(u == ku or u.endswith(ku) for ku in known_units):
                        raise ValueError(f"unsupported unit '{u}' in preferred output units")
        values = values.model_copy(update={"units_config_hash": values.hash_units_config()})
        return values


__all__ = ["UnitsSettings"]
