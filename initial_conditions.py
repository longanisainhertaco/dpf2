"""Initial conditions schema for DPF simulations.

Example YAML configuration::

    initial:
      temperature: 15000
      density: 1e-4
      gasType: D2
      sheathType: gaussian
      preionizationMethod: UV
      preionizationIntensity: 5e14
      preionizationDurationNs: 50
      preionizationProfile: gaussian
      enableDynamicIonizationRate: true
      sheathVelocityProfile:
        - [0.0, 1.0]
        - [1.0, 2.0]
      currentProfile:
        - [0.0, 10.0]
        - [1.0, 15.0]
      customGasProperties:
        molar_mass: 2.0
        Z: 1
        ionization_energy: 13.6
      breakdownModel:
        type: FN
        fieldThreshold: 5.0
        breakdownDelay: 10
      paschenModel:
        insulatorGapCm: 0.8
        gasPressureTorr: 1.2
        material: alumina
        paschenCurveModel: empirical
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal

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


# --- helpers -----------------------------------------------------------------

def to_camel_case(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class ConfigSectionBase(BaseModel):
    """Common base class providing utility helpers."""

    @classmethod
    def with_defaults(cls):
        data = {}
        for name, field in cls.model_fields.items():
            if field.default_factory is not None:
                data[name] = field.default_factory()
            elif field.default is not None:
                data[name] = field.default
        return cls(**data)

    def resolve_defaults(self):
        return self

    @classmethod
    def required_fields(cls) -> List[str]:
        return [name for name, fld in cls.model_fields.items() if fld.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        meta: Dict[str, Dict[str, Any]] = {}
        for name, fld in cls.model_fields.items():
            meta[name] = fld.json_schema_extra or {}
        return meta


TimeVoltageProfile = List[Tuple[float, float]]


# --- nested models -----------------------------------------------------------

class BreakdownModel(ConfigSectionBase):
    config_section_id: ClassVar[Literal["breakdown"]] = "breakdown"

    type: Literal["FN", "CL", "stochastic_delay", "field_threshold"]
    field_threshold: Optional[float] = Field(
        None, json_schema_extra={"units": "kV/cm", "category": "InitialConditions", "group": "Breakdown"}
    )
    breakdown_delay: Optional[float] = Field(
        None, json_schema_extra={"units": "ns", "category": "InitialConditions", "group": "Breakdown"}
    )
    stochastic_seed: Optional[int] = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def with_defaults(cls):
        return cls(type="field_threshold")

    @model_validator(mode="after")
    def check_threshold(cls, values: "BreakdownModel") -> "BreakdownModel":
        if values.type in {"FN", "CL"} and values.field_threshold is None:
            raise ValueError("field_threshold must be provided for FN or CL type")
        return values


class PaschenModel(ConfigSectionBase):
    config_section_id: ClassVar[Literal["paschen"]] = "paschen"

    insulator_gap_cm: float = Field(
        ..., json_schema_extra={"units": "cm", "category": "InitialConditions", "group": "Breakdown"}
    )
    gas_pressure_torr: float = Field(
        ..., json_schema_extra={"units": "Torr", "category": "InitialConditions", "group": "Breakdown"}
    )
    material: str
    paschen_curve_model: Literal["empirical", "semi-empirical", "analytic", "tabulated"]
    paschen_data_path: Optional[Path] = None
    paschen_curve_data_version: Optional[str] = None
    knee_voltage_override: Optional[float] = Field(
        None, json_schema_extra={"units": "kV", "category": "InitialConditions", "group": "Breakdown"}
    )

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def with_defaults(cls):
        return cls(
            insulator_gap_cm=1.0,
            gas_pressure_torr=10.0,
            material="Pyrex",
            paschen_curve_model="empirical",
        )

    @model_validator(mode="after")
    def check_paschen(cls, values: "PaschenModel") -> "PaschenModel":
        if values.paschen_curve_model == "tabulated" and values.paschen_data_path is None:
            raise ValueError("paschen_data_path must be provided when paschen_curve_model is 'tabulated'")
        return values


# --- main model --------------------------------------------------------------

class InitialConditions(ConfigSectionBase):
    config_section_id: ClassVar[Literal["initial"]] = "initial"

    temperature: float = Field(
        ..., json_schema_extra={"units": "K", "category": "InitialConditions", "group": "Thermal"}
    )
    density: float = Field(
        ..., json_schema_extra={"units": "kg/m³", "category": "InitialConditions", "group": "Thermal"}
    )
    gas_type: Literal["D2", "DT", "He", "Ne", "Ar", "Xe"] = Field(
        ..., alias="gasType", json_schema_extra={"category": "InitialConditions", "group": "Gas"}
    )
    sheath_type: Literal["slab", "gaussian"] = Field(
        ..., alias="sheathType", json_schema_extra={"category": "InitialConditions", "group": "Gas"}
    )

    sheath_velocity_profile: TimeVoltageProfile = Field(
        ..., alias="sheathVelocityProfile", json_schema_extra={"units": ["cm", "cm/us"], "category": "InitialConditions", "group": "Sheath"}
    )
    current_profile: TimeVoltageProfile = Field(
        ..., alias="currentProfile", json_schema_extra={"units": ["us", "kA"], "category": "InitialConditions", "group": "Sheath"}
    )

    preionization_method: Optional[Literal["UV", "beta_source", "external_discharge"]] = Field(
        None, alias="preionizationMethod"
    )
    preionization_intensity: Optional[float] = Field(
        None, alias="preionizationIntensity", json_schema_extra={"units": "photons/cm³/s", "category": "InitialConditions", "group": "Gas"}
    )
    preionization_duration_ns: Optional[float] = Field(
        None, alias="preionizationDurationNs", json_schema_extra={"units": "ns", "category": "InitialConditions", "group": "Gas"}
    )
    preionization_profile: Optional[Literal["uniform", "gaussian", "localized"]] = Field(
        None, alias="preionizationProfile"
    )
    enable_dynamic_ionization_rate: bool = Field(
        ..., alias="enableDynamicIonizationRate"
    )

    custom_gas_properties: Optional[Dict[str, float]] = Field(
        None, alias="customGasProperties", json_schema_extra={"category": "InitialConditions", "group": "Gas"}
    )

    breakdown_model: BreakdownModel = Field(
        ..., alias="breakdownModel", json_schema_extra={"category": "InitialConditions", "group": "Breakdown"}
    )
    paschen_model: PaschenModel = Field(
        ..., alias="paschenModel", json_schema_extra={"category": "InitialConditions", "group": "Breakdown"}
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        allow_population_by_field_name=True,
    )

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        alias_generator = to_camel_case

    @classmethod
    def with_defaults(cls):
        return cls(
            temperature=15000.0,
            density=1e-4,
            gas_type="D2",
            sheath_type="slab",
            sheath_velocity_profile=[(0.0, 0.0)],
            current_profile=[(0.0, 1.0)],
            enable_dynamic_ionization_rate=False,
            breakdown_model=BreakdownModel.with_defaults(),
            paschen_model=PaschenModel.with_defaults(),
        )

    @model_validator(mode="after")
    def validate_logic(cls, values: "InitialConditions") -> "InitialConditions":
        if values.gas_type in {"D2", "DT"} and values.temperature <= 1e4:
            raise ValueError("temperature must be > 1e4 K for deuterium gases")
        if values.preionization_method is not None:
            if values.preionization_intensity is None or values.preionization_duration_ns is None:
                raise ValueError(
                    "preionization_intensity and preionization_duration_ns required when method set"
                )
        if not values.current_profile or max(v for _, v in values.current_profile) <= 0:
            raise ValueError("current_profile must be non-empty with positive peak")
        return values


__all__ = ["InitialConditions", "BreakdownModel", "PaschenModel"]

