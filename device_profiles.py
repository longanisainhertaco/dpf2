"""Validated schema for Dense Plasma Focus device profiles."""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, ConfigDict, Field, root_validator


def model_validator(*, mode: str = "after"):
    """Compatibility helper mirroring pydantic.v2 model_validator."""
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


class DeviceEntry(BaseModel):
    """Single device entry describing geometry and circuit parameters."""

    device_label: str = Field(..., alias="deviceLabel")
    device_description: Optional[str] = Field(None, alias="deviceDescription")
    energy_kJ: float = Field(..., alias="energyKJ", metadata={"units": "kJ"})
    working_gas: str = Field(..., alias="workingGas")
    capacitor_bank: Dict[str, float] = Field(..., alias="capacitorBank")
    anode_radius_cm: float = Field(..., alias="anodeRadiusCm", ge=0.0)
    cathode_radius_cm: float = Field(..., alias="cathodeRadiusCm", ge=0.0)
    anode_length_cm: float = Field(..., alias="anodeLengthCm", ge=0.0)
    insulator_length_cm: float = Field(..., alias="insulatorLengthCm", ge=0.0)
    insulator_material: Optional[str] = Field(None, alias="insulatorMaterial")
    breakdown_voltage_kV: Optional[float] = Field(
        None, alias="breakdownVoltageKV", ge=0.0
    )
    fuel_mixture: Optional[Dict[str, float]] = Field(None, alias="fuelMixture")
    regime_category: Optional[Literal["low_energy", "medium_energy", "MJ_scale", "custom"]] = Field(
        "medium_energy", alias="regimeCategory"
    )
    reference_shot_ids: Optional[List[str]] = Field(None, alias="referenceShotIds")
    primary_observables: Optional[List[str]] = Field(None, alias="primaryObservables")
    diagnostic_capabilities: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, alias="diagnosticCapabilities"
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "DeviceEntry") -> "DeviceEntry":
        if values.fuel_mixture:
            total = sum(values.fuel_mixture.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError("fuel_mixture fractions must sum to 1.0 ±0.01")
        missing = {"C", "L", "R"} - set(values.capacitor_bank.keys())
        if missing:
            warnings.warn(f"capacitor bank missing fields: {sorted(missing)}")
        geom = [
            values.anode_radius_cm,
            values.cathode_radius_cm,
            values.anode_length_cm,
            values.insulator_length_cm,
        ]
        if any(g <= 0 for g in geom):
            raise ValueError("geometry dimensions must be positive")
        return values


class DeviceProfiles(ConfigSectionBase):
    """Repository of known DPF machine configurations."""

    config_section_id: ClassVar[Literal["device_profiles"]] = "device_profiles"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    devices: Dict[str, DeviceEntry] = Field(default_factory=dict)
    default_device_id: Optional[str] = Field(None, alias="defaultDeviceId")
    device_profiles_config_hash: Optional[str] = Field(
        None, alias="deviceProfilesConfigHash"
    )

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "DeviceProfiles":
        data = {
            "defaultDeviceId": "PF1000",
            "devices": {
                "PF1000": {
                    "deviceLabel": "PF1000",
                    "deviceDescription": "High-energy MJ-scale DPF at IFJ Krakow",
                    "energyKJ": 500.0,
                    "workingGas": "D",
                    "capacitorBank": {"C": 30e-6, "L": 15e-9, "R": 0.015},
                    "anodeRadiusCm": 2.5,
                    "cathodeRadiusCm": 6.0,
                    "anodeLengthCm": 16.0,
                    "insulatorLengthCm": 5.0,
                    "insulatorMaterial": "alumina",
                    "breakdownVoltageKV": 25.0,
                    "fuelMixture": {"D": 0.9, "Ar": 0.1},
                    "regimeCategory": "MJ_scale",
                    "referenceShotIds": ["2053", "2071", "2089"],
                    "primaryObservables": ["I(t)", "Yn", "SXR"],
                    "diagnosticCapabilities": {
                        "TOF": {"channels": 4, "resolutionNs": 1.0},
                        "SXR": {"filters": ["Al", "Ti"]},
                    },
                }
            },
        }
        return cls.model_validate(data)

    def resolve_defaults(self) -> "DeviceProfiles":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {
            n: (f.json_schema_extra or f.metadata or {})
            for n, f in cls.model_fields.items()
        }

    def normalize_units(self, spatial_units: Literal["cm", "m"]) -> "DeviceProfiles":
        factor = 0.01 if spatial_units == "m" else 1.0
        scaled: Dict[str, DeviceEntry] = {}
        for k, d in self.devices.items():
            scaled[k] = d.model_copy(
                update={
                    "anode_radius_cm": d.anode_radius_cm * factor,
                    "cathode_radius_cm": d.cathode_radius_cm * factor,
                    "anode_length_cm": d.anode_length_cm * factor,
                    "insulator_length_cm": d.insulator_length_cm * factor,
                }
            )
        return self.model_copy(update={"devices": scaled})

    def summarize(self) -> str:
        if not self.default_device_id or self.default_device_id not in self.devices:
            return "Devices: none"
        d = self.devices[self.default_device_id]
        mix = (
            "+".join(d.fuel_mixture.keys())
            if d.fuel_mixture
            else d.working_gas
        )
        refs = ", ".join(d.reference_shot_ids[:2]) if d.reference_shot_ids else "n/a"
        obs = ", ".join(d.primary_observables[:2]) if d.primary_observables else "n/a"
        vbr = d.breakdown_voltage_kV if d.breakdown_voltage_kV is not None else "n/a"
        lines = [
            "Devices:",
            f"  {d.device_label} ({d.energy_kJ:g} kJ, {mix}) | {d.regime_category}",
            f"  Geometry: a={d.anode_radius_cm:g} cm, L={d.anode_length_cm:g} cm | Vbr = {vbr} kV",
            f"  Observables: [{obs}], Ref: {refs}",
        ]
        return "\n".join(lines)

    def hash_device_profiles_config(self) -> str:
        data = self.model_dump(by_alias=True, exclude={"deviceProfilesConfigHash"})
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "DeviceProfiles") -> "DeviceProfiles":
        if values.default_device_id and values.default_device_id not in values.devices:
            raise ValueError("default_device_id must reference an existing device")
        values = values.model_copy(
            update={"device_profiles_config_hash": values.hash_device_profiles_config()}
        )
        return values


__all__ = ["DeviceProfiles", "DeviceEntry"]
