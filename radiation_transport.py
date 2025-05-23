from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, root_validator


# ---------------------------------------------------------------------------
# Compatibility helpers mirroring pydantic v2 model_validator

def model_validator(*, mode: str = "after"):
    def decorator(func):
        if mode == "after":
            def wrapper(cls, values):
                if isinstance(values, cls):
                    inst = values
                else:
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
from units_settings import UnitsSettings


class RadiationTransport(ConfigSectionBase):
    """Validated radiation transport configuration."""

    config_section_id: ClassVar[Literal["radiation_transport"]] = "radiation_transport"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Transport control
    transport_model: Literal["none", "FLD", "M1", "MonteCarlo", "RayTrace"] = "none"
    optical_depth_model: Literal["thin", "gray", "multi-group", "tabulated"] = "thin"
    radiation_coupling_mode: Optional[Literal["separate", "coupled"]] = "coupled"
    angular_quadrature_order: Optional[int] = Field(4, ge=2, le=16)
    time_stepping: Literal["explicit", "implicit", "semi_implicit"] = "explicit"
    enable_emission_sources: bool = True
    enable_absorption_sinks: bool = True
    max_radiation_iterations: Optional[int] = 20
    raytrace_camera_position: Optional[Tuple[float, float, float]] = None

    # Spectral modeling
    frequency_group_edges_eV: Optional[List[float]] = None
    xray_energy_units: Literal["eV", "keV", "MeV"] = "keV"

    # Closure & scattering
    closure_method: Optional[Literal["none", "Eddington", "VET", "M1"]] = "Eddington"
    include_scattering: bool = False
    scattering_model: Optional[Literal["Thomson", "Compton", "Mie"]] = None

    # Opacity modeling
    opacity_source: Literal["constant", "FLYCHK", "OpenADAS", "tabulated"] = "FLYCHK"
    opacity_species: List[str] = Field(default_factory=lambda: ["D", "Ar"])
    opacity_table_path: Optional[Path] = None
    opacity_units: Optional[Dict[str, str]] = None

    # Emissivity modeling
    emissivity_model: Literal["none", "LTE", "nonLTE", "custom"] = "LTE"
    emissivity_scaling_factor: Optional[float] = 1.0
    include_bremsstrahlung: bool = True
    include_line_radiation: bool = True
    include_recombination: bool = True
    emissivity_source: Optional[Literal["tabulated", "LTE", "custom"]] = "LTE"
    emissivity_table_path: Optional[Path] = None

    # Hash
    radiation_transport_config_hash: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "RadiationTransport":
        return cls(
            transport_model="FLD",
            closure_method="Eddington",
            radiation_coupling_mode="coupled",
        )

    def resolve_defaults(self) -> "RadiationTransport":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or f.metadata or {}) for n, f in cls.model_fields.items()}

    def summarize(self) -> str:
        mode = f"{self.transport_model}"
        step = self.time_stepping.capitalize()
        groups = self.frequency_group_edges_eV
        groups_str = (
            f"[{', '.join(str(int(g)) for g in groups)}] eV" if groups else "n/a"
        )
        scat = (
            f"{self.scattering_model} (ON)" if self.include_scattering and self.scattering_model else "OFF"
        )
        emiss_parts = []
        if self.include_bremsstrahlung:
            emiss_parts.append("Brehm")
        if self.include_line_radiation:
            emiss_parts.append("Line")
        if self.include_recombination:
            emiss_parts.append("Recomb")
        emiss = " + ".join(emiss_parts)
        hash_short = self.radiation_transport_config_hash[:6] if self.radiation_transport_config_hash else "none"
        return (
            f"Radiation: {mode} ({step}), Closure = {self.closure_method}\n"
            f"Groups: {groups_str} | Scattering: {scat}\n"
            f"Opacities: {self.opacity_source} → {self.opacity_species}, Emissivity: {self.emissivity_model} + {emiss}"
        )

    def normalize_units(self, units: UnitsSettings) -> "RadiationTransport":
        unit_map = units.normalize_units()
        scale = unit_map.get("spatial", 1.0)
        cam = None
        if self.raytrace_camera_position is not None:
            cam = tuple(c * scale for c in self.raytrace_camera_position)
        freq = None
        if self.frequency_group_edges_eV is not None:
            e_scale = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3}.get(self.xray_energy_units, 1.0)
            freq = [f * e_scale for f in self.frequency_group_edges_eV]
        return self.model_copy(update={
            "raytrace_camera_position": cam,
            "frequency_group_edges_eV": freq,
            "xray_energy_units": "keV",
        })

    # ------------------------------------------------------------------
    def hash_radiation_transport_config(self) -> str:
        data = self.model_dump(exclude={"radiation_transport_config_hash"}, by_alias=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "RadiationTransport") -> "RadiationTransport":
        if values.transport_model == "none" and values.closure_method != "none":
            raise ValueError("closure_method must be 'none' when transport_model is 'none'")
        if values.include_scattering and values.scattering_model is None:
            raise ValueError("scattering_model must be set when include_scattering is True")
        if values.optical_depth_model == "multi-group":
            if not values.frequency_group_edges_eV:
                raise ValueError("frequency_group_edges_eV must be defined for multi-group model")
            if sorted(values.frequency_group_edges_eV) != values.frequency_group_edges_eV:
                raise ValueError("frequency_group_edges_eV must be monotonically increasing")
        if values.opacity_source == "tabulated" and values.opacity_table_path is None:
            raise ValueError("opacity_table_path required when opacity_source is 'tabulated'")
        if values.raytrace_camera_position is not None:
            if len(values.raytrace_camera_position) != 3:
                raise ValueError("raytrace_camera_position must be a 3-tuple")
            if values.transport_model != "RayTrace":
                raise ValueError("raytrace_camera_position requires transport_model='RayTrace'")
        if not values.opacity_species:
            raise ValueError("opacity_species must contain at least one species")
        if len(values.opacity_species) != len(set(values.opacity_species)):
            raise ValueError("opacity_species must be unique")
        values = values.model_copy(update={
            "radiation_transport_config_hash": values.hash_radiation_transport_config()
        })
        return values

    # ------------------------------------------------------------------
    @classmethod
    def model_validate(cls, data: Any, **kwargs) -> "RadiationTransport":
        obj = super().model_validate(data)
        obj = cls.check_rules(obj)
        return obj


__all__ = ["RadiationTransport"]
