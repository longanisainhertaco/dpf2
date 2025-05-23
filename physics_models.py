from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

from pydantic import ConfigDict, Field, root_validator

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

from pydantic import BaseModel
if not hasattr(BaseModel, "model_validate"):
    BaseModel.model_validate = classmethod(lambda cls, d, **_: cls.parse_obj(d))
if not hasattr(BaseModel, "model_dump"):
    BaseModel.model_dump = BaseModel.dict
if not hasattr(BaseModel, "model_dump_json"):
    BaseModel.model_dump_json = BaseModel.json
if not hasattr(BaseModel, "model_copy"):
    BaseModel.model_copy = BaseModel.copy

from core_schema import (
    ConfigSectionBase,
    EOSModel,
    ResistivityModel,
    IonizationModel,
    IonizationFallback,
    RadiationModel,
    RadiationTransportModel,
    LineEscapeMethod,
    RadiationGeometryModel,
    InstabilityModel,
)
from units_settings import UnitsSettings


def to_camel_case(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class PhysicsModels(ConfigSectionBase):
    """Validated physics configuration for DPF simulations."""

    config_section_id: ClassVar[str] = "physics"

    eos_model: EOSModel = Field(
        ..., alias="eosModel", metadata={"category": "PhysicsModels", "group": "EOS"}
    )
    gamma: float = Field(
        ...,
        alias="gamma",
        description="Specific heat ratio \u03b3",
        metadata={
            "units": "dimensionless",
            "category": "PhysicsModels",
            "group": "EOS",
        },
    )
    two_temperature_model_enabled: bool = Field(
        False,
        alias="twoTemperatureModelEnabled",
        metadata={"category": "PhysicsModels", "group": "EOS"},
    )
    neutral_fluid_enabled: bool = Field(
        False,
        alias="neutralFluidEnabled",
        metadata={"category": "PhysicsModels", "group": "NeutralFluid"},
    )
    initial_neutral_pressure_torr: Optional[float] = Field(
        None,
        alias="initialNeutralPressureTorr",
        metadata={"units": "Torr", "category": "PhysicsModels", "group": "NeutralFluid"},
    )
    enable_neutral_particle_tracking: bool = Field(
        False,
        alias="enableNeutralParticleTracking",
        metadata={"category": "PhysicsModels", "group": "NeutralFluid"},
    )

    resistivity_model: ResistivityModel = Field(
        ResistivityModel.SPITZER,
        alias="resistivityModel",
        metadata={"category": "PhysicsModels", "group": "Resistivity"},
    )

    ionization_model: IonizationModel = Field(
        IonizationModel.SAHA,
        alias="ionizationModel",
        metadata={"category": "PhysicsModels", "group": "Ionization"},
    )
    fallback_if_ionization_invalid: IonizationFallback = Field(
        IonizationFallback.RAISE,
        alias="fallbackIfIonizationInvalid",
        description="Behavior if ionization model fails",
        metadata={"category": "PhysicsModels", "group": "Ionization"},
    )
    ionization_table_path: Optional[Path] = Field(
        None,
        alias="ionizationTablePath",
        metadata={"category": "PhysicsModels", "group": "Ionization"},
    )

    radiation_model: RadiationModel = Field(
        RadiationModel.NONE,
        alias="radiationModel",
        metadata={"category": "PhysicsModels", "group": "Radiation"},
    )
    radiation_transport_model: RadiationTransportModel = Field(
        RadiationTransportModel.NONE,
        alias="radiationTransportModel",
        metadata={"category": "PhysicsModels", "group": "Radiation"},
    )
    line_escape_method: LineEscapeMethod = Field(
        LineEscapeMethod.EDDINGTON,
        alias="lineEscapeMethod",
        metadata={"category": "PhysicsModels", "group": "Radiation"},
    )
    radiation_geometry_model: RadiationGeometryModel = Field(
        RadiationGeometryModel.SLAB,
        alias="radiationGeometryModel",
        metadata={"category": "PhysicsModels", "group": "Radiation"},
    )
    sxr_bandpass_nm: Optional[Tuple[float, float]] = Field(
        None,
        alias="sxrBandpassNm",
        metadata={"units": "nm", "category": "PhysicsModels", "group": "Radiation"},
    )
    enable_photon_reabsorption: bool = Field(
        False,
        alias="enablePhotonReabsorption",
        metadata={"category": "PhysicsModels", "group": "Radiation"},
    )
    opacity_table_path: Optional[Path] = Field(
        None,
        alias="opacityTablePath",
        metadata={"category": "PhysicsModels", "group": "Radiation"},
    )

    hall_mhd_enabled: bool = Field(
        False,
        alias="hallMhdEnabled",
        metadata={"category": "PhysicsModels", "group": "Instabilities"},
    )
    pease_bragginski_limit_check: bool = Field(
        False,
        alias="peaseBragginskiLimitCheck",
        metadata={"category": "PhysicsModels", "group": "Instabilities"},
    )
    instability_models_enabled: List[InstabilityModel] = Field(
        default_factory=list,
        alias="instabilityModelsEnabled",
        metadata={"category": "PhysicsModels", "group": "Instabilities"},
    )
    instability_thresholds: Optional[Dict[str, float]] = Field(
        None,
        alias="instabilityThresholds",
        metadata={"category": "PhysicsModels", "group": "Instabilities"},
    )
    temperature_cutoff_min_keV: Optional[float] = Field(
        None,
        alias="temperatureCutoffMinKeV",
        metadata={"units": "keV", "category": "PhysicsModels", "group": "Instabilities"},
    )
    density_cutoff_max_gcc: Optional[float] = Field(
        None,
        alias="densityCutoffMaxGcc",
        metadata={"units": "g/cm\u00b3", "category": "PhysicsModels", "group": "Instabilities"},
    )

    eos_table_path: Optional[Path] = Field(
        None,
        alias="eosTablePath",
        metadata={"category": "PhysicsModels", "group": "EOS"},
    )
    eos_version: Optional[str] = Field(
        None,
        alias="eosVersion",
        metadata={"category": "PhysicsModels", "group": "EOS"},
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        allow_population_by_field_name=True,
        validate_default=True,
    )


    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "PhysicsModels":
        return cls(
            eos_model=EOSModel.IDEAL,
            gamma=1.4,
            two_temperature_model_enabled=False,
            neutral_fluid_enabled=False,
            enable_neutral_particle_tracking=False,
            resistivity_model=ResistivityModel.SPITZER,
            ionization_model=IonizationModel.SAHA,
            fallback_if_ionization_invalid=IonizationFallback.RAISE,
            radiation_model=RadiationModel.NONE,
            radiation_transport_model=RadiationTransportModel.NONE,
            line_escape_method=LineEscapeMethod.EDDINGTON,
            radiation_geometry_model=RadiationGeometryModel.SLAB,
            hall_mhd_enabled=False,
            pease_bragginski_limit_check=False,
            instability_models_enabled=[],
        )

    def required_fields(self) -> List[str]:
        return [n for n, f in self.model_fields.items() if f.is_required()]

    def resolve_defaults(self) -> "PhysicsModels":
        data = self.model_dump()
        return self.model_validate(data)

    def get_field_metadata(self) -> Dict[str, Dict[str, object]]:
        return {
            name: field.json_schema_extra or {}
            for name, field in self.model_fields.items()
        }

    def summarize(self) -> str:
        parts = [
            f"EOS: {self.eos_model.value} (\u03b3={self.gamma}), ionization: {self.ionization_model.value}",
            f"Resistivity: {self.resistivity_model.value}, Radiation: {self.radiation_model.value} ({self.radiation_transport_model.value})",
            f"Neutrals: {'enabled' if self.neutral_fluid_enabled else 'disabled'}, T_n0 = {self.initial_neutral_pressure_torr if self.initial_neutral_pressure_torr is not None else 'n/a'} Torr",
        ]
        if self.instability_models_enabled:
            pairs = []
            if self.instability_thresholds:
                for m in self.instability_models_enabled:
                    t = self.instability_thresholds.get(m.value)
                    if t is not None:
                        pairs.append(f"{m.value} ({t})")
                    else:
                        pairs.append(m.value)
            else:
                pairs = [m.value for m in self.instability_models_enabled]
            parts.append(f"Instabilities: {', '.join(pairs)}")
        else:
            parts.append("Instabilities: none")
        return "\n".join(parts)

    def normalize_units(self, units: UnitsSettings) -> "PhysicsModels":
        unit_map = units.normalize_units()
        scale = unit_map.get("spatial", 1.0)
        band = self.sxr_bandpass_nm
        if band is not None:
            band = (band[0] * scale, band[1] * scale)
        return self.model_copy(update={"sxr_bandpass_nm": band})

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "PhysicsModels") -> "PhysicsModels":
        if values.radiation_model is not RadiationModel.NONE and values.sxr_bandpass_nm is None:
            raise ValueError("sxr_bandpass_nm must be set when radiation_model is not 'None'")
        if values.neutral_fluid_enabled and values.initial_neutral_pressure_torr is None:
            raise ValueError("initial_neutral_pressure_torr must be set when neutral_fluid_enabled is True")
        if values.instability_models_enabled:
            if values.instability_thresholds is None:
                raise ValueError("instability_thresholds must be provided when instability_models_enabled is set")
            missing = [m.value for m in values.instability_models_enabled if m.value not in values.instability_thresholds]
            if missing:
                raise ValueError(f"instability_thresholds missing for: {', '.join(missing)}")
        if values.radiation_transport_model is RadiationTransportModel.MONTE_CARLO and values.opacity_table_path is None:
            raise ValueError("opacity_table_path is required for MonteCarlo radiation transport")
        if values.eos_model is EOSModel.TABULATED and values.eos_table_path is None:
            raise ValueError("eos_table_path must be provided when eos_model is 'tabulated'")

        # Context-based validation omitted for compatibility with Pydantic v1
        return values


__all__ = ["PhysicsModels"]
