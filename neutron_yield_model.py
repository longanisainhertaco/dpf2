from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, ConfigDict, Field, root_validator

# ---------------------------------------------------------------------------
# Compatibility helper mirroring pydantic v2 model_validator

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
from units_settings import UnitsSettings


class NeutronYieldModel(ConfigSectionBase):
    """Configuration for neutron yield modeling in DPF simulations."""

    config_section_id: ClassVar[Literal["neutron_yield"]] = "neutron_yield"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Core fusion toggles
    fusion_fuel_type: Literal["DD", "DT"] = "DD"
    beam_target_model_enabled: bool = True
    thermonuclear_model_enabled: bool = True
    separate_yield_components: bool = True
    yield_integration_window_us: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------
    # Beam-target fusion configuration
    beam_ion_species: str
    target_density_source: Literal["diagnostics", "constant", "user_file"] = "diagnostics"
    target_density_constant: Optional[float] = Field(None, metadata={"units": "cm^-3"})
    iedf_source: Literal["diagnostics", "user_file", "synthetic_gaussian"] = "diagnostics"
    iedf_user_path: Optional[Path] = None
    iedf_format: Optional[Literal["csv", "OpenPMD", "json"]] = "csv"
    fusion_cross_section_model: Literal["Bosch-Hale", "EXFOR", "tabulated"] = "Bosch-Hale"
    cross_section_table_path: Optional[Path] = None
    cross_section_table_units: Optional[Dict[str, str]] = {"energy": "MeV", "sigma": "barn"}

    # ------------------------------------------------------------------
    # Thermonuclear fusion configuration
    reactivity_source: Literal["look-up", "analytic", "FLYCHK"] = "look-up"
    maxwellian_assumed: bool = True
    average_ion_temperature_keV: Optional[float] = None
    average_ion_density_cm3: Optional[float] = None
    dd_branching_ratio: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    reactivity_table_path: Optional[Path] = None
    reactivity_table_units: Optional[Dict[str, str]] = {"Ti": "keV", "reactivity": "cm^3/s"}

    # ------------------------------------------------------------------
    # Spectrum output and detector modeling
    neutron_spectrum_output_enabled: bool = True
    spectrum_energy_bins_MeV: Optional[List[float]] = None
    spectrum_output_format: Optional[Literal["csv", "OpenPMD", "plot"]] = "csv"
    apply_detector_response_function: bool = False
    detector_response_file: Optional[Path] = None
    detector_response_normalization: Optional[Literal["none", "area", "peak", "custom"]] = "none"

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "NeutronYieldModel":
        return cls(beam_ion_species="D+")

    def resolve_defaults(self) -> "NeutronYieldModel":
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [n for n, f in self.model_fields.items() if f.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or f.metadata or {}) for n, f in self.model_fields.items()}

    def normalize_units(self, units: UnitsSettings) -> "NeutronYieldModel":
        unit_map = units.normalize_units()
        scale_t = unit_map.get("temporal", 1.0)
        win = None
        if self.yield_integration_window_us is not None:
            win = (
                self.yield_integration_window_us[0] * scale_t,
                self.yield_integration_window_us[1] * scale_t,
            )
        return self.model_copy(update={"yield_integration_window_us": win})

    def summarize(self) -> str:
        fuel = self.fusion_fuel_type
        beam = "ON" if self.beam_target_model_enabled else "OFF"
        th = "ON" if self.thermonuclear_model_enabled else "OFF"
        ion = self.beam_ion_species
        sigma = self.fusion_cross_section_model
        ti = (
            str(self.average_ion_temperature_keV)
            if self.average_ion_temperature_keV is not None
            else "n/a"
        )
        spec_bins = (
            f"[{', '.join(str(b).rstrip('0').rstrip('.') for b in self.spectrum_energy_bins_MeV)}]"
            if self.spectrum_energy_bins_MeV
            else "None"
        )
        fmt = self.spectrum_output_format or "n/a"
        parts = [
            f"Fusion: {fuel} | Beam-target: {beam}, Ion: {ion}, σ(E): {sigma}",
            f"Thermonuclear: {th}, Maxwellian = {self.maxwellian_assumed}, Ti = {ti} keV",
            f"Branching: DDn = {self.dd_branching_ratio} | Spectrum: {spec_bins} MeV → {fmt}",
        ]
        if self.apply_detector_response_function:
            resp = self.detector_response_file.name if self.detector_response_file else "none"
            parts.append(
                f"Detector: applied, Norm = {self.detector_response_normalization}, Response = {resp}"
            )
        else:
            parts.append("Detector: none")
        return "\n".join(parts)

    def hash_neutron_yield_config(self) -> str:
        data = self.model_dump(by_alias=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "NeutronYieldModel") -> "NeutronYieldModel":
        if (
            values.thermonuclear_model_enabled
            and values.reactivity_source == "analytic"
        ):
            if (
                values.average_ion_temperature_keV is None
                or values.average_ion_density_cm3 is None
            ):
                raise ValueError(
                    "average_ion_temperature_keV and average_ion_density_cm3 required for analytic reactivity"
                )

        if (
            values.reactivity_source in {"look-up", "FLYCHK"}
            and values.thermonuclear_model_enabled
            and values.reactivity_table_path is None
        ):
            raise ValueError("reactivity_table_path required for table-based reactivity")

        if (
            values.fusion_cross_section_model == "tabulated"
            and values.cross_section_table_path is None
        ):
            raise ValueError("cross_section_table_path required for tabulated cross sections")

        if values.apply_detector_response_function and values.detector_response_file is None:
            raise ValueError("detector_response_file required when apply_detector_response_function is True")

        if values.spectrum_energy_bins_MeV is not None:
            if values.spectrum_energy_bins_MeV != sorted(values.spectrum_energy_bins_MeV):
                raise ValueError("spectrum_energy_bins_MeV must be monotonically increasing")

        if values.yield_integration_window_us is not None:
            s, e = values.yield_integration_window_us
            if s >= e:
                raise ValueError("yield_integration_window_us must have start < end")

        return values


__all__ = ["NeutronYieldModel"]
