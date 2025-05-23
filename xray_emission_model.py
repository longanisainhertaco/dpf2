"""X-ray emission configuration model for DPF simulations."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Literal, Self

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
from units_settings import UnitsSettings


# ---------------------------------------------------------------------------


class XrayEmissionModel(ConfigSectionBase):
    """Validated X-ray emission configuration."""

    config_section_id: ClassVar[Literal["xray_emission"]] = "xray_emission"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # -- Emission toggles -------------------------------------------------
    thermal_bremsstrahlung_enabled: bool = True
    nonthermal_bremsstrahlung_enabled: bool = True
    line_radiation_enabled: bool = True
    recombination_radiation_enabled: bool = True
    plasma_broadening_enabled: bool = True
    include_ipd_effects: bool = False
    include_stark_broadening: bool = True

    # -- Atomic database --------------------------------------------------
    atomic_data_source: Literal["FLYCHK", "NIST", "OpenADAS", "custom"] = "FLYCHK"
    ion_species: List[str] = Field(default_factory=lambda: ["D+", "Ar"])
    ionization_stages: Optional[Dict[str, List[int]]] = None

    # -- Spectrum configuration ------------------------------------------
    xray_energy_bins: Optional[List[float]] = None
    xray_energy_units: Literal["eV", "keV", "MeV"] = "keV"
    spectrum_resolution_keV: Optional[float] = None
    use_fixed_bins: bool = True
    time_resolved_spectrum_enabled: bool = False
    spectrum_time_bins_us: Optional[List[float]] = None

    # -- Detector filters -------------------------------------------------
    apply_detector_filter: bool = False
    xray_detector_filter_path: Optional[Path] = None
    detector_filter_normalization: Optional[Literal["area", "peak", "none"]] = "area"
    custom_emission_mask_path: Optional[Path] = None
    emission_volume_specification: Literal[
        "pinch", "entire_domain", "custom_mask"
    ] = "pinch"

    # -- Nonthermal options ----------------------------------------------
    nonthermal_source_region: Literal["bulk", "target", "both"] = "target"
    beam_energy_distribution_source: Literal[
        "diagnostics", "file", "synthetic"
    ] = "diagnostics"
    beam_distribution_file: Optional[Path] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "XrayEmissionModel":
        return cls()

    def resolve_defaults(self) -> "XrayEmissionModel":
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [n for n, f in self.model_fields.items() if f.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {name: field.json_schema_extra or {} for name, field in self.model_fields.items()}

    def normalize_units(self, units: UnitsSettings) -> "XrayEmissionModel":
        unit_map = units.normalize_units()
        # Energy scaling -------------------------------------------------
        e_scale_map = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3}
        e_scale = e_scale_map.get(self.xray_energy_units, 1.0)
        bins = None
        if self.xray_energy_bins is not None:
            bins = [b * e_scale for b in self.xray_energy_bins]
        res = self.spectrum_resolution_keV
        if res is not None:
            res = res * e_scale
        # Time scaling ---------------------------------------------------
        t_scale = unit_map.get("temporal", 1.0)
        tbins = None
        if self.spectrum_time_bins_us is not None:
            tbins = [t * 1e-6 * t_scale for t in self.spectrum_time_bins_us]
        return self.model_copy(
            update={
                "xray_energy_bins": bins,
                "spectrum_resolution_keV": res,
                "xray_energy_units": "keV",
                "spectrum_time_bins_us": tbins,
            }
        )

    def summarize(self) -> str:
        therm = "Thermal" if self.thermal_bremsstrahlung_enabled else ""
        nontherm = "Nonthermal" if self.nonthermal_bremsstrahlung_enabled else ""
        line = "Line" if self.line_radiation_enabled else ""
        recom = "Recom" if self.recombination_radiation_enabled else ""
        channels = " + ".join([p for p in [therm, nontherm, line, recom] if p])
        filt = (
            f"{self.xray_detector_filter_path.name} ({self.detector_filter_normalization})"
            if self.apply_detector_filter and self.xray_detector_filter_path
            else "none"
        )
        tbins = (
            len(self.spectrum_time_bins_us) if self.spectrum_time_bins_us else 0
        )
        resolved = "ON" if self.time_resolved_spectrum_enabled else "OFF"
        bins_str = (
            f"{self.xray_energy_bins} keV"
            if self.xray_energy_bins is not None
            else "adaptive"
        )
        return (
            f"X-ray: {channels}\n"
            f"Species: {self.ion_species}, DB = {self.atomic_data_source}\n"
            f"Spectrum bins: {bins_str}, Time bins: {tbins}, Resolved: {resolved}\n"
            f"Emission: {self.emission_volume_specification}, Filter: {filt}\n"
            f"Beam dist: {self.beam_energy_distribution_source}, Source: {self.nonthermal_source_region}"
        )

    # ------------------------------------------------------------------
    def hash_xray_emission_config(self) -> str:
        data = self.model_dump(by_alias=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "XrayEmissionModel") -> "XrayEmissionModel":
        # Spectrum bins validations
        if values.use_fixed_bins and values.xray_energy_bins is not None:
            if sorted(values.xray_energy_bins) != values.xray_energy_bins:
                raise ValueError("xray_energy_bins must be monotonically increasing")
        if values.time_resolved_spectrum_enabled:
            if not values.spectrum_time_bins_us:
                raise ValueError("spectrum_time_bins_us required when time_resolved_spectrum_enabled")
            if sorted(values.spectrum_time_bins_us) != values.spectrum_time_bins_us:
                raise ValueError("spectrum_time_bins_us must be increasing")
        # File checks
        if values.apply_detector_filter:
            if not values.xray_detector_filter_path or not Path(values.xray_detector_filter_path).exists():
                raise ValueError("xray_detector_filter_path must exist when apply_detector_filter is True")
        if values.emission_volume_specification == "custom_mask":
            if not values.custom_emission_mask_path or not Path(values.custom_emission_mask_path).exists():
                raise ValueError("custom_emission_mask_path must exist when using custom_mask emission volume")
        if values.beam_energy_distribution_source == "file":
            if not values.beam_distribution_file or not Path(values.beam_distribution_file).exists():
                raise ValueError("beam_distribution_file must exist when beam_energy_distribution_source is 'file'")
        # Species database check
        known = {"D+", "Ar", "Ne", "C", "O", "N", "Xe", "Kr", "Al", "Fe", "Cu"}
        if values.atomic_data_source != "custom":
            for sp in values.ion_species:
                if sp not in known:
                    warnings.warn(
                        f"Unknown ion species '{sp}' for database {values.atomic_data_source}",
                        UserWarning,
                    )
        if values.ionization_stages:
            for key in values.ionization_stages:
                if key not in values.ion_species:
                    raise ValueError("ionization_stages keys must match ion_species")
        return values


__all__ = ["XrayEmissionModel"]
