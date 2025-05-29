from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Literal

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
from units_settings import UnitsSettings


class SyntheticInstrument(BaseModel):
    """Per-instrument overrides for synthetic diagnostics."""

    response_file: Optional[Path] = None
    noise_model: Optional[str] = None
    geometry: Optional[str] = None
    sampling_override_ns: Optional[float] = Field(
        None, alias="samplingOverrideNs", metadata={"units": "ns"}
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )


class SyntheticDiagnostics(ConfigSectionBase):
    """Synthetic diagnostic modeling configuration."""

    config_section_id: ClassVar[Literal["synthetic_diagnostics"]] = "synthetic_diagnostics"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Global output configuration
    output_dir: str = "synthetic_diagnostics/"
    output_format: Literal["csv", "hdf5", "ascii"] = "csv"
    sampling_interval_ns: float = Field(1.0, alias="samplingIntervalNs", metadata={"units": "ns"})
    runtime_synthetic_enabled: bool = Field(True, alias="runtimeSyntheticEnabled")
    postprocessing_only: bool = Field(False, alias="postprocessingOnly")
    synthetic_diagnostics_config_hash: Optional[str] = Field(
        None, alias="syntheticDiagnosticsConfigHash"
    )

    # Detector control flags
    apply_time_response: bool = Field(True, alias="applyTimeResponse")
    apply_energy_filter: bool = Field(True, alias="applyEnergyFilter")
    apply_spatial_psf: bool = Field(False, alias="applySpatialPsf")

    # Enabled diagnostics
    synthetic_current_waveform_enabled: bool = Field(True, alias="syntheticCurrentWaveformEnabled")
    synthetic_voltage_waveform_enabled: bool = Field(True, alias="syntheticVoltageWaveformEnabled")
    synthetic_rogowski_signal_enabled: bool = Field(True, alias="syntheticRogowskiSignalEnabled")
    synthetic_bdot_signal_enabled: bool = Field(True, alias="syntheticBdotSignalEnabled")
    synthetic_neutron_tof_enabled: bool = Field(True, alias="syntheticNeutronTofEnabled")
    synthetic_xray_pinhole_enabled: bool = Field(True, alias="syntheticXrayPinholeEnabled")
    synthetic_thomson_parabola_enabled: bool = Field(False, alias="syntheticThomsonParabolaEnabled")
    synthetic_optical_interferogram_enabled: bool = Field(False, alias="syntheticOpticalInterferogramEnabled")

    # Diagnostic classification and labeling
    detector_ids: Optional[List[str]] = Field(None, alias="detectorIds")
    diagnostic_output_type: Dict[str, Literal["time_series", "spatial_map", "image"]] = Field(
        default_factory=dict, alias="diagnosticOutputType"
    )
    detector_positions_path: Optional[Path] = Field(None, alias="detectorPositionsPath")
    diagnostic_geometry_model: Optional[Literal["1D", "2D", "3D", "raycast"]] = Field(
        None, alias="diagnosticGeometryModel"
    )

    # Global paths
    detector_definitions_path: Optional[Path] = Field(None, alias="detectorDefinitionsPath")
    instrument_response_directory: Optional[Path] = Field(None, alias="instrumentResponseDirectory")

    # Noise and filter modeling
    apply_electrical_filter: bool = Field(False, alias="applyElectricalFilter")
    filter_type: Optional[Literal["RC", "bandpass", "gaussian"]] = Field(None, alias="filterType")
    filter_parameters: Optional[Dict[str, float]] = Field(None, alias="filterParameters")
    include_detector_noise: bool = Field(False, alias="includeDetectorNoise")
    noise_model: Optional[Literal["gaussian", "poisson", "custom"]] = Field(None, alias="noiseModel")
    noise_parameters: Optional[Dict[str, float]] = Field(None, alias="noiseParameters")

    # Per-instrument overrides
    instrument_overrides: Optional[Dict[str, SyntheticInstrument]] = Field(
        None, alias="instrumentOverrides"
    )

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "SyntheticDiagnostics":
        return cls(apply_time_response=False, apply_energy_filter=False)

    def resolve_defaults(self) -> "SyntheticDiagnostics":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or f.metadata or {}) for n, f in cls.model_fields.items()}

    def normalize_units(self, units: UnitsSettings) -> "SyntheticDiagnostics":
        unit_map = units.normalize_units()
        scale = unit_map.get("temporal", 1.0)
        interval = self.sampling_interval_ns * scale
        overrides = None
        if self.instrument_overrides:
            overrides = {}
            for name, inst in self.instrument_overrides.items():
                val = inst.sampling_override_ns
                if val is not None:
                    val = val * scale
                overrides[name] = inst.model_copy(update={"sampling_override_ns": val})
        return self.model_copy(update={"sampling_interval_ns": interval, "instrument_overrides": overrides})

    def summarize(self) -> str:
        diag_flags = [
            (self.synthetic_current_waveform_enabled, "Current"),
            (self.synthetic_voltage_waveform_enabled, "Voltage"),
            (self.synthetic_rogowski_signal_enabled, "Rogowski"),
            (self.synthetic_bdot_signal_enabled, "B-dot"),
            (self.synthetic_neutron_tof_enabled, "TOF"),
            (self.synthetic_xray_pinhole_enabled, "X-ray"),
        ]
        active = [name for flag, name in diag_flags if flag]
        filt = "None"
        if self.apply_electrical_filter and self.filter_type and self.filter_parameters:
            cutoff = self.filter_parameters.get("cutoff")
            unit = " Hz" if cutoff is not None else ""
            val = f"{cutoff}{unit}" if cutoff is not None else ""
            filt = f"{self.filter_type}({val})"
        noise = self.noise_model.capitalize() if self.include_detector_noise and self.noise_model else "None"
        num_det = len(self.detector_ids) if self.detector_ids else 0
        ids = ", ".join(self.detector_ids[:2]) if self.detector_ids else ""
        geom = self.diagnostic_geometry_model or "n/a"
        return (
            f"Synthetic Diagnostics: [{', '.join(active)}]\n"
            f"Output: {self.output_format.upper()} @ {self.sampling_interval_ns} ns, "
            f"TimeResponse: {'ON' if self.apply_time_response else 'OFF'}, "
            f"Filter: {filt}, Noise: {noise}\n"
            f"Detectors: {num_det}, Geometry: {geom}, IDs: {ids}"
        )

    def hash_synthetic_diagnostics_config(self) -> str:
        data = self.model_dump(by_alias=True, exclude={"synthetic_diagnostics_config_hash"})
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "SyntheticDiagnostics") -> "SyntheticDiagnostics":
        if values.apply_electrical_filter:
            if values.filter_type is None or values.filter_parameters is None:
                raise ValueError("filter_parameters required when apply_electrical_filter is True")
        if values.include_detector_noise:
            if values.noise_model is None or values.noise_parameters is None:
                raise ValueError("noise_parameters required when include_detector_noise is True")
        if (
            values.apply_time_response or values.apply_energy_filter or values.apply_spatial_psf
        ) and values.instrument_response_directory is None:
            raise ValueError("instrument_response_directory required when response modeling enabled")
        if values.instrument_overrides:
            if values.detector_ids:
                for key in values.instrument_overrides.keys():
                    if key not in values.detector_ids:
                        raise ValueError("instrument_override key not listed in detector_ids")
            new_overrides = {}
            for name, inst in values.instrument_overrides.items():
                if values.apply_time_response and inst.response_file is None and values.instrument_response_directory is None:
                    raise ValueError("response_file required for instrument when time response applied")
                if inst.sampling_override_ns is not None and inst.sampling_override_ns <= 0:
                    raise ValueError("sampling_override_ns must be positive")
                new_overrides[name] = inst
            values = values.model_copy(update={"instrument_overrides": new_overrides})
        values = values.model_copy(update={"synthetic_diagnostics_config_hash": values.hash_synthetic_diagnostics_config()})
        return values


__all__ = ["SyntheticDiagnostics", "SyntheticInstrument"]
