from __future__ import annotations

from pathlib import Path
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, ConfigDict, Field, root_validator

# Compatibility helpers -------------------------------------------------------

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
from core_schema import (
    ConfigSectionBase,
    DetectorConfig,
    to_camel_case,
)
from units_settings import UnitsSettings


# ---------------------------------------------------------------------------
class SXRModel(BaseModel):
    """Simplified soft X-ray detector model."""

    name: str
    position: Tuple[float, float, float]


class TOFModel(BaseModel):
    """Simplified neutron time-of-flight detector model."""

    name: str
    position: Tuple[float, float, float]


class OutputField(str, Enum):
    E = "E"
    B = "B"
    rho = "rho"
    T = "T"
    J = "J"
    pressure = "pressure"


class DetectorArrayGenerator(ConfigSectionBase):
    """Procedural generation for detector arrays."""

    config_section_id: ClassVar[Literal["detector_array_generator"]] = "detector_array_generator"

    type: Literal["arc", "grid", "custom"]
    center: Tuple[float, float, float]
    radius: Optional[float] = None
    count: Optional[int] = None
    plane_normal: Optional[Literal["x", "y", "z"]] = None
    start_angle_deg: Optional[float] = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    @classmethod
    def with_defaults(cls) -> "DetectorArrayGenerator":
        return cls(type="custom", center=(0.0, 0.0, 0.0))

    def summarize(self) -> str:
        return f"ArrayGen: {self.type} at {self.center}"

    @model_validator(mode="after")
    def check_rules(cls, values: "DetectorArrayGenerator") -> "DetectorArrayGenerator":
        if values.type == "arc":
            if values.radius is None or values.count is None:
                raise ValueError("radius and count required for arc array generation")
        return values


class Diagnostics(ConfigSectionBase):
    """Diagnostics configuration schema."""

    config_section_id: ClassVar[Literal["diagnostics"]] = "diagnostics"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # -- Global flags ------------------------------------------------------
    disable_all_diagnostics: bool = False
    diagnostic_mode: Literal["minimal", "standard", "full"] = "standard"

    # -- Output configuration ---------------------------------------------
    output_dir: str
    output_format: Literal["HDF5", "ADIOS2"]
    output_format_version: Optional[str] = None
    plotfile_interval: int = Field(..., metadata={"units": "steps"})
    checkpoint_interval: int = Field(..., metadata={"units": "steps"})
    restart_from: Optional[str] = None

    # -- Streaming controls ------------------------------------------------
    enable_runtime_observables_stream: bool = False
    streaming_backend: Optional[Literal["websocket", "zmq", "file", "disabled"]] = "disabled"
    stream_output_path: Optional[Path] = None
    streaming_protocol_version: Optional[str] = None
    streaming_transport_params: Optional[Dict[str, Any]] = None
    stream_throttle_interval_ms: Optional[int] = Field(100, metadata={"units": "ms"})
    stream_timeout_ms: Optional[int] = Field(5000, metadata={"units": "ms"})
    runtime_log_streams_enabled: List[
        Literal["neutron_yield", "current", "pinch_radius", "B_max", "E_max"]
    ] = Field(default_factory=list)

    # -- Sampling configuration -------------------------------------------
    diagnostic_sampling_rates: Optional[Dict[str, int]] = Field(default_factory=dict)
    sampling_unit: Literal["step", "time"] = "step"
    default_sampling_resolution: Optional[int] = None
    override_all_sampling: bool = False
    sampling_time_window_ns: Optional[Tuple[float, float]] = Field(None, metadata={"units": "ns"})
    sampling_strategy: Literal["fixed", "adaptive", "script"] = "fixed"
    sampling_profile_path: Optional[Path] = None

    # -- Output limits & compression --------------------------------------
    max_output_volume_MB: Optional[int] = None
    max_field_data_points: Optional[int] = None
    compression_backend: Optional[Literal["zlib", "blosc", "none"]] = "zlib"
    compression_level: Optional[int] = Field(5, ge=0, le=9)

    # -- Field output ------------------------------------------------------
    fields_to_output: List[OutputField] = Field(default_factory=list)
    particle_species_to_track: List[str] = Field(default_factory=list)

    # -- Diagnostic toggles ------------------------------------------------
    beam_diagnostics: bool = False
    beam_pulse_width_estimate_ns: Optional[float] = Field(None, metadata={"units": "ns"})
    enable_beam_convergence_tracking: bool = False
    enable_energy_tracking: bool = False
    enable_mhd_instability_tracking: bool = False
    enable_erosion_mapping: bool = False
    enable_failure_logging: bool = False
    enable_eedf_logging: bool = False
    enable_pitch_angle_tracking: bool = False

    # -- Detector blocks ---------------------------------------------------
    enable_sxr_detectors: bool = False
    enable_neutron_tof_detectors: bool = False
    sxr_detector_models: List[SXRModel] = Field(default_factory=list)
    neutron_tof_detectors: List[TOFModel] = Field(default_factory=list)
    wall_probe_array_config: Optional[List[DetectorConfig]] = None
    detector_config_hash: Optional[str] = None
    max_detector_count: int = 32

    # -- Angular & spectrum controls --------------------------------------
    angular_diagnostics_resolution_deg: Optional[float] = Field(None, metadata={"units": "deg"})
    neutron_energy_bins_MeV: Optional[List[float]] = None
    enable_DT_yield_modeling: bool = False
    sxr_time_window_ns: Optional[float] = Field(None, metadata={"units": "ns"})
    enable_radiative_spectral_shape_modeling: bool = False

    # -- Generator ---------------------------------------------------------
    detector_array_generator: Optional[DetectorArrayGenerator] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "Diagnostics":
        return cls(
            output_dir="diagnostics/",
            output_format="HDF5",
            plotfile_interval=50,
            checkpoint_interval=100,
        )

    def resolve_defaults(self) -> "Diagnostics":
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [name for name, f in self.model_fields.items() if f.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {name: field.json_schema_extra or {} for name, field in self.model_fields.items()}

    def normalize_units(self, units: UnitsSettings) -> "Diagnostics":
        unit_map = units.normalize_units()
        scale = unit_map.get("temporal", 1.0)
        bw = self.beam_pulse_width_estimate_ns
        if bw is not None:
            bw = bw * scale
        tw = None
        if self.sampling_time_window_ns is not None:
            tw = (self.sampling_time_window_ns[0] * scale, self.sampling_time_window_ns[1] * scale)
        sxr_tw = self.sxr_time_window_ns
        if sxr_tw is not None:
            sxr_tw = sxr_tw * scale
        return self.model_copy(update={
            "beam_pulse_width_estimate_ns": bw,
            "sampling_time_window_ns": tw,
            "sxr_time_window_ns": sxr_tw,
        })

    def summarize(self) -> str:
        return (
            f"Diagnostics output: {self.output_format} -> {self.output_dir}, "
            f"mode={self.diagnostic_mode}, streaming={self.streaming_backend}"
        )

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_flags(cls, values: "Diagnostics") -> "Diagnostics":
        if values.disable_all_diagnostics:
            toggles = [
                values.beam_diagnostics,
                values.enable_beam_convergence_tracking,
                values.enable_energy_tracking,
                values.enable_mhd_instability_tracking,
                values.enable_erosion_mapping,
                values.enable_failure_logging,
                values.enable_eedf_logging,
                values.enable_pitch_angle_tracking,
                values.enable_sxr_detectors,
                values.enable_neutron_tof_detectors,
            ]
            if any(toggles):
                raise ValueError("all diagnostic toggles must be False when disable_all_diagnostics is True")
        if values.diagnostic_mode == "minimal":
            values = values.model_copy(update={
                "beam_diagnostics": False,
                "enable_sxr_detectors": False,
                "enable_neutron_tof_detectors": False,
                "wall_probe_array_config": None,
            })
        elif values.diagnostic_mode == "full":
            values = values.model_copy(update={
                "beam_diagnostics": True,
                "enable_beam_convergence_tracking": True,
                "enable_energy_tracking": True,
                "enable_mhd_instability_tracking": True,
                "enable_erosion_mapping": True,
                "enable_failure_logging": True,
                "enable_eedf_logging": True,
                "enable_pitch_angle_tracking": True,
                "enable_sxr_detectors": True,
                "enable_neutron_tof_detectors": True,
            })
        if values.streaming_backend and values.streaming_backend != "disabled" and values.stream_output_path is None:
            raise ValueError("stream_output_path must be set when streaming_backend is enabled")
        if values.compression_backend == "blosc" and values.output_format == "HDF5":
            raise ValueError("blosc compression not supported with HDF5")
        if values.sxr_detector_models or values.neutron_tof_detectors:
            names = [d.name for d in values.sxr_detector_models] + [d.name for d in values.neutron_tof_detectors]
            if len(names) != len(set(names)):
                raise ValueError("duplicate detector names are not allowed")
            if len(names) > values.max_detector_count:
                raise ValueError("detector count exceeds max_detector_count")
        if values.detector_array_generator and values.detector_array_generator.type == "arc":
            gen = values.detector_array_generator
            if gen.radius is None or gen.count is None:
                raise ValueError("radius and count required for arc detector array")
        if values.enable_DT_yield_modeling:
            gas_type = getattr(values, "_context", {}).get("gas_type")
            if gas_type is not None and gas_type != "DT":
                raise ValueError("DT yield modeling requires DT gas")
        return values

    # Custom model_validate to capture context ---------------------------
    @classmethod
    def model_validate(cls, data: Any, **kwargs) -> "Diagnostics":
        context = kwargs.get("context") or {}
        obj = super().model_validate(data)
        setattr(obj, "_context", context)
        # Re-run validation with context aware rules
        obj = cls.check_flags(obj)
        return obj


__all__ = ["Diagnostics", "DetectorArrayGenerator", "OutputField"]
