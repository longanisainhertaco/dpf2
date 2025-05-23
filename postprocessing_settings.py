from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union, Literal

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


class PostprocessingSettings(ConfigSectionBase):
    """Configuration for DPF postprocessing and analysis."""

    config_section_id: ClassVar[Literal["postprocessing"]] = "postprocessing"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Output configuration
    enabled: bool = True
    postprocessing_output_dir: str = "postprocessing/"
    file_output_format: Literal["csv", "hdf5", "OpenPMD"] = "OpenPMD"
    output_frequency: Literal["every_step", "pinch_only", "final_state", "custom"] = "final_state"
    custom_output_times_us: Optional[List[float]] = None
    postprocessing_config_hash: Optional[str] = None

    # Pinch detection
    pinch_detection_method: Optional[Literal["Ti_peak", "Jz_peak", "radius_min"]] = "Ti_peak"
    pinch_detection_threshold: Optional[float] = None

    # Task toggles
    postprocessing_task_types: Optional[List[Literal[
        "neutron", "xray", "field", "particle", "synthetic"
    ]]] = ["neutron", "xray", "field"]

    generate_synthetic_diagnostics: bool = True
    compute_neutron_yield_breakdown: bool = True
    compute_xray_spectrum: bool = True
    extract_maxwellian_fit_parameters: bool = True
    track_plasma_centerline_evolution: bool = True
    export_particle_energy_spectra: bool = True
    integrate_signal_over_time: Optional[List[str]] = Field(default_factory=list)

    # Field analysis
    calculate_field_extrema: bool = True
    compute_spatial_averages: bool = True
    integrate_energy_density: bool = True
    output_field_slices: Optional[List[Literal["x", "y", "z", "r"]]] = ["z"]
    field_slice_positions_cm: Optional[List[float]] = None
    field_slice_units: Literal["cm", "m"] = "cm"

    # Fourier / frequency analysis
    perform_fourier_analysis: bool = False
    fourier_axes: Optional[List[Literal["x", "y", "z"]]] = None
    frequency_window_us: Optional[Tuple[float, float]] = None

    # Script + template control
    raw_data_source_dir: Optional[Path] = None
    external_filter_script_path: Optional[Path] = None
    external_filter_parameters: Optional[Dict[str, Union[str, float]]] = None
    postprocessing_template_path: Optional[Path] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "PostprocessingSettings":
        return cls()

    def resolve_defaults(self) -> "PostprocessingSettings":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or f.metadata or {}) for n, f in cls.model_fields.items()}

    def normalize_units(self, units: UnitsSettings) -> "PostprocessingSettings":
        unit_map = units.normalize_units()
        scale_t = unit_map.get("temporal", 1.0)
        scale_x = unit_map.get("spatial", 1.0)
        times = None
        if self.custom_output_times_us is not None:
            times = [t * scale_t for t in self.custom_output_times_us]
        freq_win = None
        if self.frequency_window_us is not None:
            freq_win = (
                self.frequency_window_us[0] * scale_t,
                self.frequency_window_us[1] * scale_t,
            )
        slices = None
        if self.field_slice_positions_cm is not None:
            slices = [p * scale_x for p in self.field_slice_positions_cm]
        return self.model_copy(
            update={
                "custom_output_times_us": times,
                "frequency_window_us": freq_win,
                "field_slice_positions_cm": slices,
            }
        )

    def summarize(self) -> str:
        freq = self.output_frequency
        fmt = self.file_output_format
        tasks = ", ".join(self.postprocessing_task_types or [])
        integ = ", ".join(self.integrate_signal_over_time) if self.integrate_signal_over_time else "None"
        slices = ", ".join(self.output_field_slices or [])
        pos = ", ".join(str(p) for p in (self.field_slice_positions_cm or []))
        fourier = "none"
        if self.perform_fourier_analysis:
            axes = ", ".join(self.fourier_axes or [])
            if self.frequency_window_us:
                fw = f"{self.frequency_window_us[0]}–{self.frequency_window_us[1]}"
            else:
                fw = "full"
            fourier = f"{axes} [{fw} μs]"
        pinch = self.pinch_detection_method or "none"
        thr = f" > {self.pinch_detection_threshold}" if self.pinch_detection_threshold is not None else ""
        extrema = self.calculate_field_extrema or self.compute_spatial_averages
        ext_str = "ON" if extrema else "OFF"
        return (
            f"Postprocessing: Output → {freq}, Format: {fmt}\n"
            f"Tasks: {tasks} | Time integration: {integ}\n"
            f"Field slices: {slices} @ [{pos}] {self.field_slice_units} | Fourier: {fourier}\n"
            f"Pinch: {pinch}{thr} | Extrema + Averages: {ext_str}"
        )

    def hash_postprocessing_config(self) -> str:
        data = self.model_dump(by_alias=True, exclude={"postprocessing_config_hash"})
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "PostprocessingSettings") -> "PostprocessingSettings":
        if values.output_frequency == "custom":
            if not values.custom_output_times_us:
                raise ValueError("custom_output_times_us required when output_frequency is custom")
            if values.custom_output_times_us != sorted(values.custom_output_times_us):
                raise ValueError("custom_output_times_us must be sorted ascending")
            if any(t <= 0 for t in values.custom_output_times_us):
                raise ValueError("custom_output_times_us must be positive")
        if (
            values.output_frequency == "pinch_only"
            and values.pinch_detection_threshold is None
        ):
            raise ValueError(
                "pinch_detection_threshold required when output_frequency is 'pinch_only'"
            )
        if values.external_filter_parameters is not None and values.external_filter_script_path is None:
            raise ValueError("external_filter_script_path required when external_filter_parameters provided")
        if values.external_filter_script_path is not None:
            p = Path(values.external_filter_script_path)
            if not p.exists():
                raise ValueError("external_filter_script_path must exist")
        if values.field_slice_positions_cm is not None and values.output_field_slices is not None:
            if len(values.field_slice_positions_cm) != len(values.output_field_slices):
                raise ValueError("field_slice_positions_cm length must match output_field_slices")
        values = values.model_copy(update={"postprocessing_config_hash": values.hash_postprocessing_config()})
        return values


__all__ = ["PostprocessingSettings"]
