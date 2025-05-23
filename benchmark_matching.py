from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Literal, Union

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

from core_schema import ConfigSectionBase, UnitsSystem, UNIT_SCALE_MAP, to_camel_case


class BenchmarkMatching(ConfigSectionBase):
    """Benchmarking and matching configuration."""

    config_section_id: ClassVar[Literal["benchmark"]] = "benchmark"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    dataset_id: str = Field(..., description="Benchmark dataset or facility ID (e.g., PF1000, NX2)")
    benchmark_trace_path: Optional[Path] = None
    benchmark_trace_paths: Optional[List[Path]] = None
    benchmark_format: Optional[Literal["csv", "hdf5", "json"]] = None
    benchmark_time_unit: Optional[Literal["s", "ms", "us", "ns"]] = "us"
    benchmark_units: Optional[Dict[str, str]] = None
    benchmark_selection_strategy: Optional[Literal["average", "best_fit", "worst_case", "all"]] = "best_fit"

    benchmark_fields: List[Literal["I(t)", "V(t)", "B_max", "pinch_radius"]] = Field(default_factory=list)
    compare_fields: List[Literal["E", "B", "rho", "T", "J"]] = Field(default_factory=list)

    waveform_tolerance: float = Field(..., metadata={"units": "%", "group": "Matching"})
    match_waveform_features: bool = True
    feature_alignment_method: Literal["cross_correlation", "windowed_lag", "manual"] = "cross_correlation"
    max_time_alignment_error_ns: Optional[float] = Field(None, metadata={"units": "ns"})
    match_region_start_us: Optional[float] = Field(None, metadata={"units": "us"})
    match_region_end_us: Optional[float] = Field(None, metadata={"units": "us"})
    match_directionality: Optional[Literal["rising", "falling", "both"]] = "both"

    scoring_method: Literal["RMSE", "MAE", "correlation", "composite"] = "RMSE"
    score_threshold: Optional[float] = None

    signal_preprocessing: Optional[Dict[str, Union[str, float]]] = Field(
        default_factory=lambda: {
            "smoothing": "savitzky_golay",
            "normalize": True,
            "window_size_us": 0.5,
        }
    )

    benchmark_config_hash: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "BenchmarkMatching":
        return cls(
            dataset_id="PF1000",
            waveform_tolerance=1.0,
            benchmark_fields=["I(t)"],
        )

    def resolve_defaults(self) -> "BenchmarkMatching":
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [n for n, f in self.model_fields.items() if f.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or {}) for n, f in self.model_fields.items()}

    def normalize_units(self, base_units: UnitsSystem) -> "BenchmarkMatching":
        scale = UNIT_SCALE_MAP.get(base_units, 1.0)
        update: Dict[str, Any] = {}
        if self.max_time_alignment_error_ns is not None:
            update["max_time_alignment_error_ns"] = self.max_time_alignment_error_ns * scale
        if self.match_region_start_us is not None:
            update["match_region_start_us"] = self.match_region_start_us * scale
        if self.match_region_end_us is not None:
            update["match_region_end_us"] = self.match_region_end_us * scale
        if self.signal_preprocessing and "window_size_us" in self.signal_preprocessing:
            sp = dict(self.signal_preprocessing)
            sp["window_size_us"] = sp["window_size_us"] * scale
            update["signal_preprocessing"] = sp
        return self.model_copy(update=update)

    def summarize(self) -> str:
        fields = ", ".join(self.benchmark_fields)
        return (
            f"Benchmark: {self.dataset_id} using {fields}\n"
            f"Format: {self.benchmark_format}, time unit: {self.benchmark_time_unit}\n"
            f"Waveform match: tolerance = {self.waveform_tolerance}%, method = {self.feature_alignment_method}\n"
            f"Match window: {self.match_region_start_us}\u2013{self.match_region_end_us} \u03bcs, direction = {self.match_directionality}\n"
            f"Scoring: {self.scoring_method}, threshold = {self.score_threshold}"
        )

    def hash_benchmark(self) -> str:
        data = {
            "dataset_id": self.dataset_id,
            "benchmark_trace_path": str(self.benchmark_trace_path) if self.benchmark_trace_path else None,
            "benchmark_trace_paths": [str(p) for p in self.benchmark_trace_paths] if self.benchmark_trace_paths else None,
            "waveform_tolerance": self.waveform_tolerance,
            "feature_alignment_method": self.feature_alignment_method,
            "match_region_start_us": self.match_region_start_us,
            "match_region_end_us": self.match_region_end_us,
            "signal_preprocessing": self.signal_preprocessing,
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "BenchmarkMatching") -> "BenchmarkMatching":
        if values.benchmark_trace_path and values.benchmark_trace_paths:
            raise ValueError("Only one of benchmark_trace_path or benchmark_trace_paths allowed")

        if values.benchmark_format and not (values.benchmark_trace_path or values.benchmark_trace_paths):
            raise ValueError("benchmark_format requires benchmark trace path")

        paths: List[Path] = []
        if values.benchmark_trace_path:
            paths = [values.benchmark_trace_path]
        elif values.benchmark_trace_paths:
            paths = values.benchmark_trace_paths

        for p in paths:
            if not (Path(p).exists() or str(p).startswith("file:")):
                raise ValueError(f"benchmark trace path {p} must exist or be URI-valid")

        if values.benchmark_trace_paths and len(values.benchmark_trace_paths) > 1:
            suffix = values.benchmark_trace_paths[0].suffix
            for p in values.benchmark_trace_paths:
                if p.suffix != suffix:
                    raise ValueError("benchmark_trace_paths must have consistent structure")

        if values.benchmark_units is not None:
            valid_units = {"kA", "A", "MA", "V", "kV", "T", "G", "cm", "mm", "m", "s", "ms", "us", "ns"}
            for u in values.benchmark_units.values():
                if u not in valid_units:
                    raise ValueError("benchmark_units contain unsupported unit")

        if values.compare_fields and not paths:
            raise ValueError("compare_fields requires benchmark trace path")

        if values.signal_preprocessing:
            sp = values.signal_preprocessing
            if "smoothing" in sp:
                if sp["smoothing"] not in {"moving_average", "savitzky_golay"}:
                    raise ValueError("invalid smoothing type")
                if "window_size_us" not in sp:
                    raise ValueError("window_size_us required when smoothing is set")

        if (
            values.match_region_start_us is not None
            and values.match_region_end_us is not None
            and values.match_region_start_us >= values.match_region_end_us
        ):
            raise ValueError("match_region_start_us must be < match_region_end_us")

        if values.match_waveform_features and not values.benchmark_fields:
            raise ValueError("benchmark_fields required when match_waveform_features is True")

        ctx = getattr(values, "_context", {})
        diag_fields = ctx.get("fields_to_output")
        if diag_fields is not None:
            missing = [f for f in values.compare_fields if f not in diag_fields]
            if missing:
                raise ValueError(f"compare_fields not in Diagnostics.fields_to_output: {missing}")

        values = values.model_copy(update={"benchmark_config_hash": values.hash_benchmark()})
        return values

__all__ = ["BenchmarkMatching"]
