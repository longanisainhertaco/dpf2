from __future__ import annotations

import hashlib
import json
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

from core_schema import ConfigSectionBase, UnitsSystem, UNIT_SCALE_MAP, to_camel_case


class ValidationSuite(ConfigSectionBase):
    """Validated configuration for benchmarking simulation outputs."""

    config_section_id: ClassVar[Literal["validation_suite"]] = "validation_suite"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        allow_population_by_field_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Experimental metadata
    experiment_device_id: Literal["PF1000", "NX2", "UNU", "LLNL-DPF", "custom"] = Field(
        ..., alias="experimentDeviceId"
    )
    experiment_campaign_id: str = Field(..., alias="experimentCampaignId")
    dataset_directory: Path = Field(..., alias="datasetDirectory")
    dataset_format: Literal["csv", "json", "hdf5"] = Field(..., alias="datasetFormat")
    observable_file_map: Dict[str, Path] = Field(..., alias="observableFileMap")
    observable_format_spec: Optional[Dict[str, Dict[str, str]]] = Field(
        None, alias="observableFormatSpec"
    )
    observable_uncertainties: Optional[Dict[str, float]] = Field(
        None, alias="observableUncertainties"
    )

    # Validation target configuration
    validation_targets: List[str] = Field(
        default_factory=lambda: ["I(t)", "Yn"], alias="validationTargets"
    )
    observable_tolerances: Dict[str, float] = Field(..., alias="observableTolerances")
    observable_weighting: Optional[Dict[str, float]] = Field(
        None, alias="observableWeighting"
    )
    validation_score_model: Literal["L2", "RMSE", "MAE", "weighted"] = Field(
        "RMSE", alias="validationScoreModel"
    )
    require_all_targets: bool = Field(True, alias="requireAllTargets")
    score_pass_threshold: float = Field(0.85, alias="scorePassThreshold")
    computed_validation_score: Optional[float] = Field(
        None, alias="computedValidationScore"
    )
    validation_passed: Optional[bool] = Field(None, alias="validationPassed")

    # Timing / matching
    validation_time_window_us: Optional[Tuple[float, float]] = Field(
        None, alias="validationTimeWindowUs"
    )
    resample_method: Optional[Literal["interpolate", "zero_order", "downsample"]] = Field(
        "interpolate", alias="resampleMethod"
    )
    interpolation_mode: Optional[Literal["linear", "cubic", "spline"]] = Field(
        "linear", alias="interpolationMode"
    )
    match_on_t_peak: bool = Field(False, alias="matchOnTPeak")

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "ValidationSuite":
        return cls(
            experiment_device_id="PF1000",
            experiment_campaign_id="shot0001",
            dataset_directory=Path("data/experiments"),
            dataset_format="csv",
            observable_file_map={"I(t)": Path("current.csv"), "Yn": Path("yield.csv")},
            observable_tolerances={"I(t)": 0.1, "Yn": 0.3},
        )

    def resolve_defaults(self) -> "ValidationSuite":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or f.metadata or {}) for n, f in cls.model_fields.items()}

    def normalize_units(self, base_units: UnitsSystem) -> "ValidationSuite":
        scale = UNIT_SCALE_MAP.get(base_units, 1.0)
        win = None
        if self.validation_time_window_us is not None:
            win = (
                self.validation_time_window_us[0] * scale,
                self.validation_time_window_us[1] * scale,
            )
        return self.model_copy(update={"validation_time_window_us": win})

    def summarize(self) -> str:
        targets = ", ".join(self.validation_targets)
        tol_vals = [self.observable_tolerances.get(t, 0.0) for t in self.validation_targets]
        tstr = ", ".join(f"{v*100:.0f}%" for v in tol_vals)
        resample = self.resample_method or "none"
        interp = self.interpolation_mode or "linear"
        match = "ON" if self.match_on_t_peak else "OFF"
        return (
            "Validation Suite:\n"
            f"  Device = {self.experiment_device_id}, Campaign = {self.experiment_campaign_id}\n"
            f"  Targets: {targets} | Tolerance: {tstr}\n"
            f"  Score Model: {self.validation_score_model} → Pass ≥ {self.score_pass_threshold}\n"
            f"  Resample = {resample}({interp}), Match on T-peak: {match}\n"
            f"  Format: {self.dataset_format}, Files in {self.dataset_directory}"
        )

    def hash_validation_suite_config(self) -> str:
        data = self.model_dump(by_alias=True, exclude={"computed_validation_score", "validation_passed"})
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "ValidationSuite") -> "ValidationSuite":
        if not Path(values.dataset_directory).exists():
            raise ValueError("dataset_directory must exist")
        for obs, path in values.observable_file_map.items():
            p = Path(path)
            if not p.exists():
                raise ValueError(f"observable file {p} must exist")
        if values.observable_format_spec:
            if set(values.observable_format_spec.keys()) != set(values.observable_file_map.keys()):
                raise ValueError("observable_format_spec keys must match observable_file_map")
            for spec in values.observable_format_spec.values():
                if "time" not in spec or "value" not in spec:
                    raise ValueError("observable_format_spec entries must contain time and value")
        if values.validation_time_window_us is not None:
            start, end = values.validation_time_window_us
            if start >= end:
                raise ValueError("validation_time_window_us start must be < end")
        if values.require_all_targets:
            missing = [t for t in values.validation_targets if t not in values.observable_file_map]
            if missing:
                raise ValueError(f"missing targets in observable_file_map: {missing}")
        weights = values.observable_weighting or {t: 1.0 for t in values.validation_targets}
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("observable_weighting must sum to > 0")
        norm_weights = {k: v / total for k, v in weights.items()}
        update: Dict[str, Any] = {}
        if values.observable_weighting:
            update["observable_weighting"] = norm_weights
        if values.observable_uncertainties and values.validation_score_model == "weighted":
            score = 1.0
            for t in values.validation_targets:
                u = values.observable_uncertainties.get(t, 0.0) if values.observable_uncertainties else 0.0
                w = norm_weights.get(t, 0.0)
                score -= u * w
            update["computed_validation_score"] = score
        score = update.get("computed_validation_score", values.computed_validation_score)
        if score is not None:
            update["validation_passed"] = score >= values.score_pass_threshold
        if update:
            values = values.model_copy(update=update)
        return values


__all__ = ["ValidationSuite"]
