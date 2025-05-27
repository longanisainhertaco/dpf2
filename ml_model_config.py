from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Literal

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

# Local imports ---------------------------------------------------------------
from core_schema import ConfigSectionBase, to_camel_case
from metadata import Metadata  # noqa: F401 - used for integration hints
from postprocessing_settings import PostprocessingSettings  # noqa: F401
from validation_suite import ValidationSuite  # noqa: F401


class MLModelConfig(ConfigSectionBase):
    """Validated configuration for machine learning models."""

    config_section_id: ClassVar[Literal["ml_model"]] = "ml_model"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Model specification
    model_type: Literal["regression", "classification"] = "regression"
    surrogate_model_enabled: bool = False
    model_name: str
    model_engine: Literal[
        "sklearn", "tensorflow", "pytorch", "xgboost", "lightgbm", "gpytorch"
    ] = "sklearn"
    model_architecture_template: Optional[
        Literal["MLP", "CNN1D", "LSTM", "Transformer", "Custom"]
    ] = "MLP"
    model_architecture: Optional[str] = None

    # ------------------------------------------------------------------
    # Training configuration
    training_dataset_path: Path
    input_features: List[str]
    output_targets: List[str]
    normalization_method: Literal["zscore", "minmax", "custom"] = "zscore"
    training_epochs: Optional[int] = 100
    batch_size: Optional[int] = 32
    optimizer_type: Optional[Literal["adam", "sgd", "lbfgs", "custom"]] = "adam"
    loss_function: Optional[
        Literal["mse", "mae", "crossentropy", "logloss", "custom"]
    ] = "mse"
    validation_split: Optional[float] = Field(0.2, ge=0.0, le=0.5)
    early_stopping_enabled: bool = True
    early_stopping_patience: Optional[int] = 10
    split_mode: Optional[Literal["train_val", "train_val_test"]] = "train_val"
    cv_folds: Optional[int] = Field(None, ge=2, le=10)
    save_best_model_enabled: bool = True
    best_model_output_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Inference configuration
    inference_enabled: bool = False
    load_existing_model: bool = False
    existing_model_path: Optional[Path] = None
    inference_input_path: Optional[Path] = None
    inference_output_path: Optional[Path] = None
    confidence_threshold: Optional[float] = None
    normalize_input_features: bool = True

    # ------------------------------------------------------------------
    # Evaluation
    evaluation_metrics: List[
        Literal[
            "rmse",
            "mae",
            "r2",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "logloss",
        ]
    ] = ["rmse", "r2"]
    track_feature_importance: bool = True
    model_config_hash: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "MLModelConfig":
        return cls(
            model_name="default_model",
            training_dataset_path=Path("datasets/train.csv"),
            input_features=["V0"],
            output_targets=["neutron_yield"],
        )

    def resolve_defaults(self) -> "MLModelConfig":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or f.metadata or {}) for n, f in cls.model_fields.items()}

    def normalize_units(self) -> "MLModelConfig":
        return self

    def summarize(self) -> str:
        arch = self.model_architecture_template or "None"
        parts = [
            f"ML Model: {self.model_name} [{self.model_engine}], Template: {arch}",
            f"Task: {self.model_type}, Loss = {self.loss_function}, Opt = {self.optimizer_type}, ",
            f"CV = {self.cv_folds or 'none'} folds",
            f"Features: [{', '.join(self.input_features)}] → [{', '.join(self.output_targets)}]",
        ]
        if self.inference_output_path:
            inf = str(self.inference_output_path)
            conf = (
                f" < {self.confidence_threshold}" if self.confidence_threshold is not None else ""
            )
            parts.append(f"Inference: {inf} | Confidence{conf}")
        return "\n".join(parts)

    def hash_ml_model_config(self) -> str:
        data = self.model_dump(by_alias=True, exclude={"model_config_hash"}, exclude_none=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "MLModelConfig") -> "MLModelConfig":
        if not values.input_features:
            raise ValueError("input_features must not be empty")
        if not values.output_targets:
            raise ValueError("output_targets must not be empty")
        if values.model_type == "classification" and values.loss_function not in {"crossentropy", "logloss"}:
            raise ValueError("classification models require crossentropy or logloss loss_function")
        if values.inference_enabled:
            if values.inference_input_path is None or values.inference_output_path is None:
                raise ValueError("inference_input_path and inference_output_path required when inference_enabled")
        if values.cv_folds is not None and values.validation_split is not None:
            raise ValueError("validation_split must be None when cv_folds is set")
        if values.load_existing_model and values.existing_model_path is None:
            raise ValueError("existing_model_path required when load_existing_model is True")
        values = values.model_copy(update={"model_config_hash": values.hash_ml_model_config()})
        return values


__all__ = ["MLModelConfig"]
