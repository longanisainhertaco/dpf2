"""Run metadata and provenance tracking for DPF simulations."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, ConfigDict, Field, root_validator


def model_validator(*, mode: str = "after"):
    """Compatibility helper mirroring pydantic.v2 model_validator."""

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


class MLMetadata(BaseModel):
    """Metadata describing a surrogate or ML model."""

    model: str
    version: str
    engine: Optional[str] = None
    optimizer: Optional[str] = None
    notes: Optional[str] = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )


class MLResult(BaseModel):
    """Results reported by the surrogate/ML run."""

    predicted_yield: Optional[float] = None
    latency: Optional[float] = None
    model_error: Optional[float] = None
    confidence: Optional[float] = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )


class Metadata(ConfigSectionBase):
    """Validated configuration for run metadata."""

    config_section_id: ClassVar[Literal["metadata"]] = "metadata"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Core fields
    schema_version: str
    sim_version: str
    created_by: str = Field(..., alias="createdBy")
    commit_hash: str = Field(..., alias="commitHash")
    code_origin_url: Optional[str] = Field(
        None, description="URL of the codebase used for this run"
    )
    run_uuid: str
    creation_time: datetime = Field(
        ..., alias="creationTime", description="Must be UTC-aware ISO 8601"
    )
    summary_id: Optional[str] = Field(
        None, alias="summaryId", description="Short 6-character summary hash"
    )
    run_label: Optional[str] = Field(
        None, alias="runLabel", description="Human-readable short name or tag for this run"
    )

    # Campaign & traceability
    campaign_mode_enabled: bool = False
    ensemble_shot_configs: Optional[List[Path]] = Field(default_factory=list)
    run_lineage: Optional[List[str]] = Field(default_factory=list)
    restart_origin_uuid: Optional[str] = None
    restart_config_hash: Optional[str] = Field(
        None, alias="restartConfigHash", pattern=r"^[a-fA-F0-9]{6,64}$"
    )
    config_hash: Optional[str] = Field(
        None, alias="configHash", pattern=r"^[a-fA-F0-9]{6,64}$"
    )

    # Platform tagging
    platform_tag: Optional[str] = Field(
        None, alias="platformTag", description="Platform tag or compute cluster identifier"
    )
    arch_id: Optional[str] = Field(
        None, alias="archId", description="Architecture ID or backend fingerprint (e.g., cuda-12.2-ampere)"
    )

    # ML usage
    ml_metadata: Optional[MLMetadata] = Field(None, alias="mlMetadata")
    ml_objective: Optional[str] = Field(None, alias="mlObjective")
    ml_parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = Field(
        None, alias="mlParameterBounds"
    )
    ml_result: Optional[MLResult] = Field(None, alias="mlResult")
    surrogate_confidence_threshold: Optional[float] = Field(
        None, alias="surrogateConfidenceThreshold"
    )
    use_surrogate_model: Optional[str] = Field(None, alias="useSurrogateModel")
    yield_targeting_enabled: Optional[bool] = Field(
        None, alias="yieldTargetingEnabled"
    )

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "Metadata":
        return cls(
            schema_version="1.0",
            sim_version="0.1",
            created_by="unknown",
            commit_hash="none",
            run_uuid=str(uuid.uuid4()),
            creation_time=datetime.now(timezone.utc),
        )

    def resolve_defaults(self) -> "Metadata":
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [n for n, f in self.model_fields.items() if f.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {n: field.json_schema_extra or {} for n, field in self.model_fields.items()}

    def summarize(self) -> str:
        surrogate_desc = "none"
        if self.use_surrogate_model and self.ml_metadata:
            surrogate_desc = f"{self.ml_metadata.model}"
        conf = (
            f" < {self.surrogate_confidence_threshold}" if self.surrogate_confidence_threshold is not None else ""
        )
        return (
            f"Schema: {self.schema_version}, Code: {self.sim_version} (commit {self.commit_hash})\n"
            f"UUID: {self.run_uuid}, Label: {self.run_label or 'none'}, Short ID: {self.summary_id or 'n/a'}\n"
            f"Campaign: {'enabled' if self.campaign_mode_enabled else 'disabled'}, "
            f"Restart: {self.restart_origin_uuid or 'none'}, Hash: {self.config_hash or 'none'}\n"
            f"Platform: {self.platform_tag or 'unknown'}, Arch: {self.arch_id or 'unknown'}\n"
            f"Surrogate: {surrogate_desc}{(' (confidence' + conf + ')') if self.use_surrogate_model else ''}, "
            f"ML obj: {self.ml_objective or 'none'}"
        )

    # Helper ---------------------------------------------------------------
    def hash_metadata(self) -> str:
        data = self.model_dump(by_alias=True, exclude={"summary_id"}, exclude_none=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "Metadata") -> "Metadata":
        if values.use_surrogate_model:
            if not values.ml_metadata or not values.ml_metadata.model or not values.ml_metadata.version:
                raise ValueError("ml_metadata.model and version required when use_surrogate_model is set")

        if values.ml_parameter_bounds:
            for k, (low, high) in values.ml_parameter_bounds.items():
                if low >= high:
                    raise ValueError(f"invalid bounds for {k}")

        hex_re = re.compile(r"^[a-fA-F0-9]{6,64}$")
        if values.restart_config_hash and not hex_re.match(values.restart_config_hash):
            raise ValueError("restart_config_hash format invalid")
        if values.config_hash and not hex_re.match(values.config_hash):
            raise ValueError("config_hash format invalid")

        if values.summary_id is None:
            sid = values.hash_metadata()[:6]
            values = values.model_copy(update={"summary_id": sid})

        return values


__all__ = ["Metadata", "MLMetadata", "MLResult"]

