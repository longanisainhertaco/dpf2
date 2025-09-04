from __future__ import annotations

from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..core_schema import to_camel_case


# -----------------------------------------------------------------------------
# Compatibility layer for Pydantic v1/v2 differences
if not hasattr(BaseModel, "model_validate"):
    BaseModel.model_validate = classmethod(lambda cls, d, **_: cls(**d))
if not hasattr(BaseModel, "model_dump"):
    BaseModel.model_dump = BaseModel.dict
if not hasattr(BaseModel, "model_dump_json"):
    BaseModel.model_dump_json = BaseModel.json
if not hasattr(BaseModel, "model_copy"):
    BaseModel.model_copy = BaseModel.copy


class MaterialRef(BaseModel):
    """Reference to a material and optional coating information."""

    material_id: str = Field(..., alias="materialId")
    coating_id: Optional[str] = Field(None, alias="coatingId")

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )


__all__ = ["MaterialRef"]
