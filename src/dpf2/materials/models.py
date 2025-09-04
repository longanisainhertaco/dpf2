from __future__ import annotations

from typing import ClassVar, Optional

from ..utils import BaseModel, ConfigDict, Field

from ..core_schema import to_camel_case


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
