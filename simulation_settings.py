from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict, Field

from core_schema import (
    ConfigSectionBase,
    GeometryType,
    ModeType,
    model_validator,
    to_camel_case,
)


class SimulationSettings(ConfigSectionBase):
    """Simulation control parameters."""

    config_section_id: ClassVar[str] = "simulation"

    geometry: GeometryType = Field(
        default=GeometryType.RZ_2D,
        metadata={"units": "-", "category": "Simulation", "group": "General"},
    )
    mode: ModeType = Field(
        default=ModeType.FLUID,
        metadata={"units": "-", "category": "Simulation", "group": "General"},
    )
    time_start: float = Field(
        default=0.0,
        metadata={"units": "us", "category": "Simulation", "group": "Time"},
    )
    time_end: float = Field(
        default=1.0,
        metadata={"units": "us", "category": "Simulation", "group": "Time"},
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    @classmethod
    def with_defaults(cls) -> "SimulationSettings":
        return cls()

    @model_validator(mode="after")
    def check_times(cls, values: "SimulationSettings") -> "SimulationSettings":
        if values.time_end <= values.time_start:
            raise ValueError("time_end must be greater than time_start")
        return values


__all__ = ["SimulationSettings"]
