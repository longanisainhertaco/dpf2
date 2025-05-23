"""Circuit configuration schema for DPF simulations.

Example YAML::

    circuit:
      lExt: 1.2
      rExt: 2.5
      cExt: 80.0
      v0: 25.0
      switchDelay: 50.0
      switchingModel: multi-bank
      triggerJitterStddev: 1.5
      enableFieldTriggeredSwitchClosure: true
      switchFeedbackDelayNs: 20.0
      overrideInductiveVoltageLimit: 12.0
      interShotRecoveryTimeNs: 5000.0
      waveformProfile:
        - [0.0, 0.0]
        - [1.0, 20.0]
        - [2.0, 25.0]
      waveformFormatVersion: "1.0"
      waveformTimeUnit: us
      waveformConflictResolution: prefer_inline
      onNoCurrentBehavior: abort
      circuitFaultFlags: [arc, timeout]
      failureConditions:
        current_peak_max: 140000.0
      failureTags: ["overvoltage", "timing_error"]
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Literal

from pydantic import ConfigDict, Field, root_validator

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

from pydantic import BaseModel
if not hasattr(BaseModel, "model_validate"):
    BaseModel.model_validate = classmethod(lambda cls, d, **_: cls.parse_obj(d))
if not hasattr(BaseModel, "model_dump"):
    BaseModel.model_dump = BaseModel.dict
if not hasattr(BaseModel, "model_dump_json"):
    BaseModel.model_dump_json = BaseModel.json
if not hasattr(BaseModel, "model_copy"):
    BaseModel.model_copy = BaseModel.copy

from core_schema import (
    ConfigSectionBase,
    TimeVoltageProfile,
    UnitsSystem,
    UNIT_SCALE_MAP,
    to_camel_case,
    CircuitFaultTypeEnum,
)


class CircuitConfig(ConfigSectionBase):
    """Validated external circuit configuration."""

    config_section_id: ClassVar[Literal["circuit"]] = "circuit"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        allow_population_by_field_name=True,
        validate_default=True,
    )


    # --- LRC Parameters -------------------------------------------------
    L_ext: float = Field(
        ..., alias="lExt", metadata={"units": "uH", "category": "Circuit", "group": "LRC"}
    )
    R_ext: float = Field(
        ..., alias="rExt", metadata={"units": "mΩ", "category": "Circuit", "group": "LRC"}
    )
    C_ext: float = Field(
        ..., alias="cExt", metadata={"units": "μF", "category": "Circuit", "group": "LRC"}
    )
    V0: float = Field(
        ..., alias="v0", metadata={"units": "kV", "category": "Circuit", "group": "LRC"}
    )
    switch_delay: float = Field(
        ..., alias="switchDelay", metadata={"units": "ns", "category": "Circuit", "group": "LRC"}
    )

    # --- Switching Behavior --------------------------------------------
    switching_model: Literal["ideal", "jittered", "multi-bank"] = Field(
        ..., alias="switchingModel", metadata={"category": "Circuit", "group": "Switching"}
    )
    trigger_jitter_stddev: float = Field(
        ..., alias="triggerJitterStddev", metadata={"units": "ns", "category": "Circuit", "group": "Switching"}
    )
    enable_field_triggered_switch_closure: bool = Field(
        ..., alias="enableFieldTriggeredSwitchClosure", metadata={"category": "Circuit", "group": "Switching"}
    )
    switch_feedback_delay_ns: Optional[float] = Field(
        None, alias="switchFeedbackDelayNs", metadata={"units": "ns", "category": "Circuit", "group": "Switching"}
    )
    override_inductive_voltage_limit: Optional[float] = Field(
        None, alias="overrideInductiveVoltageLimit", metadata={"units": "kV", "category": "Circuit", "group": "Switching"}
    )
    inter_shot_recovery_time_ns: Optional[float] = Field(
        None, alias="interShotRecoveryTimeNs", metadata={"units": "ns", "category": "Circuit", "group": "Switching"}
    )
    abort_on_no_current: bool = Field(
        ..., alias="abortOnNoCurrent", metadata={"category": "Circuit", "group": "Switching"}
    )
    on_no_current_behavior: Literal["log", "skip", "abort"] = Field(
        "abort", alias="onNoCurrentBehavior", metadata={"category": "Circuit", "group": "Switching"}
    )

    # --- Waveform Profile ----------------------------------------------
    waveform_profile: Optional[TimeVoltageProfile] = Field(
        None, alias="waveformProfile", metadata={"units": ["μs", "kV"], "category": "Circuit", "group": "Waveform"}
    )
    waveform_profile_path: Optional[Path] = Field(
        None, alias="waveformProfilePath", metadata={"category": "Circuit", "group": "Waveform"}
    )
    waveform_format_version: Optional[str] = Field(
        None, alias="waveformFormatVersion", metadata={"category": "Circuit", "group": "Waveform"}
    )
    waveform_time_unit: Literal["s", "ms", "us", "ns"] = Field(
        "us", alias="waveformTimeUnit", metadata={"category": "Circuit", "group": "Waveform"}
    )
    waveform_conflict_resolution: Literal["prefer_inline", "prefer_path"] = Field(
        "prefer_inline", alias="waveformConflictResolution", metadata={"category": "Circuit", "group": "Waveform"}
    )

    # --- Fault Modeling -------------------------------------------------
    circuit_fault_flags: List[CircuitFaultTypeEnum] = Field(
        default_factory=list,
        alias="circuitFaultFlags",
        metadata={"category": "Circuit", "group": "Faults"},
    )
    failure_conditions: Optional[Dict[str, float]] = Field(
        None, alias="failureConditions", metadata={"category": "Circuit", "group": "Faults"}
    )
    failure_tags: Optional[List[str]] = Field(
        None, alias="failureTags", metadata={"category": "Circuit", "group": "Faults"}
    )

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "CircuitConfig":
        return cls(
            L_ext=1.2,
            R_ext=2.5,
            C_ext=80.0,
            V0=25.0,
            switch_delay=50.0,
            switching_model="ideal",
            trigger_jitter_stddev=1.0,
            enable_field_triggered_switch_closure=False,
            abort_on_no_current=False,
            waveform_profile=[(0.0, 0.0), (1.0, 25.0)],
        )

    def resolve_defaults(self) -> "CircuitConfig":
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [name for name, f in self.model_fields.items() if f.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, object]]:
        return {name: (field.json_schema_extra or field.metadata or {}) for name, field in self.model_fields.items()}

    def summarize(self) -> str:
        w_desc = "none"
        if self.waveform_profile is not None:
            w_desc = f"inline ({len(self.waveform_profile)} points)"
        elif self.waveform_profile_path is not None:
            w_desc = f"path={self.waveform_profile_path}"
        faults = [f.value for f in self.circuit_fault_flags]
        return (
            f"Circuit: {self.C_ext} μF, {self.L_ext} μH, {self.R_ext} mΩ, V0 = {self.V0} kV\n"
            f"Switching: {self.switching_model} (jitter = ±{self.trigger_jitter_stddev} ns), "
            f"delay = {self.switch_delay} ns\n"
            f"Waveform: {w_desc}, unit = {self.waveform_time_unit}\n"
            f"Faults: [{', '.join(faults)}]"
        )

    def normalize_units(self, base_units: UnitsSystem) -> "CircuitConfig":
        scale = UNIT_SCALE_MAP.get(base_units, 1.0)
        wf = None
        if self.waveform_profile is not None:
            wf = [(t * scale, v * scale) for t, v in self.waveform_profile]
        return self.model_copy(update={"waveform_profile": wf})

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "CircuitConfig") -> "CircuitConfig":
        if values.switching_model == "multi-bank" and values.switch_feedback_delay_ns is None:
            raise ValueError("switch_feedback_delay_ns must be set for multi-bank switching")

        if values.waveform_profile is None and values.waveform_profile_path is None:
            raise ValueError("waveform_profile or waveform_profile_path must be provided")

        if values.waveform_profile is not None and values.waveform_profile_path is not None:
            if values.waveform_conflict_resolution == "prefer_inline":
                values.waveform_profile_path = None
            elif values.waveform_conflict_resolution == "prefer_path":
                values.waveform_profile = None

        if (
            values.override_inductive_voltage_limit is not None
            and values.override_inductive_voltage_limit <= 0
        ):
            raise ValueError("override_inductive_voltage_limit must be > 0")

        if values.abort_on_no_current:
            if not values.failure_conditions or "current_peak_max" not in values.failure_conditions:
                raise ValueError("failure_conditions must include 'current_peak_max' when abort_on_no_current is True")

        if (
            values.waveform_format_version is not None
            and values.waveform_profile is None
            and values.waveform_profile_path is None
        ):
            raise ValueError("waveform data must be provided when waveform_format_version is set")

        return values


__all__ = ["CircuitConfig", "CircuitFaultTypeEnum"]
