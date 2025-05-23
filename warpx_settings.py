from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal

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


class AdaptiveTimeStep(BaseModel):
    """Configuration for adaptive time stepping."""

    cfl: float = Field(..., ge=0.01, le=1.0)
    dt_min: Optional[float] = Field(None, alias="dtMin")
    dt_max: Optional[float] = Field(None, alias="dtMax")


class SpeciesEntry(BaseModel):
    mass: float
    charge: float
    injection: Literal["plasma", "beam", "uniform", "custom"]
    temperature_keV: Optional[float] = Field(None, alias="temperatureKeV")
    energy_distribution: Optional[Literal["mono", "thermal", "custom"]] = Field(
        None, alias="energyDistribution"
    )


class WarpXSettings(ConfigSectionBase):
    """Validated WarpX PIC solver configuration."""

    config_section_id: ClassVar[Literal["warpx"]] = "warpx"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    field_solver: Literal["Yee", "PSATD", "PSTD", "FDTD"] = Field(
        "Yee", alias="fieldSolver"
    )
    interpolation_order: int = Field(2, ge=1, le=5, alias="interpolationOrder")
    particle_shape: Literal["linear", "quadratic", "cubic"] = Field(
        "linear", alias="particleShape"
    )
    particle_shape_order: int = Field(1, ge=1, le=3, alias="particleShapeOrder")
    particle_push_algorithm: Literal["Boris", "Vay", "HigueraCary"] = Field(
        "Boris", alias="particlePushAlgorithm"
    )
    time_step_type: Literal["constant", "adaptive"] = Field(
        "constant", alias="timeStepType"
    )
    adaptive_time_step_config: Optional[AdaptiveTimeStep] = Field(
        None, alias="adaptiveTimeStepConfig"
    )

    ionization_model: Optional[Literal["ADK", "Keldysh", "None"]] = Field(
        "None", alias="ionizationModel"
    )
    collision_model: Optional[Literal["binary", "MonteCarlo", "None"]] = Field(
        "None", alias="collisionModel"
    )

    species_config: Dict[str, SpeciesEntry] = Field(
        default_factory=dict, alias="speciesConfig"
    )

    boundary_conditions: Optional[
        Dict[str, Dict[str, Literal["reflecting", "absorbing", "periodic"]]]
    ] = Field(None, alias="boundaryConditions")

    galilean_shift_velocity: Optional[Tuple[float, float, float]] = Field(
        None, alias="galileanShiftVelocity"
    )
    moving_window_velocity: Optional[Tuple[float, float, float]] = Field(
        None, alias="movingWindowVelocity"
    )
    emission_profile_path: Optional[Path] = Field(
        None, alias="emissionProfilePath"
    )

    field_deposition: Literal["standard", "Esirkepov"] = Field(
        "standard", alias="fieldDeposition"
    )
    current_correction: Optional[bool] = Field(True, alias="currentCorrection")
    current_smoothing_enabled: Optional[bool] = Field(
        False, alias="currentSmoothingEnabled"
    )
    current_smoothing_kernel: Optional[Tuple[int, int, int]] = Field(
        None, alias="currentSmoothingKernel"
    )

    max_particles_per_cell: Optional[int] = Field(
        None, ge=1, alias="maxParticlesPerCell"
    )
    warpx_config_hash: Optional[str] = Field(None, alias="warpxConfigHash")

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls, geometry: str) -> "WarpXSettings":
        data: Dict[str, Any] = {
            "field_solver": "Yee",
            "interpolation_order": 2,
            "particle_shape": "linear",
            "particle_shape_order": 1,
            "particle_push_algorithm": "Boris",
            "time_step_type": "constant",
            "ionization_model": "None",
            "collision_model": "None",
            "species_config": {},
            "boundary_conditions": None,
            "galilean_shift_velocity": None,
            "moving_window_velocity": None,
            "emission_profile_path": None,
            "field_deposition": "standard",
            "current_correction": True,
            "current_smoothing_enabled": False,
            "current_smoothing_kernel": None,
            "max_particles_per_cell": None,
        }
        return cls.model_validate(data, context={"geometry": geometry})

    def resolve_defaults(self, geometry: str) -> "WarpXSettings":
        data = self.model_dump()
        defaults = self.with_defaults(geometry).model_dump()
        defaults.update({k: v for k, v in data.items() if v is not None})
        return self.model_validate(defaults, context={"geometry": geometry})

    @classmethod
    def required_fields(cls) -> List[str]:
        return [name for name, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {
            name: (field.json_schema_extra or field.metadata or {})
            for name, field in cls.model_fields.items()
        }

    def normalize_units(self) -> "WarpXSettings":
        return self

    def summarize(self) -> str:
        solver = (
            f"WarpX: {self.field_solver} solver, order={self.interpolation_order}, "
            f"shape={self.particle_shape}, push={self.particle_push_algorithm}"
        )
        if self.time_step_type == "adaptive" and self.adaptive_time_step_config:
            adapt = (
                f"Adaptive timestep: CFL={self.adaptive_time_step_config.cfl}, "
                f"Ionization: {self.ionization_model}, Collisions: {self.collision_model}"
            )
        else:
            adapt = (
                f"Constant timestep, Ionization: {self.ionization_model}, Collisions: {self.collision_model}"
            )
        species_names = ", ".join(self.species_config.keys()) or "none"
        emission = (
            self.emission_profile_path.name if self.emission_profile_path else "none"
        )
        species_line = (
            f"Species: {species_names}, PPC = {self.max_particles_per_cell or 'n/a'}, "
            f"Emission profile = {emission}"
        )
        kernel = (
            f"{list(self.current_smoothing_kernel)}"
            if self.current_smoothing_enabled and self.current_smoothing_kernel
            else "none"
        )
        current_line = f"Current smoothing: kernel={kernel}"
        return "\n".join([solver, adapt, species_line, current_line])

    def hash_warpx_config(self) -> str:
        data = self.model_dump(exclude={"warpx_config_hash"}, by_alias=True, exclude_none=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @classmethod
    def check_rules(cls, values: "WarpXSettings") -> "WarpXSettings":
        ctx = getattr(values, "_context", {})
        geometry = ctx.get("geometry")
        gas_type = ctx.get("gas_type")

        if values.time_step_type == "adaptive" and values.adaptive_time_step_config is None:
            raise ValueError("adaptive_time_step_config required when time_step_type is 'adaptive'")

        if values.ionization_model != "None" and gas_type is not None:
            if gas_type not in {"D2", "DT", "He", "Ne", "Ar", "Xe"}:
                raise ValueError("ionization_model specified but gas type is not ionizable")

        if values.collision_model == "MonteCarlo" and values.max_particles_per_cell is not None:
            if values.max_particles_per_cell < 32:
                warnings.warn("MonteCarlo collisions typically require ≥ 32 particles per cell")

        if values.boundary_conditions:
            required_faces = {"xLow", "xHigh", "yLow", "yHigh", "zLow", "zHigh"}
            for spec, mapping in values.boundary_conditions.items():
                unknown = set(mapping) - required_faces
                if unknown:
                    warnings.warn(f"unrecognized boundary keys for {spec}: {sorted(unknown)}")
                missing = required_faces - set(mapping)
                if missing:
                    raise ValueError(f"boundary_conditions for {spec} missing faces: {sorted(missing)}")

        values = values.model_copy(update={"warpx_config_hash": values.hash_warpx_config()})
        return values

    # ------------------------------------------------------------------
    @classmethod
    def model_validate(cls, data: Any, **kwargs) -> "WarpXSettings":
        context = kwargs.get("context") or {}
        obj = super().model_validate(data)
        object.__setattr__(obj, "_context", context)
        obj = cls.check_rules(obj)
        object.__setattr__(obj, "_context", context)
        return obj


__all__ = ["WarpXSettings", "SpeciesEntry", "AdaptiveTimeStep"]
