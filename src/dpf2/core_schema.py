"""Core configuration schemas for DPF simulations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Self,
    TYPE_CHECKING,
)

import logging
from enum import Enum


try:  # pragma: no cover - allow operation without pydantic
    from pydantic import BaseModel, ConfigDict, Field, root_validator
except Exception:  # pragma: no cover
    from pydantic_stub import BaseModel, ConfigDict, Field, root_validator  # type: ignore



logger = logging.getLogger(__name__)

def model_validator(*, mode: str = "after"):
    """Compatibility helper implementing a subset of pydantic v2 behavior."""

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
    if hasattr(BaseModel, "dict"):
        BaseModel.model_dump = BaseModel.dict
    else:  # pragma: no cover - stub behaviour
        BaseModel.model_dump = lambda self, *_, **__: self.__dict__
if not hasattr(BaseModel, "model_dump_json"):
    if hasattr(BaseModel, "json"):
        BaseModel.model_dump_json = BaseModel.json
    else:  # pragma: no cover - stub behaviour
        import json as _json

        BaseModel.model_dump_json = lambda self, *_, **__: _json.dumps(self.__dict__)
if True:  # replace ``model_copy`` with lightweight implementation
    def _copy(self, update=None, **__):
        new = self.__class__()
        for k, v in self.__dict__.items():
            setattr(new, k, v)
        if update:
            for k, v in update.items():
                setattr(new, k, v)
        return new

    BaseModel.model_copy = _copy

# ---------------------------------------------------------------------------
# Type aliases
TimeVoltageProfile = List[Tuple[float, float]]
DetectorConfig = Dict[str, Any]
ConfigOverride = Dict[str, Union[float, str]]
FieldUnit = str

if TYPE_CHECKING:
    from .simulation_settings import SimulationSettings
    from .grid_resolution import GridResolution
    from .initial_conditions import InitialConditions
    from .physics_models import PhysicsModels
    from .circuit_config import CircuitConfig
    from .amrex_settings import AmrexSettings
    from .warpx_settings import WarpXSettings
    from dpf2.diagnostics import Diagnostics
    from .experimental_variability import ExperimentalVariabilityModel
    from .benchmark_matching import BenchmarkMatching
    from .boundary_conditions import BoundaryConditions
    from .parallel_settings import ParallelSettings
    from .metadata import Metadata
    from .advanced_options import AdvancedOptions
    from .units_settings import UnitsSettings

# ---------------------------------------------------------------------------
# Helper for camelCase aliasing

def to_camel_case(string: str) -> str:
    parts = string.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

# ---------------------------------------------------------------------------
# Enums

class GeometryType(str, Enum):
    """Defines valid simulation geometries."""

    RZ_2D = "2D_RZ"
    XYZ_3D = "3D_Cartesian"

class ModeType(str, Enum):
    """Defines valid DPF solver modes."""

    FLUID = "fluid"
    PIC = "PIC"
    HYBRID = "hybrid"

class UnitsSystem(str, Enum):
    """Defines the base unit system used for internal normalization."""

    SI = "SI"
    CGS = "cgs"
    LAB = "lab"

class ValidationPolicy(str, Enum):
    """Controls schema enforcement mode on load/override."""

    STRICT = "strict"
    WARN = "warn"
    SILENT = "silent"

# ---------------------------------------------------------------------------
# Physics-related enums

class EOSModel(str, Enum):
    IDEAL = "ideal"
    TABULATED = "tabulated"
    REAL_GAS = "real_gas"


class ResistivityModel(str, Enum):
    SPITZER = "Spitzer"
    LHDI = "LHDI"
    ANOMALOUS = "anomalous"


class IonizationModel(str, Enum):
    SAHA = "Saha"
    FLYCHK = "FLYCHK"
    NONE = "None"


class IonizationFallback(str, Enum):
    RAISE = "raise"
    WARN = "warn"
    SWITCH_TO_SAHA = "switch_to_Saha"


class RadiationModel(str, Enum):
    BREMSSTRAHLUNG = "Bremsstrahlung"
    LINE = "Line"
    NONE = "None"


class RadiationTransportModel(str, Enum):
    ESCAPE_FACTOR = "EscapeFactor"
    MONTE_CARLO = "MonteCarlo"
    NONE = "None"


class LineEscapeMethod(str, Enum):
    EDDINGTON = "Eddington"
    ROSSELAND = "Rosseland"
    SCHUSTER = "Schuster"


class RadiationGeometryModel(str, Enum):
    SLAB = "slab"
    SPHERICAL = "spherical"


class InstabilityModel(str, Enum):
    KINK = "kink"
    SAUSAGE = "sausage"
    RT = "RT"
    RESISTIVE_MHD = "resistive_MHD"
    HALL_MHD = "Hall_MHD"


class CircuitFaultTypeEnum(str, Enum):
    """Enumerates possible circuit fault types."""

    ARC = "arc"
    TIMEOUT = "timeout"
    NO_DISCHARGE = "no_discharge"
    EARLY_TRIGGER = "early_trigger"

# ---------------------------------------------------------------------------
# Unit normalization helpers

UNIT_SCALE_MAP: Dict[UnitsSystem, float] = {
    UnitsSystem.SI: 1.0,
    UnitsSystem.CGS: 0.01,
    UnitsSystem.LAB: 1.0,
}

# ---------------------------------------------------------------------------
# Base configuration section

class ConfigSectionBase(BaseModel):
    """Base class for all configuration sections."""

    title: Optional[str] = Field(
        default=None,
        alias="title",
        metadata={"units": "-", "category": "Metadata", "group": "General"},
    )
    description: Optional[str] = Field(
        default=None,
        alias="description",
        metadata={"units": "-", "category": "Metadata", "group": "General"},
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        alias="tags",
        metadata={"units": "-", "category": "Metadata", "group": "General"},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        alias="metadata",
        metadata={"units": "-", "category": "Metadata", "group": "General"},
    )

    config_section_id: ClassVar[str]

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        allow_population_by_field_name=True,
        validate_default=True,
        frozen=True,
    )


    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> Self:
        return cls()  # type: ignore[call-arg]

    def required_fields(self) -> List[str]:
        return [name for name, field in self.model_fields.items() if field.is_required()]

    def resolve_defaults(self) -> Self:
        data = self.model_dump()
        return self.model_validate(data)

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: field.json_schema_extra or {}
            for name, field in self.model_fields.items()
        }


# ---------------------------------------------------------------------------
# Radiation configuration


class RadiationSettings(ConfigSectionBase):
    """Basic radiation configuration including group counts and opacities."""

    config_section_id: ClassVar[str] = "radiation"

    group_count: int = Field(
        1,
        alias="groupCount",
        metadata={
            "units": "-",
            "category": "Radiation",
            "group": "MultiGroup",
        },
    )

    group_opacities: List[float] = Field(
        default_factory=list,
        alias="groupOpacities",
        metadata={
            "units": "1/m",
            "category": "Radiation",
            "group": "MultiGroup",
        },
    )

    material_opacities: Dict[str, List[float]] = Field(
        default_factory=dict,
        alias="materialOpacities",
        metadata={
            "units": "1/m",
            "category": "Radiation",
            "group": "MultiGroup",
        },
    )

    @model_validator(mode="after")
    def _validate_groups(cls, values: Self) -> Self:
        """Ensure opacity lists match the configured group count."""
        if values.group_opacities and len(values.group_opacities) != values.group_count:
            raise ValueError("group_opacities must match group_count")
        for mat, vals in values.material_opacities.items():
            if len(vals) != values.group_count:
                raise ValueError(
                    f"material_opacities for {mat} must match group_count"
                )
        return values

    def opacity_for(self, material: str | None = None) -> List[float]:
        """Return opacities for ``material`` or default group values."""

        if material and material in self.material_opacities:
            return self.material_opacities[material]
        return self.group_opacities
# ---------------------------------------------------------------------------
# Root configuration model

class DPFConfig(BaseModel):
    """Root configuration object."""

    # Dedicated section models
    simulation: SimulationSettings = Field(
        ..., alias="simulation", metadata={"units": "-", "category": "Model", "group": "Simulation"}
    )
    grid: GridResolution = Field(
        ..., alias="grid", metadata={"units": "-", "category": "Model", "group": "Grid"}
    )
    initial: InitialConditions = Field(
        ..., alias="initial", metadata={"units": "-", "category": "Model", "group": "Initial"}
    )
    physics: PhysicsModels = Field(
        ..., alias="physics", metadata={"units": "-", "category": "Model", "group": "Physics"}
    )
    radiation: RadiationSettings = Field(
        default_factory=RadiationSettings.with_defaults,
        alias="radiation",
        metadata={"units": "-", "category": "Model", "group": "Radiation"},
    )
    circuit: CircuitConfig = Field(
        ..., alias="circuit", metadata={"units": "-", "category": "Model", "group": "Circuit"}
    )
    amrex: AmrexSettings = Field(
        ..., alias="amrex", metadata={"units": "-", "category": "Model", "group": "AMReX"}
    )
    warpx: WarpXSettings = Field(
        ..., alias="warpx", metadata={"units": "-", "category": "Model", "group": "WarpX"}
    )
    diagnostics: Diagnostics = Field(
        ..., alias="diagnostics", metadata={"units": "-", "category": "Model", "group": "Diagnostics"}
    )
    variability: ExperimentalVariabilityModel = Field(
        ..., alias="variability", metadata={"units": "-", "category": "Model", "group": "Variability"}
    )
    benchmark: BenchmarkMatching = Field(
        ..., alias="benchmark", metadata={"units": "-", "category": "Model", "group": "Benchmark"}
    )
    boundary: BoundaryConditions = Field(
        ..., alias="boundary", metadata={"units": "-", "category": "Model", "group": "Boundary"}
    )
    parallel: ParallelSettings = Field(
        ..., alias="parallel", metadata={"units": "-", "category": "Model", "group": "Parallel"}
    )
    metadata: Metadata = Field(
        ..., alias="metadata", metadata={"units": "-", "category": "Model", "group": "Metadata"}
    )
    advanced: AdvancedOptions = Field(
        ..., alias="advanced", metadata={"units": "-", "category": "Model", "group": "Advanced"}
    )
    units: UnitsSettings = Field(
        ..., alias="units", metadata={"units": "-", "category": "Model", "group": "Units"}
    )

    run_uuid: str = Field(
        ..., alias="runUuid", metadata={"units": "-", "category": "Audit", "group": "Global"}
    )
    schema_version: str = Field(
        ..., alias="schemaVersion", metadata={"units": "-", "category": "Audit", "group": "Global"}
    )
    created_at: datetime = Field(
        ..., alias="createdAt", metadata={"units": "datetime", "category": "Audit", "group": "Global"}
    )
    config_hash: Optional[str] = Field(
        default=None, alias="configHash", metadata={"units": "-", "category": "Audit", "group": "Global"}
    )
    restart_hash: Optional[str] = Field(
        default=None, alias="restartHash", metadata={"units": "-", "category": "Audit", "group": "Global"}
    )
    run_lineage: Optional[List[str]] = Field(
        default=None, alias="runLineage", metadata={"units": "-", "category": "Audit", "group": "Global"}
    )

    on_validation_error: ValidationPolicy = Field(
        default=ValidationPolicy.STRICT,
        alias="onValidationError",
        metadata={"units": "-", "category": "Validation", "group": "Global"},
    )
    validation_output_path: Optional[Path] = Field(
        default=None,
        alias="validationOutputPath",
        metadata={"units": "path", "category": "Validation", "group": "Global"},
    )

    schema_migrations: ClassVar[Dict[str, Callable]] = {}

    model_config = ConfigDict(
        extra="allow",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> Self:
        from .simulation_settings import SimulationSettings
        from .grid_resolution import GridResolution
        from .initial_conditions import InitialConditions
        from .physics_models import PhysicsModels
        from .circuit_config import CircuitConfig
        from .amrex_settings import AmrexSettings
        from .warpx_settings import WarpXSettings
        from dpf2.diagnostics import Diagnostics
        from .experimental_variability import ExperimentalVariabilityModel
        from .benchmark_matching import BenchmarkMatching
        from .boundary_conditions import BoundaryConditions
        from .parallel_settings import ParallelSettings
        from .metadata import Metadata
        from .advanced_options import AdvancedOptions
        from .units_settings import UnitsSettings

        cls.model_rebuild()
        sim = SimulationSettings.with_defaults()
        return cls(
            simulation=sim,
            grid=GridResolution.with_defaults(
                sim.geometry.value if isinstance(sim.geometry, GeometryType) else sim.geometry
            ),
            initial=InitialConditions.with_defaults(),
            physics=PhysicsModels.with_defaults(),
            radiation=RadiationSettings.with_defaults(),
            circuit=CircuitConfig.with_defaults(),
            amrex=AmrexSettings.with_defaults(),
            warpx=WarpXSettings.with_defaults(
                sim.geometry.value if isinstance(sim.geometry, GeometryType) else sim.geometry
            ),
            diagnostics=Diagnostics.with_defaults(),
            variability=ExperimentalVariabilityModel.with_defaults(),
            benchmark=BenchmarkMatching.with_defaults(),
            boundary=BoundaryConditions.with_defaults(),
            parallel=ParallelSettings.with_defaults(),
            metadata=Metadata.with_defaults(),
            advanced=AdvancedOptions.with_defaults(),
            units=UnitsSettings.with_defaults(),
            run_uuid="0",
            schema_version="1.0",
            created_at=datetime.utcnow(),
            on_validation_error=ValidationPolicy.STRICT,
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Self:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        if p.suffix in {".yaml", ".yml"}:
            import yaml
            data = yaml.safe_load(p.read_text())
        else:
            import json
            data = json.loads(p.read_text())
        return cls.model_validate(data)

    def to_file(self, path: Union[str, Path], format: Union[str, None] = "json") -> None:
        p = Path(path)
        if format == "yaml":
            import yaml
            p.write_text(yaml.safe_dump(self.model_dump(by_alias=True)))
        else:
            import json
            p.write_text(json.dumps(self.model_dump(by_alias=True), indent=2))

    def override(self, key_path: str, value: Any) -> Self:
        parts = key_path.split(".")
        data = self.model_dump()
        ref = data
        for p in parts[:-1]:
            ref = ref.setdefault(p, {})
        ref[parts[-1]] = value
        return self.model_validate(data)

    def validate(self) -> Self:
        return self.model_validate(self.model_dump())

    def unit_scale(self) -> float:
        return UNIT_SCALE_MAP.get(UnitsSystem.SI, 1.0)

    def duration(self) -> float:
        sim = self.simulation
        start = getattr(sim, "time_start", 0.0)
        end = getattr(sim, "time_end", 0.0)
        return end - start

    def schema_export(self, path: Union[str, Path], format: str = "json") -> None:
        p = Path(path)
        if format == "yaml":
            import yaml
            p.write_text(yaml.safe_dump(self.model_json_schema()))
        else:
            import json
            p.write_text(json.dumps(self.model_json_schema(), indent=2))

    def normalize_units(self, base_units: UnitsSystem) -> None:
        scale = UNIT_SCALE_MAP.get(base_units, 1.0)
        if hasattr(self.simulation, "time_start") and hasattr(self.simulation, "time_end"):
            start = getattr(self.simulation, "time_start")
            end = getattr(self.simulation, "time_end")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                self.simulation = self.simulation.model_copy(update={
                    "time_start": start * scale,
                    "time_end": end * scale,
                })

    def summarize(self) -> str:
        return (
            f"DPFConfig(version={self.schema_version}, run_uuid={self.run_uuid}, "
            f"created_at={self.created_at.isoformat()})"
        )

    @classmethod
    def validate_and_fill(cls, raw: Union[str, Path, Dict[str, Any]]) -> Self:
        if isinstance(raw, (str, Path)):
            return cls.from_file(raw)
        return cls.model_validate(raw)

    def resolve_defaults(self) -> Self:
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [name for name, field in self.model_fields.items() if field.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: field.json_schema_extra or {}
            for name, field in self.model_fields.items()
        }

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_global_cross_dependencies(cls, values: Self) -> Self:
        sim = values.simulation
        grid = values.grid
        physics = values.physics
        diag = values.diagnostics
        if getattr(sim, "geometry", None) == GeometryType.RZ_2D:
            if getattr(grid, "ny", 1) != 1:
                raise ValueError("ny must be 1 for 2D_RZ geometry")
        if getattr(sim, "mode", None) == ModeType.PIC:
            if not getattr(physics, "neutral_fluid_enabled", False):
                raise ValueError("neutral_fluid_enabled must be True in PIC mode")
        if getattr(diag, "neutron_fluence_map", False):
            gas_type = getattr(values.initial, "gas_type", "")
            if gas_type not in {"D2", "DT"}:
                raise ValueError("neutron diagnostics require D2 or DT gas")
        if getattr(values.circuit, "switching_model", "") == "multi-bank":
            if getattr(values.circuit, "switch_feedback_delay_ns", None) is None:
                raise ValueError("switch_feedback_delay_ns is required for multi-bank model")
        return values

# Attempt to resolve forward references now that section models may be available
try:  # pragma: no cover - optional import for forward refs
    from .simulation_settings import SimulationSettings  # type: ignore
    from .grid_resolution import GridResolution  # type: ignore
    from .initial_conditions import InitialConditions  # type: ignore
    from .physics_models import PhysicsModels  # type: ignore
    from .circuit_config import CircuitConfig  # type: ignore
    from .amrex_settings import AmrexSettings  # type: ignore
    from .warpx_settings import WarpXSettings  # type: ignore
    from dpf2.diagnostics import Diagnostics  # type: ignore
    from .experimental_variability import ExperimentalVariabilityModel  # type: ignore
    from .benchmark_matching import BenchmarkMatching  # type: ignore
    from .boundary_conditions import BoundaryConditions  # type: ignore
    from .parallel_settings import ParallelSettings  # type: ignore
    from .metadata import Metadata  # type: ignore
    from .advanced_options import AdvancedOptions  # type: ignore
    from .units_settings import UnitsSettings  # type: ignore
    DPFConfig.model_rebuild()
except Exception:  # pragma: no cover - missing optional deps
    logger.debug("DPFConfig forward reference resolution skipped", exc_info=True)

# ---------------------------------------------------------------------------
# Example usage

if __name__ == "__main__":
    cfg = DPFConfig.from_file("example_config.yml")
    cfg.validate()
    cfg.normalize_units(UnitsSystem.SI)
    print(cfg.summarize())
    cfg.schema_export("dpf_schema.json")

# ---------------------------------------------------------------------------
__all__ = [
    "ConfigSectionBase",
    "DPFConfig",
    "RadiationSettings",
    "TimeVoltageProfile",
    "DetectorConfig",
    "ConfigOverride",
    "FieldUnit",
    "GeometryType",
    "ModeType",
    "UnitsSystem",
    "ValidationPolicy",
    "EOSModel",
    "ResistivityModel",
    "IonizationModel",
    "IonizationFallback",
    "RadiationModel",
    "RadiationTransportModel",
    "LineEscapeMethod",
    "RadiationGeometryModel",
    "InstabilityModel",
    "CircuitFaultTypeEnum",
]
