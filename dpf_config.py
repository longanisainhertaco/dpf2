from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal, Any

from pydantic import BaseModel, Field, model_validator


class SimulationControl(BaseModel):
    """General run parameters."""

    mode: Literal["fluid", "PIC", "hybrid"]
    geometry: Literal["2D_RZ", "3D_Cartesian"]
    time_start: float
    time_end: float
    max_steps: int
    spatial_units: Literal["cm", "m"] = "cm"
    validate_only: bool = False
    dry_run: bool = False
    debug_output: bool = False
    termination_conditions: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_times(cls, values: "SimulationControl") -> "SimulationControl":
        if values.time_end <= values.time_start:
            raise ValueError("time_end must be greater than time_start")
        return values


class GridResolution(BaseModel):
    nx: int
    ny: int
    nz: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


class BreakdownModel(BaseModel):
    type: Literal[
        "field_threshold",
        "hot_seed",
        "stochastic_delay",
        "beta_preionization",
    ]
    field_threshold: Optional[float] = None
    breakdown_delay: Optional[float] = None
    stochastic_seed: Optional[int] = None


class PaschenModel(BaseModel):
    insulator_gap_cm: float
    gas_pressure_torr: float
    material: str
    knee_voltage_override: Optional[float] = None


class InitialConditions(BaseModel):
    temperature: float
    density: float
    sheath_velocity_profile: List[Tuple[float, float]] = Field(default_factory=list)
    current_profile: List[Tuple[float, float]] = Field(default_factory=list)
    sheath_type: Literal["slab", "gaussian"]
    gas_type: Literal["D2", "He", "Ne", "Ar", "Xe", "DT"]
    breakdown_model: BreakdownModel
    paschen_model: PaschenModel


class TimeControl(BaseModel):
    t_max: float
    dt_initial: float
    cfl_number: float


class PhysicsModels(BaseModel):
    eos_model: str
    gamma: float
    resistivity_model: Literal["Spitzer", "anomalous", "LHDI"]
    hall_mhd_enabled: bool = False
    radiation_model: str
    ionization_model: str
    optical_depth_model: str
    pease_bragginski_limit_check: bool = False


class CircuitConfig(BaseModel):
    L_ext: float
    R_ext: float
    C_ext: float
    V0: float
    switch_delay: float
    switching_model: Literal["ideal", "jittered", "multi-bank"] = "ideal"
    waveform_profile: List[Tuple[float, float]] = Field(default_factory=list)
    trigger_jitter_stddev: Optional[float] = None


class ElectrodeGeometry(BaseModel):
    cathode_type: str
    cathode_bar_count: int
    cathode_gap_degrees: float
    anode_shape: str
    knife_edge_enabled: bool = False
    emitter_field_enhancement: Optional[float] = None
    mesh_file: Optional[Path] = None
    material_tagging_enabled: bool = False


class AmrexSettings(BaseModel):
    amr_levels: int
    stencil_order: int
    embedded_boundary: bool = False
    integrator: Literal["RK4", "Godunov"] = "RK4"
    electrode_geometry: ElectrodeGeometry


class WarpXSettings(BaseModel):
    field_solver: str
    interpolation_order: int
    collision_model: str
    ionization_model: str
    particle_shape: str
    particle_shape_order: int


class Diagnostics(BaseModel):
    plotfile_interval: int
    checkpoint_interval: int
    output_format: Literal["HDF5", "ADIOS2"] = "HDF5"
    output_dir: Path
    restart_from: Optional[Path] = None
    synthetic_neutron_yield: bool = False
    neutron_fluence_map: bool = False
    angular_neutron_spectrum_output: bool = False
    neutron_tof_detectors: List[Dict[str, Any]] = Field(default_factory=list)
    sxr_detector_models: List[Dict[str, Any]] = Field(default_factory=list)
    fields_to_output: List[str] = Field(default_factory=list)
    particle_species_to_track: List[str] = Field(default_factory=list)
    particle_diagnostics_interval: int = 0
    enable_energy_tracking: bool = False
    enable_pinch_radius_tracking: bool = False
    enable_mhd_instability_tracking: bool = False
    enable_electrode_erosion_tracking: bool = False
    enable_reproducibility_statistics: bool = False
    enable_energy_budget_reporting: bool = False
    enable_knife_edge_effects: bool = False
    unity_websocket: bool = False


class ExperimentalVariabilityModel(BaseModel):
    pressure_jitter_pct: float = 0.0
    trigger_jitter_ns: float = 0.0
    emission_area_variation_pct: float = 0.0
    surface_degradation_factor: float = 1.0
    stochastic_run_id: Optional[int] = None


class BenchmarkMatching(BaseModel):
    dataset_id: Optional[str] = None
    target_curves: List[str] = Field(default_factory=list)
    tolerance: float = 0.0
    compare_fields: List[str] = Field(default_factory=list)
    match_waveform_features: bool = False


class BoundaryConditions(BaseModel):
    x_low: Literal["reflecting", "conducting", "absorbing"]
    x_high: Literal["reflecting", "conducting", "absorbing"]
    y_low: Literal["reflecting", "conducting", "absorbing"]
    y_high: Literal["reflecting", "conducting", "absorbing"]
    z_low: Literal["reflecting", "conducting", "absorbing"]
    z_high: Literal["reflecting", "conducting", "absorbing"]


class ParallelSettings(BaseModel):
    mpi_ranks: int
    gpu_backend: Literal["CUDA", "HIP", "None"]
    decomposition_strategy: str


class Metadata(BaseModel):
    schema_version: str
    sim_version: str
    created_by: str
    commit_hash: str


class DPFConfig(BaseModel):
    simulation_control: SimulationControl
    grid_resolution: GridResolution
    initial_conditions: InitialConditions
    time_control: TimeControl
    physics_models: PhysicsModels
    circuit_config: CircuitConfig
    amrex_settings: AmrexSettings
    warpx_settings: WarpXSettings
    diagnostics: Diagnostics
    experimental_variability: ExperimentalVariabilityModel
    benchmark_matching: BenchmarkMatching
    boundary_conditions: BoundaryConditions
    parallel_settings: ParallelSettings
    metadata: Metadata

    @model_validator(mode="after")
    def validate_config(cls, values: "DPFConfig") -> "DPFConfig":
        if (
            values.simulation_control.geometry == "2D_RZ"
            and values.grid_resolution.ny != 1
        ):
            raise ValueError("ny must be 1 for 2D_RZ geometry")
        return values

    def unit_scale(self) -> float:
        return 0.01 if self.simulation_control.spatial_units == "cm" else 1.0

    def duration(self) -> float:
        return self.simulation_control.time_end - self.simulation_control.time_start

    @classmethod
    def from_file(cls, path: str) -> "DPFConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        if p.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except Exception as e:  # pragma: no cover - dependency optional
                raise ImportError("PyYAML is required for YAML files") from e
            data = yaml.safe_load(p.read_text())
        else:
            import json

            data = json.loads(p.read_text())
        return cls.model_validate(data)

    def override(self, **updates: Any) -> "DPFConfig":
        data = self.model_dump()
        data.update(updates)
        return self.__class__.model_validate(data)

    def validate(self) -> None:
        _ = self.model_validate(self.model_dump())


__all__ = [
    "SimulationControl",
    "GridResolution",
    "InitialConditions",
    "TimeControl",
    "PhysicsModels",
    "CircuitConfig",
    "AmrexSettings",
    "WarpXSettings",
    "Diagnostics",
    "ExperimentalVariabilityModel",
    "BenchmarkMatching",
    "BoundaryConditions",
    "ParallelSettings",
    "Metadata",
    "DPFConfig",
]
