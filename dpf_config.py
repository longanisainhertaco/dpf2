from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal, Union
from pydantic import BaseModel, Field, validator, root_validator

from experimental_variability import ExperimentalVariabilityModel

def model_validator(*, mode: str = "after"):
    """Compatibility helper mirroring pydantic.v2 model_validator."""

    def decorator(func):
        return root_validator(pre=(mode == "before"), skip_on_failure=True)(func)

    return decorator

if not hasattr(BaseModel, "model_validate"):
    BaseModel.model_validate = classmethod(lambda cls, d: cls.parse_obj(d))
if not hasattr(BaseModel, "model_dump"):
    BaseModel.model_dump = BaseModel.dict
if not hasattr(BaseModel, "model_dump_json"):
    BaseModel.model_dump_json = BaseModel.json
if not hasattr(BaseModel, "model_copy"):
    BaseModel.model_copy = BaseModel.copy

# --- Submodels ---

class SimulationControl(BaseModel):
    mode: Literal["fluid", "PIC", "hybrid"]
    geometry: Literal["2D_RZ", "3D_Cartesian"]
    time_start: float
    time_end: float
    max_steps: int
    min_dt: Optional[float] = None
    max_dt: Optional[float] = None
    stiffness_strategy: Optional[str] = None
    collapse_detect_dt_scale: Optional[float] = None
    spatial_units: Literal["cm", "m"] = "cm"
    dry_run: bool = False
    debug_output: bool = False
    validate_only: bool = False
    termination_conditions: Dict[str, float] = Field(default_factory=dict)
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            mode="fluid",
            geometry="2D_RZ",
            time_start=0.0,
            time_end=1.0,
            max_steps=1000,
        )

    @model_validator(mode="after")
    def check_times(cls, values):
        if values["time_end"] <= values["time_start"]:
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
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            nx=128, ny=1, nz=128,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )

class BreakdownModel(BaseModel):
    type: Literal["field_threshold", "hot_seed", "stochastic_delay", "beta_preionization"]
    field_threshold: Optional[float] = None
    breakdown_delay: Optional[float] = None
    stochastic_seed: Optional[int] = None
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(type="field_threshold")

class PaschenModel(BaseModel):
    insulator_gap_cm: float
    gas_pressure_torr: float
    material: str
    paschen_curve_model: Optional[str] = None
    paschen_data_path: Optional[Path] = None
    knee_voltage_override: Optional[float] = None
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(insulator_gap_cm=1.0, gas_pressure_torr=10.0, material="Pyrex")

class InitialConditions(BaseModel):
    temperature: float
    density: float
    sheath_velocity_profile: List[Tuple[float, float]] = Field(default_factory=list)
    current_profile: List[Tuple[float, float]] = Field(default_factory=list)
    gas_type: Literal["D2", "He", "Ne", "Ar", "Xe", "DT"]
    sheath_type: Literal["slab", "gaussian"]
    breakdown_model: BreakdownModel
    paschen_model: PaschenModel
    preionization_method: Optional[str] = None
    preionization_intensity: Optional[float] = None
    preionization_duration_ns: Optional[float] = None
    preionization_profile: Optional[List[Tuple[float, float]]] = None
    enable_dynamic_ionization_rate: bool = False
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            temperature=1.0,
            density=1.0,
            sheath_velocity_profile=[],
            current_profile=[],
            gas_type="D2",
            sheath_type="slab",
            breakdown_model=BreakdownModel.with_defaults(),
            paschen_model=PaschenModel.with_defaults(),
        )

class PhysicsModels(BaseModel):
    eos_model: str
    gamma: float
    two_temperature_model_enabled: bool = False
    neutral_fluid_enabled: bool = False
    initial_neutral_pressure_torr: Optional[float] = None
    enable_neutral_particle_tracking: bool = False
    resistivity_model: Optional[str] = None
    ionization_model: Optional[str] = None
    radiation_model: Optional[str] = None
    radiation_transport_model: Optional[str] = None
    line_escape_method: Optional[str] = None
    radiation_geometry_model: Optional[str] = None
    sxr_bandpass_nm: Optional[float] = None
    enable_photon_reabsorption: bool = False
    hall_mhd_enabled: bool = False
    pease_bragginski_limit_check: bool = False
    instability_models_enabled: bool = False
    instability_thresholds: Optional[Dict[str, float]] = None
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            eos_model="ideal_gas",
            gamma=1.4
        )

class CircuitConfig(BaseModel):
    L_ext: float
    R_ext: float
    C_ext: float
    V0: float
    switch_delay: float
    waveform_profile: List[Tuple[float, float]] = Field(default_factory=list)
    waveform_profile_path: Optional[Path] = None
    switching_model: Optional[str] = None
    trigger_jitter_stddev: Optional[float] = None
    abort_on_no_current: bool = False
    circuit_fault_flags: Optional[List[str]] = None
    log_dynamic_impedance: bool = False
    switch_feedback_delay_ns: Optional[float] = None
    enable_field_triggered_switch_closure: bool = False
    override_inductive_voltage_limit: Optional[float] = None
    inter_shot_recovery_time_ns: Optional[float] = None
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            L_ext=1e-6,
            R_ext=0.1,
            C_ext=1e-6,
            V0=20e3,
            switch_delay=0.0,
        )

class ElectrodeGeometry(BaseModel):
    cathode_type: str
    cathode_bar_count: int
    cathode_gap_degrees: float
    anode_shape: str
    knife_edge_enabled: bool = False
    emitter_field_enhancement: Optional[float] = None
    mesh_file: Optional[Path] = None
    material_tagging_enabled: bool = False

    @classmethod
    def with_defaults(cls):
        return cls(
            cathode_type="bar",
            cathode_bar_count=10,
            cathode_gap_degrees=36.0,
            anode_shape="cylinder"
        )

class AmrexSettings(BaseModel):
    amr_levels: int
    stencil_order: int
    integrator: Literal["RK4", "Godunov"] = "RK4"
    embedded_boundary: bool = False
    electrode_geometry: ElectrodeGeometry
    electrode_material: Optional[str] = None
    erosion_mechanisms_enabled: bool = False
    material_properties_override: Optional[Dict[str, Any]] = None
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            amr_levels=1,
            stencil_order=2,
            electrode_geometry=ElectrodeGeometry.with_defaults()
        )

class WarpXSettings(BaseModel):
    field_solver: str
    interpolation_order: int
    collision_model: str
    ionization_model: str
    particle_shape: str
    particle_shape_order: int
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            field_solver="PSATD",
            interpolation_order=2,
            collision_model="binary",
            ionization_model="field",
            particle_shape="cloud_in_cell",
            particle_shape_order=2
        )

class Diagnostics(BaseModel):
    output_dir: Path
    output_format: Literal["HDF5", "ADIOS2"] = "HDF5"
    plotfile_interval: int
    checkpoint_interval: int
    restart_from: Optional[Path] = None
    synthetic_neutron_yield: bool = False
    neutron_fluence_map: bool = False
    angular_neutron_spectrum_output: bool = False
    neutron_energy_bins_MeV: Optional[List[float]] = None
    enable_DT_yield_modeling: bool = False
    sxr_detector_models: List[Dict[str, Any]] = Field(default_factory=list)
    neutron_tof_detectors: List[Dict[str, Any]] = Field(default_factory=list)
    wall_probe_array_config: Optional[Dict[str, Any]] = None
    fields_to_output: List[str] = Field(default_factory=list)
    particle_species_to_track: List[str] = Field(default_factory=list)
    beam_diagnostics: Optional[Dict[str, Any]] = None
    beam_pulse_width_estimate_ns: Optional[float] = None
    enable_beam_convergence_tracking: bool = False
    enable_energy_tracking: bool = False
    enable_mhd_instability_tracking: bool = False
    enable_erosion_mapping: bool = False
    enable_failure_logging: bool = False
    enable_eedf_logging: bool = False
    enable_pitch_angle_tracking: bool = False
    angular_diagnostics_resolution_deg: Optional[float] = None
    unity_websocket: bool = False
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            output_dir=Path("./output"),
            plotfile_interval=10,
            checkpoint_interval=100
        )


class BenchmarkMatching(BaseModel):
    dataset_id: Optional[str] = None
    target_curves: List[str] = Field(default_factory=list)
    waveform_tolerance: Optional[float] = None
    compare_fields: List[str] = Field(default_factory=list)
    match_waveform_features: bool = False
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls()

class FaceType(str):
    pass

class BoundaryConditions(BaseModel):
    x_low: Literal["reflecting", "absorbing", "conducting"]
    x_high: Literal["reflecting", "absorbing", "conducting"]
    y_low: Literal["reflecting", "absorbing", "conducting"]
    y_high: Literal["reflecting", "absorbing", "conducting"]
    z_low: Literal["reflecting", "absorbing", "conducting"]
    z_high: Literal["reflecting", "absorbing", "conducting"]
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            x_low="reflecting", x_high="reflecting",
            y_low="reflecting", y_high="reflecting",
            z_low="reflecting", z_high="reflecting"
        )

class ParallelSettings(BaseModel):
    mpi_ranks: int
    gpu_backend: Literal["CUDA", "HIP", "None"]
    decomposition_strategy: Optional[str] = None
    load_balancing_strategy: Optional[str] = None
    amr_refinement_criteria: Optional[Dict[str, Any]] = None
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(mpi_ranks=1, gpu_backend="None")

class Metadata(BaseModel):
    schema_version: str
    sim_version: str
    created_by: str
    commit_hash: str
    run_uuid: Optional[str] = None
    creation_time: Optional[str] = None
    campaign_mode_enabled: bool = False
    ensemble_shot_configs: Optional[List[Dict[str, Any]]] = None
    ml_metadata: Optional[Dict[str, Any]] = None
    use_surrogate_model: bool = False
    yield_targeting_enabled: bool = False
    doc: Optional[str] = None

    @classmethod
    def with_defaults(cls):
        return cls(
            schema_version="1.0",
            sim_version="0.1",
            created_by="unknown",
            commit_hash="none"
        )

# --- Top-level config ---

class DPFConfig(BaseModel):
    simulation_control: SimulationControl
    grid_resolution: GridResolution
    initial_conditions: InitialConditions
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
    def validate_cross_fields(cls, values):
        sc = values.get("simulation_control")
        gr = values.get("grid_resolution")
        if sc and gr and sc.geometry == "2D_RZ" and gr.ny != 1:
            raise ValueError("ny must be 1 for 2D_RZ geometry")
        return values

    @classmethod
    def with_defaults(cls):
        return cls(
            simulation_control=SimulationControl.with_defaults(),
            grid_resolution=GridResolution.with_defaults(),
            initial_conditions=InitialConditions.with_defaults(),
            physics_models=PhysicsModels.with_defaults(),
            circuit_config=CircuitConfig.with_defaults(),
            amrex_settings=AmrexSettings.with_defaults(),
            warpx_settings=WarpXSettings.with_defaults(),
            diagnostics=Diagnostics.with_defaults(),
            experimental_variability=ExperimentalVariabilityModel.with_defaults(),
            benchmark_matching=BenchmarkMatching.with_defaults(),
            boundary_conditions=BoundaryConditions.with_defaults(),
            parallel_settings=ParallelSettings.with_defaults(),
            metadata=Metadata.with_defaults(),
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "DPFConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        if p.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except Exception as e:
                raise ImportError("PyYAML is required for YAML files") from e
            data = yaml.safe_load(p.read_text())
        else:
            data = json.loads(p.read_text())
        return cls.model_validate(data)

    def override(self, **updates: Any) -> "DPFConfig":
        data = self.model_dump()
        data.update(updates)
        return self.__class__.model_validate(data)

    def to_file(self, path: Union[str, Path], format: Literal["json", "yaml"] = "json") -> None:
        if format == "json":
            self.to_json(path)
        elif format == "yaml":
            self.to_yaml(path)
        else:
            raise ValueError("format must be 'json' or 'yaml'")

    @classmethod
    def validate_and_fill(cls, raw: Union[str, Path, Dict[str, Any]]):
        if isinstance(raw, (str, Path)):
            return cls.from_file(raw)
        return cls.model_validate(raw)

    def resolve_defaults(self) -> "DPFConfig":
        defaults = self.with_defaults().model_dump()
        defaults.update(self.model_dump())
        return self.__class__.model_validate(defaults)

    def validate(self) -> None:
        _ = self.model_validate(self.model_dump())

    def unit_scale(self) -> float:
        return 0.01 if self.simulation_control.spatial_units == "cm" else 1.0

    def duration(self) -> float:
        return self.simulation_control.time_end - self.simulation_control.time_start

    def schema_export(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            if hasattr(self, "model_json_schema"):
                schema = self.model_json_schema()
            else:
                schema = self.schema()
            json.dump(schema, f, indent=2)

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.model_dump_json())

    def to_yaml(self, path: Union[str, Path]) -> None:
        try:
            import yaml
        except Exception as e:
            raise ImportError("PyYAML required") from e
        Path(path).write_text(yaml.safe_dump(self.model_dump()))

    def summarize(self) -> str:
        return (
            "DPF Simulation Configuration Summary:"
            f"\n  Mode: {self.simulation_control.mode}, Geometry: {self.simulation_control.geometry}"
        )

    def required_fields(self) -> List[str]:
        return []

__all__ = [
    "SimulationControl",
    "GridResolution",
    "BreakdownModel",
    "PaschenModel",
    "InitialConditions",
    "PhysicsModels",
    "CircuitConfig",
    "ElectrodeGeometry",
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
